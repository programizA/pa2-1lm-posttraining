"""Task 2, Step 2: clipping study.

Part A (cached batch, no training): rebuild the supplied rollout batch, compute KL-shaped rewards,
GAE advantages and returns, and the importance ratio of the supplied PPO midpoint policy against
the cached old-policy log-probs. For eps in {0.05, 0.20, 0.50} report the clipped surrogate, the
clip fraction (ratio outside [1-eps, 1+eps]) and the affected-token fraction (tokens where the
clipped branch is selected, i.e. the token's gradient is zeroed), under the same response mask.
Then, from identical starting weights, take `ppo_epochs` policy steps on this batch for each eps to
show how quickly ratios reach the clip boundary.

Part B (forks): matched short continuations from the same midpoint for each eps (identical prompts,
sampling seeds, reward/KL settings, update and generation budgets), then the common held-out
evaluation and a stability summary from the training logs.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import load_json, save_json
from common.metrics import masked_mean
from common.models import load_policy, load_reward_model, load_tokenizer, load_value_model, token_values, trainable_parameters
from common.rl_utils import disable_dropout, model_device, row_id, run_module, token_logprobs
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _as_1d(x):
    t = torch.as_tensor(x).float().flatten()
    return t


def rebuild_batch(rows, cfg, tok, prompt_pool):
    """Turn cached rows into padded tensors: sequences, attention, response ids/mask, old/ref logp."""
    by_id = {}
    for i, r in enumerate(prompt_pool):
        by_id.setdefault(row_id(r, i), r)
        if "source_index" in r:
            by_id.setdefault(str(r["source_index"]), r)

    items, notes = [], {"length_mismatch": 0, "prompt_from_pool": 0}
    for row in rows:
        msgs = _first(row, ["messages", "prompt_messages"])
        if msgs is None:
            p = _first(row, ["prompt", "query"])
            if isinstance(p, list):
                msgs = p
            elif isinstance(p, str):
                msgs = [{"role": "user", "content": p}]
        if msgs is None:
            pr = by_id.get(str(row["source_index"]))
            if pr is None:
                raise KeyError(f"Cannot recover prompt for source_index={row['source_index']}")
            msgs = prompt_messages(pr)
            notes["prompt_from_pool"] += 1
        prompt_ids = _first(row, ["query_ids", "prompt_ids", "query_token_ids"])
        if prompt_ids is None:
            prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        prompt_ids = [int(x) for x in _as_1d(prompt_ids).long().tolist()]
        resp_ids = _first(row, ["response_ids", "response_token_ids", "completion_ids"])
        old = _as_1d(row["old_logprobs"])
        if resp_ids is None:
            resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
            if len(resp_ids) == len(old) - 1:  # cached rollouts include the terminating EOS
                resp_ids = resp_ids + [tok.eos_token_id]
        resp_ids = [int(x) for x in _as_1d(resp_ids).long().tolist()]
        T = min(len(resp_ids), len(old))
        if len(resp_ids) != len(old):
            notes["length_mismatch"] += 1
        items.append({
            "msgs": msgs, "prompt_ids": prompt_ids, "resp_ids": resp_ids[:T],
            "old": old[:T], "ref": _as_1d(row["ref_logprobs"])[:T],
            "reward": _first(row, ["reward", "score", "rm_score", "task_reward"]),
            "values": _first(row, ["values", "old_values", "value"]),
            "response": row["response"], "source_index": str(row["source_index"]),
        })

    P = max(len(it["prompt_ids"]) for it in items)
    T = max(len(it["resp_ids"]) for it in items)
    pad = tok.pad_token_id
    B = len(items)
    seq = torch.full((B, P + T), pad, dtype=torch.long)
    attn = torch.zeros((B, P + T), dtype=torch.long)
    resp = torch.full((B, T), pad, dtype=torch.long)
    mask = torch.zeros((B, T))
    old = torch.zeros((B, T))
    ref = torch.zeros((B, T))
    for b, it in enumerate(items):
        p, t = len(it["prompt_ids"]), len(it["resp_ids"])
        seq[b, P - p : P] = torch.tensor(it["prompt_ids"])  # left-pad prompt
        attn[b, P - p : P + t] = 1
        seq[b, P : P + t] = torch.tensor(it["resp_ids"])
        resp[b, :t] = torch.tensor(it["resp_ids"])
        mask[b, :t] = 1.0
        old[b, :t] = it["old"]
        ref[b, :t] = it["ref"]
    return {"items": items, "seq": seq, "attn": attn, "pw": P, "resp": resp, "mask": mask, "old": old, "ref": ref, "notes": notes}


def cached_batch_study(cfg, args):
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows), "| keys:", sorted(rows[0].keys()))
    tok = load_tokenizer(cfg["base_model"])
    batch = rebuild_batch(rows, cfg, tok, read_jsonl(cfg["paths"]["rl_prompt_train"]))
    print("rebuild notes:", batch["notes"])

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    dev = model_device(policy)
    seq, attn, resp = batch["seq"].to(dev), batch["attn"].to(dev), batch["resp"].to(dev)
    mask, old, ref, pw = batch["mask"].to(dev), batch["old"].to(dev), batch["ref"].to(dev), batch["pw"]
    T = resp.shape[1]

    # Rewards: cached if present, else re-scored with the fixed reward model (+ missing-EOS penalty).
    if all(it["reward"] is not None for it in batch["items"]):
        task_reward = torch.tensor([float(_as_1d(it["reward"]).sum()) for it in batch["items"]], device=dev)
        reward_source = "cache"
    else:
        rm, rm_tok = load_reward_model(cfg)
        task_reward = score_reward_pairs(rm, rm_tok, [it["msgs"] for it in batch["items"]],
                                         [it["response"] for it in batch["items"]], int(cfg["reward_max_length"])).to(dev)
        eos = torch.tensor([float(it["resp_ids"][-1] == tok.eos_token_id) for it in batch["items"]], device=dev)
        task_reward = task_reward - float(cfg["missing_eos_penalty"]) * (1.0 - eos)
        reward_source = "reward_model"
        del rm
    # Values: cached if present, else from the supplied midpoint critic.
    if all(it["values"] is not None for it in batch["items"]):
        values = torch.zeros_like(mask)
        for b, it in enumerate(batch["items"]):
            v = _as_1d(it["values"])[: int(mask[b].sum())]
            values[b, : len(v)] = v.to(dev)
        value_source = "cache"
    else:
        vm = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
        with torch.no_grad():
            values = token_values(vm, seq.to(model_device(vm)), attn.to(model_device(vm)))[:, pw - 1 : pw - 1 + T].to(dev)
        value_source = "critic"
        del vm
    values = values * mask

    rewards = shaped_rewards(task_reward, old, ref, mask, float(cfg["kl_beta"]))
    adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv_n = normalize_advantages(adv, mask)

    params = trainable_parameters(policy)
    init_state = [p.detach().clone() for p in params]
    mb = int(args.micro_batch)
    chunks = [slice(i, min(i + mb, seq.shape[0])) for i in range(0, seq.shape[0], mb)]
    with torch.no_grad():
        mid_lp = torch.cat([token_logprobs(policy, seq[c], attn[c], pw, resp[c])[0] for c in chunks])
    ratio0 = torch.exp(mid_lp - old)
    valid = ratio0[mask.bool()].cpu().numpy()
    log_r = np.log(valid)

    out = {
        "n_rollouts": len(batch["items"]), "valid_tokens": int(mask.sum().item()),
        "reward_source": reward_source, "value_source": value_source, "rebuild_notes": batch["notes"],
        "task_reward_mean": float(task_reward.mean()), "kl_token_mean_old_vs_ref": float(masked_mean(old - ref, mask)),
        "midpoint_vs_cached_old_ratio": {
            "mean": float(valid.mean()), "std": float(valid.std()),
            "quantiles": {q: float(np.quantile(valid, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
            "abs_log_ratio_mean": float(np.abs(log_r).mean()),
        },
        "per_epsilon": {},
    }
    for eps in cfg["clip_values"]:
        eps = float(eps)
        d0 = clip_diagnostics(ratio0, adv_n, mask, eps)
        # Short simulated optimisation on this fixed batch from identical weights.
        with torch.no_grad():
            for p, s in zip(params, init_state):
                p.copy_(s)
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
        steps = []
        n_tok = mask.sum()
        for k in range(int(args.sim_steps)):
            opt.zero_grad(set_to_none=True)
            ratios, loss_total = [], 0.0
            for c in chunks:
                new_lp, _ = token_logprobs(policy, seq[c], attn[c], pw, resp[c])
                _, ratio_c, _ = ppo_policy_loss(new_lp, old[c], adv_n[c], mask[c], eps)
                # Chunk share of the batch-level masked mean, so accumulated grads equal the full-batch loss.
                r = torch.exp(new_lp - old[c])
                obj = torch.minimum(r * adv_n[c], r.clamp(1 - eps, 1 + eps) * adv_n[c])
                loss_c = -(obj * mask[c]).sum() / n_tok
                loss_c.backward()
                loss_total += float(loss_c.item())
                ratios.append(ratio_c)
            ratio = torch.cat(ratios)
            d = clip_diagnostics(ratio, adv_n, mask, eps)
            loss = torch.tensor(loss_total)
            torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
            opt.step()
            steps.append({"step": k, "loss": float(loss.item()), **d})
        out["per_epsilon"][f"{eps:g}"] = {"at_midpoint": d0, "simulated_steps": steps}
        print(f"  eps={eps:g}: clip={d0['clip_fraction']:.4f} affected={d0['affected_fraction']:.4f} "
              f"surrogate={d0['clipped_surrogate']:.4f} (unclipped {d0['unclipped_surrogate']:.4f})")
    with torch.no_grad():
        for p, s in zip(params, init_state):
            p.copy_(s)
    return out


def stability_summary(log_path, threshold):
    recs = read_jsonl(log_path)
    ak = [e["approx_kl_old_new"] for r in recs for e in r["epochs"]]
    gn = [e["policy_grad_norm"] for r in recs for e in r["epochs"]]
    pl = [r["policy_loss"] for r in recs]
    rw = [r["rm_score_mean"] for r in recs]
    return {
        "approx_kl_mean": float(np.mean(ak)), "approx_kl_max": float(np.max(ak)),
        "unstable_epoch_fraction": float(np.mean(np.array(ak) > threshold)),
        "policy_grad_norm_mean": float(np.mean(gn)), "policy_grad_norm_max": float(np.max(gn)),
        "policy_loss_std": float(np.std(pl)), "train_reward_std": float(np.std(rw)),
        "train_clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in recs])),
        "train_affected_fraction_mean": float(np.mean([r["affected_fraction"] for r in recs])),
        "generated_tokens": int(recs[-1]["cumulative_generated_tokens"]),
        "updates": len(recs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--part", choices=["all", "cache", "forks", "summary"], default="all")
    ap.add_argument("--sim-steps", type=int, default=None, help="policy steps on the cached batch (default: ppo_epochs)")
    ap.add_argument("--micro-batch", type=int, default=4, help="rollouts per forward pass on the cached batch")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    args.sim_steps = args.sim_steps or int(cfg["ppo_epochs"])
    results = repo_path(cfg["results_dir"])
    out_root = repo_path(cfg["output"]).parent
    print("Required epsilon values:", cfg["clip_values"])

    if args.part in ("all", "cache"):
        save_json(results / "clipping_cached_batch.json", cached_batch_study(cfg, args))

    if args.part in ("all", "forks"):
        for eps in cfg["clip_values"]:
            name = f"clip_{float(eps):g}"
            out = out_root / name
            if not (args.skip_existing and (out / "adapter_config.json").exists()):
                run_module("task2_ppo.continue_train", ["--config", args.config, "--run-name", name, "--output", out,
                                                        "--updates", cfg["fork_updates"], "--clip-epsilon", eps])
            if not (args.skip_existing and (results / f"eval_{name}.json").exists()):
                run_module("task2_ppo.evaluate", ["--config", args.config, "--adapter", out, "--name", name])

    rows = []
    cache = load_json(results / "clipping_cached_batch.json") if (results / "clipping_cached_batch.json").exists() else {}
    for eps in cfg["clip_values"]:
        name = f"clip_{float(eps):g}"
        row = {"epsilon": float(eps)}
        c = cache.get("per_epsilon", {}).get(f"{float(eps):g}")
        if c:
            row.update({"cache_clip_fraction": c["at_midpoint"]["clip_fraction"],
                        "cache_affected_fraction": c["at_midpoint"]["affected_fraction"],
                        "cache_clipped_surrogate": c["at_midpoint"]["clipped_surrogate"],
                        "cache_affected_after_steps": c["simulated_steps"][-1]["affected_fraction"]})
        if (results / f"train_{name}.jsonl").exists():
            row.update(stability_summary(results / f"train_{name}.jsonl", float(cfg.get("stability_approx_kl_threshold", 0.02))))
        if (results / f"eval_{name}.json").exists():
            ev = load_json(results / f"eval_{name}.json")
            row.update({"heldout_reward": ev["reward"]["mean"], "heldout_kl_token": ev["kl_token_mean"],
                        "heldout_kl_seq": ev["kl_seq_mean"], "heldout_entropy": ev["entropy_token_mean"],
                        "heldout_len_mean": ev["response_tokens"]["mean"], "heldout_len_std": ev["response_tokens"]["std"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results / "clipping_summary.csv", index=False)
    save_json(results / "clipping_summary.json", rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
