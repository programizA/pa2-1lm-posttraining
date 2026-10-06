from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.rl_utils import (
    disable_dropout,
    fixed_prompt_order,
    grad_norm,
    peak_vram_gib,
    prompt_fits,
    reset_peak_vram,
    row_id,
    token_logprobs,
)
from task3_grpo.grpo import (
    group_relative_advantages,
    group_statistics,
    grpo_sequence_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompts": prompts,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def compute_advantages(rewards, group_ids, loss_type, cfg):
    if loss_type == "dr_grpo" and not bool(cfg.get("dr_grpo_keep_std_normalization", True)):
        adv = torch.zeros_like(rewards)
        for g in torch.unique(group_ids):
            idx = group_ids == g
            adv[idx] = rewards[idx] - rewards[idx].mean()
        return adv
    return group_relative_advantages(rewards, group_ids)


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}.jsonl"
    comp_path = results_dir / f"train_{run_name}_completions.jsonl"
    for p in (log_path, comp_path):
        if p.exists():
            p.unlink()

    tok, policy, opt = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rm_tok, prompt_rows = bundle["reward_model"], bundle["reward_tokenizer"], bundle["prompt_rows"]
    disable_dropout(policy)
    params = trainable_parameters(policy)

    seed = int(cfg["seed"])
    n_updates, epochs = int(cfg["updates"]), int(cfg.get("policy_epochs", 1))
    K, ppu = int(cfg["num_generations"]), int(cfg["prompts_per_update"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    tol = float(cfg.get("informative_std_tol", 1e-6))
    gen_cfg = cfg["generation"]

    order = fixed_prompt_order(len(prompt_rows), seed)
    order = [i for i in order if prompt_fits(tok, prompt_messages(prompt_rows[i]), int(cfg["max_prompt_length"]))]
    schedule = [order[u * ppu : (u + 1) * ppu] for u in range(n_updates)]
    print(f"[GRPO:{run_name}] loss_type={loss_type} updates={n_updates} K={K} prompts/update={ppu} eps={eps} beta={beta}", flush=True)

    reset_peak_vram()
    elapsed = wall_timer()
    total_tokens = 0
    for u in range(n_updates):
        rows = [prompt_rows[i] for i in schedule[u]]
        msgs = [prompt_messages(r) for r in rows for _ in range(K)]
        gids = torch.tensor([g for g in range(len(rows)) for _ in range(K)])

        set_seed(seed + 2000 + u)  # same sampling RNG at update u in every fork
        gen = batch_generate(policy, tok, msgs, int(cfg["max_prompt_length"]), max_len,
                             float(gen_cfg["temperature"]), float(gen_cfg["top_p"]), True)
        seq, attn, pw, resp, mask = gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], gen["response_mask"]
        dev = seq.device
        total_tokens += int(mask.sum().item())
        train_mask = mask_truncated_sequences(mask, gen["truncated"]) if cfg.get("mask_truncated_completions", True) else mask

        with torch.no_grad():
            rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg.get("reward_max_length", 1280))).to(dev)
            adv = compute_advantages(rewards, gids.to(dev), loss_type, cfg)
            gstats = group_statistics(rewards, gids.to(dev), tol=tol)
            old_lp, ent = token_logprobs(policy, seq, attn, pw, resp, with_entropy=True)
            with reference_mode(policy):
                ref_lp, _ = token_logprobs(policy, seq, attn, pw, resp)

        n_seq = seq.shape[0]
        total_valid = float(train_mask.sum().item())
        ep_logs, per_comp = [], []
        for epoch in range(epochs):
            acc = [torch.zeros_like(p) for p in params]
            loss_total, clip_num = 0.0, 0.0
            comp = []
            for i in range(n_seq):
                sl = slice(i, i + 1)
                new_lp, _ = token_logprobs(policy, seq[sl], attn[sl], pw, resp[sl])
                loss_i, parts = grpo_sequence_loss(new_lp, old_lp[sl], adv[sl], train_mask[sl], ref_lp[sl], eps, beta,
                                                   n_seq, total_valid, loss_type, max_len)
                for p in params:
                    p.grad = None
                if train_mask[sl].sum() > 0:
                    # Policy-term gradient first (length-normalisation statistic), then add the KL term.
                    parts["policy_part"].backward(retain_graph=True)
                    g_policy = grad_norm(params)
                    (beta * parts["kl_part"]).backward()
                    g_total = grad_norm(params)
                    for a, p in zip(acc, params):
                        if p.grad is not None:
                            a.add_(p.grad)
                else:
                    g_policy = g_total = 0.0
                ratio = torch.exp(new_lp.detach() - old_lp[sl])
                clip_num += float((((ratio < 1 - eps) | (ratio > 1 + eps)).float() * train_mask[sl]).sum().item())
                loss_total += float(loss_i.item())
                n_tok = int(mask[i].sum().item())
                denom = max(int(train_mask[i].sum().item()), 1) if loss_type == "grpo" else max_len
                comp.append({
                    "update": u + 1, "epoch": epoch, "prompt_id": row_id(rows[int(gids[i])], schedule[u][int(gids[i])]),
                    "group": int(gids[i]), "completion_index": i, "length": n_tok,
                    "truncated": bool(gen["truncated"][i]), "trained": bool(train_mask[i].sum() > 0),
                    "reward": float(rewards[i]), "advantage": float(adv[i]),
                    "per_token_weight": abs(float(adv[i])) / denom / n_seq if train_mask[i].sum() > 0 else 0.0,
                    "sequence_weight": abs(float(adv[i])) * int(train_mask[i].sum().item()) / denom / n_seq,
                    "grad_norm_policy": g_policy, "grad_norm_total": g_total,
                    "kl_seq": float(((old_lp[i] - ref_lp[i]) * mask[i]).sum().item()),
                })
            for a, p in zip(acc, params):
                p.grad = a
            gnorm = grad_norm(params)
            torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            ep_logs.append({"policy_loss": loss_total, "grad_norm": gnorm, "clip_fraction": clip_num / max(total_valid, 1.0)})
            per_comp.extend(comp)

        for c in per_comp:
            append_jsonl(comp_path, c)
        rec = {
            "update": u + 1,
            "loss_type": loss_type,
            "prompt_ids": [row_id(r, i) for r, i in zip(rows, schedule[u])],
            "reward_mean": float(rewards.mean().item()),
            "reward_std_all": float(rewards.std(unbiased=False).item()),
            "group_reward_std_mean": gstats["group_reward_std_mean"],
            "uninformative_group_fraction": gstats["uninformative_group_fraction"],
            "kl_token_mean": float(masked_mean(old_lp - ref_lp, mask).item()),
            "kl_seq_mean": float(((old_lp - ref_lp) * mask).sum(-1).mean().item()),
            "entropy_token_mean": float(masked_mean(ent, mask).item()),
            "sampled_entropy_token_mean": float(masked_mean(-old_lp, mask).item()),
            "response_tokens_mean": float(mask.sum(-1).float().mean().item()),
            "response_tokens_std": float(mask.sum(-1).float().std(unbiased=False).item()),
            "truncated_fraction": float(sum(gen["truncated"]) / n_seq),
            "trained_completion_fraction": float((train_mask.sum(-1) > 0).float().mean().item()),
            "policy_loss": sum(e["policy_loss"] for e in ep_logs) / len(ep_logs),
            "grad_norm": sum(e["grad_norm"] for e in ep_logs) / len(ep_logs),
            "clip_fraction": sum(e["clip_fraction"] for e in ep_logs) / len(ep_logs),
            "cumulative_generated_tokens": total_tokens,
            "elapsed_s": elapsed(),
            "peak_vram_gib": peak_vram_gib(),
        }
        append_jsonl(log_path, rec)
        print(f"  update {u+1}/{n_updates} reward={rec['reward_mean']:.3f} grpStd={rec['group_reward_std_mean']:.3f} "
              f"uninf={rec['uninformative_group_fraction']:.2f} kl={rec['kl_token_mean']:.4f} len={rec['response_tokens_mean']:.0f} "
              f"trunc={rec['truncated_fraction']:.2f} gnorm={rec['grad_norm']:.3f} t={rec['elapsed_s']:.0f}s", flush=True)

    policy.save_pretrained(str(out))
    meta = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "num_generations": K,
        "prompts_per_update": ppu, "policy_epochs": epochs, "learning_rate": float(cfg["learning_rate"]),
        "clip_epsilon": eps, "kl_beta": beta, "max_completion_length": max_len,
        "mask_truncated_completions": bool(cfg.get("mask_truncated_completions", True)),
        "dr_grpo_keep_std_normalization": bool(cfg.get("dr_grpo_keep_std_normalization", True)),
        "seed": seed, "prompt_schedule": [[row_id(prompt_rows[i], i) for i in s] for s in schedule],
        "cumulative_generated_tokens": total_tokens, "wall_clock_s": elapsed(), "peak_vram_gib": peak_vram_gib(),
        "start_checkpoint": cfg["paths"]["grpo_midpoint_policy"],
        "kl_estimator": "k3: exp(ref-new) - (ref-new) - 1 (loss); sampled log-ratio (logged KL)",
        "dropout": "disabled so old/new log-probs match exactly",
    }
    save_json(out / "run_meta.json", meta)
    save_json(results_dir / f"train_{run_name}_meta.json", meta)
    print(f"[GRPO:{run_name}] saved to {out} ({elapsed():.0f}s, peak {peak_vram_gib()} GiB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
