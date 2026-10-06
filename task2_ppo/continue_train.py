from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
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
from task2_ppo.ppo import (
    clip_diagnostics,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    # Keep trainable critic parameters (LoRA + head copy) in float32 for AdamW stability; the
    # frozen backbone stays in the configured half precision. Initial values are unchanged.
    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def ppo_prompt_schedule(cfg, tokenizer, prompt_rows, n_updates):
    """Fixed seeded prompt order, shared by the standard run and every fork (forks use a prefix)."""
    order = fixed_prompt_order(len(prompt_rows), int(cfg["seed"]))
    order = [i for i in order if prompt_fits(tokenizer, prompt_messages(prompt_rows[i]), int(cfg["max_prompt_length"]))]
    ppu = int(cfg["prompts_per_update"])
    need = ppu * n_updates
    if len(order) < need:
        raise ValueError(f"Prompt pool too small: need {need}, have {len(order)}")
    return [order[u * ppu : (u + 1) * ppu] for u in range(n_updates)]


def critic_values(value_model, sequences, attention_mask, prompt_width, steps):
    """V(s_t) for each response step t: the critic's output at the position *before* token t."""
    v = token_values(value_model, sequences, attention_mask)
    return v[:, prompt_width - 1 : prompt_width - 1 + steps]


def explained_variance(pred, target, mask):
    m = mask.bool()
    y, p = target[m], pred[m]
    var_y = y.var(unbiased=False)
    if var_y.item() == 0:
        return float("nan")
    return float(1.0 - (y - p).var(unbiased=False) / var_y)


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}.jsonl"
    if log_path.exists():
        log_path.unlink()

    tok, policy, value_model = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok, prompt_rows = bundle["reward_model"], bundle["reward_tokenizer"], bundle["prompt_rows"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    disable_dropout(policy)
    disable_dropout(value_model)
    p_params, v_params = trainable_parameters(policy), trainable_parameters(value_model)

    seed = int(cfg["seed"])
    n_updates, epochs = int(cfg["updates"]), int(cfg["ppo_epochs"])
    eps, beta_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    max_norm, vcoef = float(cfg["max_grad_norm"]), float(cfg["value_coef"])
    eos_pen = float(cfg.get("missing_eos_penalty", 0.0))
    gen_cfg = cfg["generation"]
    schedule = ppo_prompt_schedule(cfg, tok, prompt_rows, n_updates)
    print(f"[PPO:{run_name}] updates={n_updates} epochs={epochs} eps={eps} kl_beta={beta_kl}", flush=True)

    reset_peak_vram()
    elapsed = wall_timer()
    total_tokens = 0
    for u in range(n_updates):
        rows = [prompt_rows[i] for i in schedule[u]]
        msgs = [prompt_messages(r) for r in rows]

        # ---------------------------------------------------------------- rollout
        set_seed(seed + 1000 + u)  # same sampling RNG at update u in every fork
        gen = batch_generate(policy, tok, msgs, int(cfg["max_prompt_length"]), int(cfg["max_response_length"]),
                             float(gen_cfg["temperature"]), float(gen_cfg["top_p"]), True)
        seq, attn, pw, resp, mask = gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], gen["response_mask"]
        T = resp.shape[1]
        total_tokens += int(mask.sum().item())

        with torch.no_grad():
            old_lp, ent = token_logprobs(policy, seq, attn, pw, resp, with_entropy=True)
            with reference_mode(policy):
                ref_lp, _ = token_logprobs(policy, seq, attn, pw, resp)
            rm_score = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg["reward_max_length"])).to(seq.device)
            no_eos = torch.tensor([not t for t in gen["terminated_with_eos"]], device=seq.device, dtype=torch.float32)
            task_reward = rm_score - eos_pen * no_eos
            values = critic_values(value_model, seq, attn, pw, T) * mask

            rewards = shaped_rewards(task_reward, old_lp, ref_lp, mask, beta_kl)
            adv, returns = compute_gae(rewards, values, mask, gamma, lam)
            adv_n = normalize_advantages(adv, mask)
            ev_before = explained_variance(values, returns, mask)

        # ---------------------------------------------------------------- PPO epochs
        ep_logs = []
        for epoch in range(epochs):
            new_lp, _ = token_logprobs(policy, seq, attn, pw, resp)
            ploss, ratio, cf = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            popt.zero_grad(set_to_none=True)
            ploss.backward()
            pg = grad_norm(p_params)
            torch.nn.utils.clip_grad_norm_(p_params, max_norm)
            popt.step()
            cd = clip_diagnostics(ratio, adv_n, mask, eps)
            log_ratio = (new_lp.detach() - old_lp)
            approx_kl = float(masked_mean((torch.exp(log_ratio) - 1.0) - log_ratio, mask).item())

            v_new = critic_values(value_model, seq, attn, pw, T)
            vloss = vcoef * value_mse_loss(v_new, returns, mask)
            vopt.zero_grad(set_to_none=True)
            vloss.backward()
            vg = grad_norm(v_params)
            torch.nn.utils.clip_grad_norm_(v_params, max_norm)
            vopt.step()
            ep_logs.append({"policy_loss": float(ploss.item()), "value_loss": float(vloss.item()),
                            "policy_grad_norm": pg, "value_grad_norm": vg, "approx_kl_old_new": approx_kl, **cd,
                            "ratio_max": float((ratio * mask).max().item()),
                            "ratio_min": float(torch.where(mask.bool(), ratio, torch.ones_like(ratio)).min().item())})

        n_valid = mask.sum(-1)
        kl_tok = masked_mean(old_lp - ref_lp, mask)
        rec = {
            "update": u + 1,
            "prompt_ids": [row_id(prompt_rows[i], i) for i in schedule[u]],
            "rm_score_mean": float(rm_score.mean().item()),
            "task_reward_mean": float(task_reward.mean().item()),
            "kl_token_mean": float(kl_tok.item()),
            "kl_seq_mean": float(((old_lp - ref_lp) * mask).sum(-1).mean().item()),
            "kl_penalty_seq_mean": float((-beta_kl * (old_lp - ref_lp) * mask).sum(-1).mean().item()),
            "entropy_token_mean": float(masked_mean(ent, mask).item()),
            "response_tokens_mean": float(n_valid.float().mean().item()),
            "truncated_fraction": float(sum(gen["truncated"]) / len(gen["truncated"])),
            "eos_fraction": float(sum(gen["terminated_with_eos"]) / len(gen["terminated_with_eos"])),
            "value_mean": float(masked_mean(values, mask).item()),
            "return_mean": float(masked_mean(returns, mask).item()),
            "value_first_token_mean": float(values[:, 0].mean().item()),
            "explained_variance": ev_before,
            "advantage_raw_mean": float(masked_mean(adv, mask).item()),
            "advantage_raw_std": float(adv[mask.bool()].std(unbiased=False).item()) if mask.sum() > 1 else 0.0,
            "epochs": ep_logs,
            "policy_loss": sum(e["policy_loss"] for e in ep_logs) / epochs,
            "value_loss": sum(e["value_loss"] for e in ep_logs) / epochs,
            "policy_grad_norm": sum(e["policy_grad_norm"] for e in ep_logs) / epochs,
            "value_grad_norm": sum(e["value_grad_norm"] for e in ep_logs) / epochs,
            "clip_fraction": sum(e["clip_fraction"] for e in ep_logs) / epochs,
            "affected_fraction": sum(e["affected_fraction"] for e in ep_logs) / epochs,
            "approx_kl_old_new": sum(e["approx_kl_old_new"] for e in ep_logs) / epochs,
            "cumulative_generated_tokens": total_tokens,
            "elapsed_s": elapsed(),
            "peak_vram_gib": peak_vram_gib(),
            "response_preview": gen["responses"][0][:300],
        }
        append_jsonl(log_path, rec)
        print(f"  update {u+1}/{n_updates} rm={rec['rm_score_mean']:.3f} kl={rec['kl_token_mean']:.4f} "
              f"len={rec['response_tokens_mean']:.0f} clip={rec['clip_fraction']:.3f} vloss={rec['value_loss']:.3f} "
              f"EV={ev_before:.2f} t={rec['elapsed_s']:.0f}s", flush=True)

    policy.save_pretrained(str(out))
    try:
        value_model.save_pretrained(str(out / "value_adapter"))
    except Exception as exc:  # critic save is informational only
        print("  (value adapter not saved:", exc, ")")
    meta = {
        "run_name": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta_kl,
        "ppo_epochs": epochs, "prompts_per_update": int(cfg["prompts_per_update"]),
        "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_learning_rates": [float(cfg["value_lora_learning_rate"]), float(cfg["value_head_learning_rate"])],
        "gamma": gamma, "gae_lambda": lam, "value_coef": vcoef, "missing_eos_penalty": eos_pen,
        "max_response_length": int(cfg["max_response_length"]), "seed": seed,
        "prompt_schedule": [[row_id(prompt_rows[i], i) for i in s] for s in schedule],
        "cumulative_generated_tokens": total_tokens,
        "wall_clock_s": elapsed(), "peak_vram_gib": peak_vram_gib(),
        "start_checkpoint": cfg["paths"]["ppo_midpoint_policy"], "value_checkpoint": cfg["paths"]["ppo_midpoint_value"],
        "dropout": "disabled in policy and critic so old/new log-probs match exactly",
    }
    save_json(out / "run_meta.json", meta)
    save_json(results_dir / f"train_{run_name}_meta.json", meta)
    print(f"[PPO:{run_name}] saved to {out} ({elapsed():.0f}s, peak {peak_vram_gib()} GiB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
