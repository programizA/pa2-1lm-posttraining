"""Derived numbers quoted in the report that are not already a single field of a saved result file.

    python -m analysis.report_numbers            # writes results/report_numbers.json

Everything here is computed from files under results/ (no model calls):
  * paired held-out comparisons on the same prompts: mean reward difference with a 95% bootstrap interval over
    prompts, and the fraction of byte-identical responses (how far the policy actually moved);
  * Wilson 95% intervals for DPO held-out preference accuracy (overall and per length stratum);
  * PPO critic diagnostics over the 20-update continuation;
  * GRPO masking / informativeness counts over the 20-update continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 6304


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def paired(ref_file, cur_file, reps=4000):
    ref = {r["prompt_id"]: r for r in jl(ref_file)}
    cur = {r["prompt_id"]: r for r in jl(cur_file)}
    ids = [k for k in ref if k in cur]
    d = np.array([cur[k]["reward"] - ref[k]["reward"] for k in ids], dtype=float)
    rng = np.random.default_rng(SEED)
    boots = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return {"n": len(ids), "reward_ref": float(np.mean([ref[k]["reward"] for k in ids])),
            "reward_cur": float(np.mean([cur[k]["reward"] for k in ids])),
            "reward_diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "identical_text": float(np.mean([cur[k]["response"] == ref[k]["response"] for k in ids])),
            "len_ref": float(np.mean([ref[k]["response_tokens"] for k in ids])),
            "len_cur": float(np.mean([cur[k]["response_tokens"] for k in ids]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    R = Path(ap.parse_args().results)
    out = {}

    # Task 1
    t1 = R / "task1_dpo"
    out["t1_vs_sft"] = {c: paired(t1 / "eval_sft_generations.jsonl", t1 / f"eval_{c}_generations.jsonl")
                        for c in ["standard", "beta_0.03", "beta_0.1", "beta_0.3", "length_balanced"]}
    acc = {}
    for c in ["standard", "beta_0.03", "beta_0.1", "beta_0.3", "length_balanced"]:
        pairs = jl(t1 / f"eval_{c}_pairs.jsonl")
        k = sum(bool(p["correct"]) for p in pairs)
        acc[c] = {"n": len(pairs), "acc": k / len(pairs), "ci95": wilson(k, len(pairs))}
    out["t1_heldout_acc"] = acc
    strat = {}
    for c in ["standard", "length_balanced"]:
        pairs = jl(t1 / f"eval_{c}_dpo_length_stratified_eval_pairs.jsonl")
        strat[c] = {}
        for s in sorted({p["stratum"] for p in pairs}):
            ps = [p for p in pairs if p["stratum"] == s]
            k = sum(bool(p["correct"]) for p in ps)
            strat[c][s] = {"n": len(ps), "acc": k / len(ps), "ci95": wilson(k, len(ps)),
                           "chosen_logratio": float(np.mean([p["policy_logratio_chosen"] for p in ps])),
                           "rejected_logratio": float(np.mean([p["policy_logratio_rejected"] for p in ps]))}
    out["t1_strata"] = strat
    tr = jl(t1 / "train_standard.jsonl")
    out["t1_train_standard"] = {"first_loss": tr[0]["loss"], "last_loss": tr[-1]["loss"],
                                "last10_loss_mean": float(np.mean([r["loss"] for r in tr[-10:]])),
                                "last10_acc_mean": float(np.mean([r["preference_accuracy"] for r in tr[-10:]])),
                                "chosen_reward_last": tr[-1]["chosen_reward_mean"],
                                "rejected_reward_last": tr[-1]["rejected_reward_mean"]}

    # Task 2
    t2 = R / "task2_ppo"
    out["t2_vs_midpoint"] = {c: paired(t2 / "eval_midpoint_generations.jsonl", t2 / f"eval_{c}_generations.jsonl")
                             for c in ["standard", "clip_0.05", "clip_0.2", "clip_0.5", "kl_0", "kl_0.1", "kl_0.2"]}
    out["t2_fork_identity"] = {
        f"{a}_vs_{b}": float(np.mean([x["response"] == y["response"] for x, y in
                                      zip(jl(t2 / f"eval_{a}_generations.jsonl"), jl(t2 / f"eval_{b}_generations.jsonl"))]))
        for a, b in [("clip_0.05", "clip_0.2"), ("clip_0.2", "clip_0.5"), ("kl_0", "kl_0.2"), ("kl_0.1", "kl_0.2")]}
    log = jl(t2 / "train_standard.jsonl")
    v0 = np.array([r["value_first_token_mean"] for r in log])
    rw = np.array([r["task_reward_mean"] for r in log])
    ratios = [(e["ratio_min"], e["ratio_max"]) for r in log for e in r["epochs"]]
    out["t2_critic"] = {"corr_value_first_token_vs_reward": float(np.corrcoef(v0, rw)[0, 1]),
                        "explained_variance_mean": float(np.mean([r["explained_variance"] for r in log])),
                        "explained_variance_median": float(np.median([r["explained_variance"] for r in log])),
                        "ratio_min": float(min(a for a, _ in ratios)), "ratio_max": float(max(b for _, b in ratios)),
                        "clip_fraction_max": float(max(r["clip_fraction"] for r in log)),
                        "kl_token_range": [float(min(r["kl_token_mean"] for r in log)), float(max(r["kl_token_mean"] for r in log))],
                        "reward_range": [float(rw.min()), float(rw.max())],
                        "len_range": [float(min(r["response_tokens_mean"] for r in log)), float(max(r["response_tokens_mean"] for r in log))]}

    # Task 3
    t3 = R / "task3_grpo"
    out["t3_vs_midpoint"] = {c: paired(t3 / "eval_midpoint_generations.jsonl", t3 / f"eval_{c}_generations.jsonl")
                             for c in ["standard", "norm_grpo", "norm_dr_grpo"]}
    log = jl(t3 / "train_standard.jsonl")
    out["t3_train_standard"] = {
        "masked_fraction": float(1 - np.mean([r["trained_completion_fraction"] for r in log])),
        "fully_masked_updates": [r["update"] for r in log if r["trained_completion_fraction"] == 0],
        "uninformative_updates": [r["update"] for r in log if r["uninformative_group_fraction"] == 1],
        "group_std_mean": float(np.mean([r["group_reward_std_mean"] for r in log])),
        "kl_token_range": [float(min(r["kl_token_mean"] for r in log)), float(max(r["kl_token_mean"] for r in log))],
        "clip_fraction_max": float(max(r["clip_fraction"] for r in log))}
    a, b = jl(t3 / "train_norm_grpo.jsonl"), jl(t3 / "train_standard.jsonl")
    out["t3_norm_fork_matches_standard_updates"] = sum(
        abs(x["reward_mean"] - y["reward_mean"]) < 1e-9 for x, y in zip(a, b[: len(a)]))

    (R / "report_numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
