"""Task 3, Step 3: canonical GRPO (1/|o_i| per sequence) vs Dr. GRPO-style constant normalisation.

Both forks start from the identical supplied midpoint and share: reward model, beta, eps, prompt
schedule, sampling seeds per update, K, generation cap and update budget. Only the sequence
normalisation in the loss changes (see dr_grpo_keep_std_normalization in configs/grpo.yaml).

Length-conditioned statistics (from the per-completion training logs):
* grad_norm_policy: L2 norm of ONE completion's policy-term gradient (measured exactly by
  back-propagating completions one at a time), binned by completion length;
* per_token_weight |A|/(denominator * N): how much each token of that completion is weighted;
* corr(length, grad_norm_policy), split by advantage sign.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.rl_utils import run_module


def length_conditioned(comps, edges):
    df = pd.DataFrame([c for c in comps if c["trained"]])
    if df.empty:
        return {}, {}
    if len(edges) >= 2:
        df["len_bin"] = pd.cut(df["length"], bins=edges, include_lowest=True).astype(str)
    else:
        df["len_bin"] = "all"
    agg = df.groupby("len_bin").agg(
        n=("length", "size"), length_mean=("length", "mean"), abs_adv_mean=("advantage", lambda a: float(np.mean(np.abs(a)))),
        grad_norm_policy_mean=("grad_norm_policy", "mean"), per_token_weight_mean=("per_token_weight", "mean"),
        sequence_weight_mean=("sequence_weight", "mean")).reset_index()
    corr = {
        "all": safe_corr(df["length"], df["grad_norm_policy"]),
        "positive_adv": safe_corr(df[df.advantage > 0]["length"], df[df.advantage > 0]["grad_norm_policy"]),
        "negative_adv": safe_corr(df[df.advantage < 0]["length"], df[df.advantage < 0]["grad_norm_policy"]),
        "length_vs_advantage": safe_corr(df["length"], df["advantage"]),
        # Gradient per unit |advantage| removes the reward-scale factor and isolates normalisation.
        "length_vs_gradnorm_per_abs_adv": safe_corr(df["length"], df["grad_norm_policy"] / df["advantage"].abs().clip(lower=1e-6)),
    }
    return agg.to_dict(orient="records"), corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    out_root = repo_path(cfg["output"]).parent
    print("Fork updates:", cfg["fork_updates"], "| comparing loss_type='grpo' vs 'dr_grpo' from the same midpoint")

    names = {"grpo": "norm_grpo", "dr_grpo": "norm_dr_grpo"}
    if not args.summary_only:
        for lt, name in names.items():
            out = out_root / name
            if not (args.skip_existing and (out / "adapter_config.json").exists()):
                run_module("task3_grpo.continue_train", ["--config", args.config, "--run-name", name, "--output", out,
                                                         "--updates", cfg["fork_updates"], "--loss-type", lt])
            if not (args.skip_existing and (results / f"eval_{name}.json").exists()):
                run_module("task3_grpo.evaluate", ["--config", args.config, "--adapter", out, "--name", name])

    comps = {lt: read_jsonl(results / f"train_{name}_completions.jsonl") for lt, name in names.items()}
    all_len = [c["length"] for v in comps.values() for c in v if c["trained"]]
    edges = sorted(set(np.quantile(all_len, [0, 0.25, 0.5, 0.75, 1.0]).round().tolist())) if all_len else [0, 1]
    report = {"length_bin_edges_tokens": edges, "bin_rule": "pooled quartiles of trained-completion length across both forks",
              "conditions": {}}
    rows = []
    for lt, name in names.items():
        ev = load_json(results / f"eval_{name}.json")
        log = read_jsonl(results / f"train_{name}.jsonl")
        table, corr = length_conditioned(comps[lt], edges)
        report["conditions"][lt] = {
            "heldout": {k: ev[k] for k in ("reward", "kl_token_mean", "kl_seq_mean", "entropy_token_mean", "response_tokens", "truncation_rate")},
            "train_length_trajectory": [r["response_tokens_mean"] for r in log],
            "train_reward_trajectory": [r["reward_mean"] for r in log],
            "train_grad_norm_trajectory": [r["grad_norm"] for r in log],
            "length_conditioned": table, "correlations": corr,
            "generated_tokens": log[-1]["cumulative_generated_tokens"],
        }
        rows.append({"loss_type": lt, "heldout_reward": ev["reward"]["mean"], "heldout_kl_token": ev["kl_token_mean"],
                     "heldout_len_mean": ev["response_tokens"]["mean"], "heldout_len_std": ev["response_tokens"]["std"],
                     "train_len_first": log[0]["response_tokens_mean"], "train_len_last": log[-1]["response_tokens_mean"],
                     "train_grad_norm_mean": float(np.mean([r["grad_norm"] for r in log])),
                     "corr_len_gradnorm": corr.get("all"), "corr_len_gradnorm_per_abs_adv": corr.get("length_vs_gradnorm_per_abs_adv"),
                     "generated_tokens": log[-1]["cumulative_generated_tokens"]})
    save_json(results / "normalization_comparison.json", report)
    pd.DataFrame(rows).to_csv(results / "normalization_comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    for lt in names:
        print(f"\n{lt} length-conditioned:")
        print(pd.DataFrame(report["conditions"][lt]["length_conditioned"]).to_string(index=False))


if __name__ == "__main__":
    main()
