"""Task 2, Step 3: reward-overoptimisation study over the KL-shaping coefficient.

Forks for beta_KL in {0, 0.10, 0.20}, each from the identical supplied midpoint (policy + critic),
with identical prompt schedule, sampling seeds, clip epsilon and update/generation budgets.
Outputs: held-out reward/KL/entropy/length per fork, per-update trajectories side by side, and a
"which observable moves first" table (first update at which a fork departs from the beta=0.20 fork
by more than one standard deviation of that metric's trajectory in the standard run).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.rl_utils import run_module

TRAJ_METRICS = ["rm_score_mean", "kl_token_mean", "kl_seq_mean", "entropy_token_mean", "response_tokens_mean",
                "value_loss", "explained_variance", "clip_fraction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    out_root = repo_path(cfg["output"]).parent
    print("KL beta conditions:", cfg["kl_values"], "| fork update budget:", cfg["fork_updates"])

    names = [f"kl_{float(b):g}" for b in cfg["kl_values"]]
    if not args.summary_only:
        for b, name in zip(cfg["kl_values"], names):
            out = out_root / name
            if not (args.skip_existing and (out / "adapter_config.json").exists()):
                run_module("task2_ppo.continue_train", ["--config", args.config, "--run-name", name, "--output", out,
                                                        "--updates", cfg["fork_updates"], "--kl-beta", b])
            if not (args.skip_existing and (results / f"eval_{name}.json").exists()):
                run_module("task2_ppo.evaluate", ["--config", args.config, "--adapter", out, "--name", name])

    rows, traj = [], []
    for b, name in zip(cfg["kl_values"], names):
        ev = load_json(results / f"eval_{name}.json")
        log = read_jsonl(results / f"train_{name}.jsonl")
        rows.append({
            "kl_beta": float(b),
            "heldout_reward_mean": ev["reward"]["mean"], "heldout_reward_std": ev["reward"]["std"],
            "heldout_kl_token": ev["kl_token_mean"], "heldout_kl_seq": ev["kl_seq_mean"],
            "heldout_entropy": ev["entropy_token_mean"],
            "heldout_len_mean": ev["response_tokens"]["mean"], "heldout_len_std": ev["response_tokens"]["std"],
            "heldout_truncation_rate": ev["truncation_rate"],
            "train_final_rm_score": log[-1]["rm_score_mean"], "train_final_kl_token": log[-1]["kl_token_mean"],
            "generated_tokens": log[-1]["cumulative_generated_tokens"],
        })
        for r in log:
            traj.append({"kl_beta": float(b), "update": r["update"], **{m: r.get(m) for m in TRAJ_METRICS}})
    pd.DataFrame(rows).to_csv(results / "kl_summary.csv", index=False)
    tdf = pd.DataFrame(traj)
    tdf.to_csv(results / "kl_trajectories.csv", index=False)

    # Which observable departs first (relative to the strongest-KL fork)?
    first = {}
    std_log = results / "train_standard.jsonl"
    scale_src = read_jsonl(std_log) if std_log.exists() else None
    anchor = float(max(cfg["kl_values"]))
    for b in cfg["kl_values"]:
        if float(b) == anchor:
            continue
        a = tdf[tdf.kl_beta == anchor].set_index("update")
        f = tdf[tdf.kl_beta == float(b)].set_index("update")
        first[f"{float(b):g}_vs_{anchor:g}"] = {}
        for m in TRAJ_METRICS:
            series = [r.get(m) for r in scale_src] if scale_src else list(a[m])
            sd = float(np.nanstd(np.asarray(series, dtype=float))) or 1e-12
            diff = (f[m] - a[m]).abs() / sd
            hit = diff[diff > 1.0]
            first[f"{float(b):g}_vs_{anchor:g}"][m] = int(hit.index[0]) if len(hit) else None
    save_json(results / "kl_first_departure.json", {"rule": "first update with |metric(fork) - metric(anchor)| > 1 sd "
                                                     "of that metric over the standard 20-update run",
                                                     "anchor_kl_beta": anchor, "first_update": first})
    print(pd.DataFrame(rows).to_string(index=False))
    print("first departure:", first)


if __name__ == "__main__":
    main()
