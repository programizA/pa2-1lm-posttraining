"""Task 3, Step 2: equal-generation group-size study on the supplied K=8 completion/reward cache.

No training. The supplied cache holds 8 completions per prompt. Two equal-budget regroupings:

* partition (primary): every cached prompt keeps all 8 of its completions, split in
  generation_index order into 8/K disjoint groups of size K. Total completions = 8 x N_prompts
  for every K, so the comparison is at an equal generation budget on identical samples.
* coverage (secondary): a fixed budget of B = 2 x N_prompts completions is spent on B/K prompts,
  i.e. K=2 uses 2 completions of every prompt, K=4 uses 4 completions of half the prompts, K=8 all 8
  of a quarter. Prompt subsets are drawn with a fixed seed and the draw is repeated `--repeats`
  times; this matches the "more prompts vs more samples per prompt" trade-off directly.

Prompt difficulty (defined once): a prompt's difficulty is the mean reward of its 8 cached
completions; prompts are split into tertile bins of that value (low-reward = "hard",
high-reward = "easy"). With a binary reward the bins are instead fixed thresholds of the 8-sample
success rate: hard <= 1/8, easy >= 7/8, medium otherwise.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition scheme: split each prompt's first 8 completions into 8/k disjoint groups of size k.

    Returns a list of (prompt_id, sub_group_index, [rewards...]). Total completions is 8 x N_prompts
    for every k, i.e. the generation budget is identical across group sizes.
    """
    if 8 % k:
        raise ValueError("k must divide 8")
    groups = []
    for pid, rows in by_prompt.items():
        r = [float(x["reward"]) for x in rows[:8]]
        for s in range(8 // k):
            groups.append((pid, s, r[s * k : (s + 1) * k]))
    return groups


def coverage_groups(by_prompt, k: int, budget: int, rng):
    pids = list(by_prompt)
    n_prompts = budget // k
    chosen = rng.choice(len(pids), size=min(n_prompts, len(pids)), replace=False)
    groups = []
    for j in chosen:
        pid = pids[j]
        r = [float(x["reward"]) for x in by_prompt[pid][:8]]
        idx = rng.choice(8, size=k, replace=False)
        groups.append((pid, 0, [r[i] for i in idx]))
    return groups


def group_metrics(groups, full_mean, tol):
    stds, centered, normed, baseline_err = [], [], [], []
    informative = []
    for pid, _, r in groups:
        r = np.asarray(r, dtype=float)
        sd = r.std(ddof=0)
        stds.append(sd)
        informative.append(sd > tol)
        c = r - r.mean()
        centered.extend(c.tolist())
        normed.extend((c / (sd + 1e-6)).tolist())
        baseline_err.append((r.mean() - full_mean[pid]) ** 2)
    informative = np.asarray(informative)
    n_completions = sum(len(g[2]) for g in groups)
    nonzero_adv = sum(len(g[2]) for g, inf in zip(groups, informative) if inf)
    return {
        "n_groups": len(groups),
        "n_completions": n_completions,
        "n_prompts": len({g[0] for g in groups}),
        "informative_group_rate": float(informative.mean()) if len(groups) else None,
        "uninformative_group_rate": float(1 - informative.mean()) if len(groups) else None,
        "fraction_completions_with_signal": nonzero_adv / max(n_completions, 1),
        "prompts_with_any_informative_group": len({g[0] for g, inf in zip(groups, informative) if inf}),
        "mean_within_group_reward_std": float(np.mean(stds)) if stds else None,
        "var_centered_advantage": float(np.var(centered)) if centered else None,
        "var_normalized_advantage": float(np.var(normed)) if normed else None,
        "baseline_mse_vs_8sample_mean": float(np.mean(baseline_err)) if baseline_err else None,
    }


def difficulty_bins(by_prompt):
    means = {pid: float(np.mean([float(x["reward"]) for x in rows[:8]])) for pid, rows in by_prompt.items()}
    all_r = np.array([float(x["reward"]) for rows in by_prompt.values() for x in rows[:8]])
    binary = set(np.unique(all_r).tolist()).issubset({0.0, 1.0})
    if binary:
        rule = "binary reward: hard if success rate <= 1/8, easy if >= 7/8, else medium"
        bins = {pid: ("hard" if m <= 1 / 8 else "easy" if m >= 7 / 8 else "medium") for pid, m in means.items()}
    else:
        q1, q2 = np.quantile(list(means.values()), [1 / 3, 2 / 3])
        rule = f"tertiles of 8-sample mean reward: hard <= {q1:.4f} < medium <= {q2:.4f} < easy"
        bins = {pid: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for pid, m in means.items()}
    return means, bins, rule, binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--repeats", type=int, default=20, help="random prompt draws for the coverage scheme")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    tol = float(cfg.get("informative_std_tol", 1e-6))
    print("Cached prompts:", len(by_prompt), "| group sizes:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    full_mean, bins, rule, binary = difficulty_bins(by_prompt)
    all_r = np.array([float(x["reward"]) for rows in by_prompt.values() for x in rows[:8]])
    # Secondary, scale-aware tolerance for continuous rewards: a group whose std is under 5% of the
    # pooled reward std carries almost no ranking signal even if it is not exactly zero.
    rel_tol = 0.05 * float(all_r.std()) if not binary else tol

    out = {"n_prompts": len(by_prompt), "reward_is_binary": binary, "difficulty_rule": rule,
           "tolerance": tol, "relative_tolerance": rel_tol, "partition": {}, "partition_rel_tol": {}, "coverage": {}, "by_difficulty": {}}
    rows = []
    budget = 2 * len(by_prompt)
    for k in cfg["group_sizes"]:
        k = int(k)
        groups = regroup_equal_generation_budget(by_prompt, k)
        out["partition"][k] = group_metrics(groups, full_mean, tol)
        out["partition_rel_tol"][k] = group_metrics(groups, full_mean, rel_tol)
        rng = np.random.default_rng(int(cfg["seed"]))
        reps = [group_metrics(coverage_groups(by_prompt, k, budget, rng), full_mean, tol) for _ in range(args.repeats)]
        out["coverage"][k] = {key: float(np.mean([r[key] for r in reps])) for key in reps[0] if reps[0][key] is not None}
        out["coverage"][k]["budget_completions"] = budget
        out["by_difficulty"][k] = {}
        for b in ("hard", "medium", "easy"):
            gb = [g for g in groups if bins[g[0]] == b]
            if gb:
                out["by_difficulty"][k][b] = group_metrics(gb, full_mean, tol)
                out["by_difficulty"][k][b]["n_prompts_in_bin"] = sum(1 for v in bins.values() if v == b)
        p = out["partition"][k]
        rows.append({"K": k, **{f"partition_{m}": p[m] for m in ("n_groups", "informative_group_rate", "fraction_completions_with_signal",
                                                                  "mean_within_group_reward_std", "var_centered_advantage",
                                                                  "var_normalized_advantage", "baseline_mse_vs_8sample_mean")},
                     "partition_informative_rate_reltol": out["partition_rel_tol"][k]["informative_group_rate"],
                     "coverage_prompts": out["coverage"][k]["n_prompts"],
                     "coverage_informative_rate": out["coverage"][k]["informative_group_rate"],
                     "coverage_prompts_with_signal": out["coverage"][k]["prompts_with_any_informative_group"],
                     **{f"{b}_informative_rate": out["by_difficulty"][k].get(b, {}).get("informative_group_rate") for b in ("hard", "medium", "easy")},
                     **{f"{b}_reward_std": out["by_difficulty"][k].get(b, {}).get("mean_within_group_reward_std") for b in ("hard", "medium", "easy")}})
    results = repo_path(cfg["results_dir"])
    save_json(results / "group_size_study.json", out)
    pd.DataFrame(rows).to_csv(results / "group_size_study.csv", index=False)
    print("difficulty rule:", rule)
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
