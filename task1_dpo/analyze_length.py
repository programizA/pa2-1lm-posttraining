"""Task 1, Step 3: length-confounding study.

1. Train one DPO model on the supplied length-balanced subset (same init/optimiser/beta as standard).
2. Evaluate standard and length-balanced DPO on the supplied length-stratified held-out set,
   per stratum (preferred-longer / matched / rejected-longer).
3. Compare generated length and word-limit compliance on the common prompt sets.
4. Describe the length structure of the training DATA separately from the policy's outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import word_count
from common.rl_utils import describe, run_module
from task1_dpo.evaluate import stratum_of


def dataset_length_profile(rows):
    lc = np.array([word_count(preference_responses(r)[0]) for r in rows], dtype=float)
    lr = np.array([word_count(preference_responses(r)[1]) for r in rows], dtype=float)
    strata = Counter(stratum_of(r)[0] for r in rows)
    return {
        "n_pairs": len(rows),
        "preferred_longer_fraction": float((lc > lr).mean()) if len(rows) else None,
        "chosen_words": describe(lc),
        "rejected_words": describe(lr),
        "log_length_ratio_chosen_over_rejected": describe(np.log((lc + 1) / (lr + 1))),
        "stratum_counts": dict(strata),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    std_out, bal_out = repo_path(cfg["standard_output"]), repo_path(cfg["length_output"])
    strat_path = cfg["paths"]["dpo_length_eval"]
    strat_tag = "_" + repo_path(strat_path).stem

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(strat_path)
    print("Length-balanced train rows:", len(balanced), "| length-stratified eval rows:", len(stratified))

    if not args.summary_only:
        if not (std_out / "adapter_config.json").exists():
            raise SystemExit("Train the standard DPO model first: python -m task1_dpo.train --config configs/dpo.yaml --run-name standard")
        if not (args.skip_existing and (bal_out / "adapter_config.json").exists()):
            run_module("task1_dpo.train", ["--config", args.config, "--run-name", "length_balanced",
                                           "--dataset", cfg["paths"]["dpo_length_train"], "--output", bal_out])
        jobs = [
            ("standard", std_out, ["--pairs", strat_path, "--skip-generation"], f"eval_standard{strat_tag}.json"),
            ("length_balanced", bal_out, ["--pairs", strat_path, "--skip-generation"], f"eval_length_balanced{strat_tag}.json"),
            ("length_balanced", bal_out, [], "eval_length_balanced.json"),  # common prompts + word limits
        ]
        if not (results / "eval_standard.json").exists():
            jobs.append(("standard", std_out, [], "eval_standard.json"))
        for name, adapter, extra, fname in jobs:
            if args.skip_existing and (results / fname).exists():
                continue
            run_module("task1_dpo.evaluate", ["--config", args.config, "--adapter", adapter, "--name", name, *extra])

    report = {
        "dataset_profiles": {
            "standard_train": dataset_length_profile(read_jsonl(cfg["paths"]["dpo_standard_train"])),
            "length_balanced_train": dataset_length_profile(balanced),
            "length_stratified_eval": dataset_length_profile(stratified),
        },
        "models": {},
    }
    table = []
    for name in ("standard", "length_balanced"):
        strat = load_json(results / f"eval_{name}{strat_tag}.json")
        common = load_json(results / f"eval_{name}.json")
        gen, wl = common.get("generation", {}), common.get("word_limit", {})
        report["models"][name] = {
            "stratified_pairs": strat["pairs"],
            "generation_tokens": gen.get("response_tokens"),
            "generation_reward": gen.get("reward"),
            "kl_token_mean": gen.get("kl_token_mean"),
            "word_limit": wl,
        }
        row = {"model": name}
        for s, b in strat["pairs"]["by_stratum"].items():
            row[f"acc_{s}"] = b["preference_accuracy"]
            row[f"margin_{s}"] = b["margin"].get("mean")
            row[f"n_{s}"] = b["n"]
        row.update({
            "acc_overall_stratified": strat["pairs"]["overall"]["preference_accuracy"],
            "gen_len_mean": (gen.get("response_tokens") or {}).get("mean"),
            "gen_len_std": (gen.get("response_tokens") or {}).get("std"),
            "word_limit_compliance": wl.get("compliance_rate"),
            "word_limit_words_mean": (wl.get("response_words") or {}).get("mean"),
            "reward_mean": (gen.get("reward") or {}).get("mean"),
        })
        table.append(row)

    # Per-prompt word-limit comparison (qualitative evidence candidates).
    wl = {}
    for name in ("standard", "length_balanced", "sft"):
        p = results / f"eval_{name}_word_limit.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                wl.setdefault(r["prompt_id"], {"prompt": r["prompt"]})[name] = {
                    "words": r["response_words"], "compliant": r.get("word_limit_compliant"), "reward": r["reward"],
                    "response": r["response"]}
    report["word_limit_per_prompt"] = wl

    save_json(results / "length_confounding.json", report)
    df = pd.DataFrame(table)
    df.to_csv(results / "length_confounding.csv", index=False)
    print(df.to_string(index=False))
    print("stratum labels:", {k: v["stratum_counts"] for k, v in report["dataset_profiles"].items()})


if __name__ == "__main__":
    main()
