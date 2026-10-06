"""Task 1, Step 2: matched short-run DPO forks over beta, then a common evaluation.

Every fork starts from the original Qwen2.5-1.5B-Instruct initialisation with a fresh LoRA, sees
the same first `short_ablation_examples` training pairs in the same shuffled order (fixed seed),
and uses identical optimiser/LoRA settings. Only beta changes.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.rl_utils import run_module


def fmt(b: float) -> str:
    return f"{b:g}"


def summary_row(name, ev, budget):
    pairs, gen = ev["pairs"]["overall"], ev.get("generation", {})
    wl = ev.get("word_limit", {})
    return {
        "condition": name,
        "budget": budget,
        "beta": ev.get("beta_for_loss"),
        "heldout_dpo_loss": pairs["dpo_loss"],
        "heldout_pref_acc": pairs["preference_accuracy"],
        "heldout_margin_mean": pairs["margin"].get("mean"),
        "kl_token_mean": gen.get("kl_token_mean"),
        "kl_seq_mean": gen.get("kl_seq_mean"),
        "reward_mean": gen.get("reward", {}).get("mean"),
        "reward_std": gen.get("reward", {}).get("std"),
        "len_tokens_mean": gen.get("response_tokens", {}).get("mean"),
        "len_tokens_std": gen.get("response_tokens", {}).get("std"),
        "len_tokens_iqr": gen.get("response_tokens", {}).get("iqr"),
        "truncation_rate": gen.get("truncation_rate"),
        "word_limit_compliance": wl.get("compliance_rate"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="reuse forks/evaluations already on disk")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_root = repo_path(cfg["standard_output"]).parent
    results = repo_path(cfg["results_dir"])
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", cfg["betas"], "| short-run examples per condition:", n)

    if not args.summary_only:
        for b in cfg["betas"]:
            name = f"beta_{fmt(b)}"
            out = out_root / name
            if not (args.skip_existing and (out / "adapter_config.json").exists()):
                run_module("task1_dpo.train", ["--config", args.config, "--run-name", name, "--output", out,
                                               "--beta", b, "--max-examples", n])
            if not (args.skip_existing and (results / f"eval_{name}.json").exists()):
                run_module("task1_dpo.evaluate", ["--config", args.config, "--adapter", out, "--name", name])

    rows = []
    for name, budget in [("sft", "none (reference)"), ("standard", "1 epoch, full standard set")] + [
        (f"beta_{fmt(b)}", f"short: first {n} pairs") for b in cfg["betas"]
    ]:
        p = results / f"eval_{name}.json"
        if p.exists():
            rows.append(summary_row(name, load_json(p), budget))
        else:
            print("  (missing, skipped in table)", p)
    df = pd.DataFrame(rows)
    df.to_csv(results / "dpo_summary.csv", index=False)
    save_json(results / "dpo_summary.json", rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
