"""Task 5 synthesis: one table joining in-domain, transfer and controlled-diagnostic results."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, tr = load_json(d / "gsm_summary.json"), load_json(d / "transfer_summary.json")
    rows = []
    for name in ("sft", "rlvr", "rlaif"):
        g, t = gsm["policies"][name], tr["policies"][name]
        gw, tw = g.get("vs_sft", {}), t.get("vs_sft", {})
        rows.append({
            "policy": name,
            "gsm_acc": g["exact_accuracy"], "svamp_acc": t["exact_accuracy"],
            "acc_drop": g["exact_accuracy"] - t["exact_accuracy"],
            "gsm_format": g["format_compliance"], "svamp_format": t["format_compliance"],
            "gsm_winrate_vs_sft": gw.get("win_rate"), "svamp_winrate_vs_sft": tw.get("win_rate"),
            "winrate_drop": (gw["win_rate"] - tw["win_rate"]) if gw.get("win_rate") is not None and tw.get("win_rate") is not None else None,
            "gsm_ties": gw.get("ties"), "svamp_ties": tw.get("ties"),
            "gsm_len_mean": g["response_tokens"]["mean"], "svamp_len_mean": t["response_tokens"]["mean"],
            "gsm_verifier_judge_agree": gw.get("verifier_judge_agreement_3way"),
            "gsm_judge_picks_correct": gw.get("judge_picks_verified_correct_rate"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(d / "feedback_comparison.csv", index=False)
    out = {"policies": rows}
    diag_path = d / "diagnostic_summary.json"
    if diag_path.exists():
        diag = load_json(diag_path)
        out["diagnostics"] = {k: v for k, v in diag.items() if k.startswith("S_")} | {"by_category": diag["by_category"]}
    cache = d / "pairwise_judge_cache.json"
    if cache.exists():
        # Inference-cost proxy: judge generations performed (cached, so each unique pair counted once).
        out["judge_calls_total"] = len(json.loads(cache.read_text(encoding="utf-8")))
        out["cost_note"] = ("verifier: one regex per response, no model call; RLAIF group reward: K(K-1)/2 judge "
                            "generations per prompt group (6 for K=4) with a 3B judge")
    save_json(d / "feedback_comparison.json", out)
    print(df.to_string(index=False))
    if "diagnostics" in out:
        print({k: v for k, v in out["diagnostics"].items() if k != "by_category"})


if __name__ == "__main__":
    main()
