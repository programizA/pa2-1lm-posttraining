"""Task 5 follow-up analysis on saved outputs (CPU only, no model calls).

Three questions the summary tables cannot answer on their own:

1. Format vs maths. The verifier only reads a `#### <number>` line. How many answers are mathematically right
   but unformatted (e.g. `\\boxed{30}` or `**21**`) and so get reward 0? We re-score with a lenient rule: the
   last number in the response equals the gold answer. Lenient is a diagnostic, not a replacement metric.
2. Are the policies different at all? Exact-text overlap with SFT, and paired McNemar tests on the same problems
   for both the exact and the lenient score.
3. How noisy is the judge? When the policy's solution is byte-identical to SFT's, the only honest verdict is TIE;
   any A/B verdict there is position or sampling noise, and sets a noise floor for the win rates. We also count how
   often the judge picks the verified-correct solution when exactly one of the two is correct.

    python3 -m task5_feedback.analyze_outputs --config configs/feedback.yaml
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from math import comb

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

POLICIES = ["sft", "rlvr", "rlaif"]
DATASETS = {"gsm": "GSM8K", "transfer": "SVAMP"}
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _float(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None


def lenient_correct(response: str, gold) -> bool:
    """Last number anywhere in the response equals the gold answer."""
    nums = _NUM.findall(str(response))
    if not nums:
        return False
    a, b = _float(nums[-1]), _float(gold)
    return a is not None and b is not None and abs(a - b) < 1e-9


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    report, fmt_rows = {}, []

    for ds, ds_name in DATASETS.items():
        files = {p: outdir / f"{ds}_generations_{p}.jsonl" for p in POLICIES}
        if not all(f.exists() for f in files.values()):
            print("skipping", ds, "(generations missing)")
            continue
        g = {p: {r["problem_id"]: r for r in read_jsonl(f)} for p, f in files.items()}
        ids = list(g["sft"])
        n = len(ids)
        exact = {p: {k: bool(g[p][k]["correct"]) for k in ids} for p in POLICIES}
        len_ok = {p: {k: lenient_correct(g[p][k]["response"], g[p][k]["gold_final"]) for k in ids} for p in POLICIES}
        fmt = {p: {k: bool(g[p][k]["format_ok"]) for k in ids} for p in POLICIES}
        rep = {"n": n, "policies": {}, "paired": {}, "identical_text": {}}

        # 1. Format vs maths
        for p in POLICIES:
            right_fmt = sum(exact[p][k] for k in ids)
            right_unfmt = sum(len_ok[p][k] and not fmt[p][k] for k in ids)
            row = {"dataset": ds_name, "policy": p, "n": n,
                   "exact_accuracy": right_fmt / n, "lenient_accuracy": sum(len_ok[p].values()) / n,
                   "format_compliance": sum(fmt[p].values()) / n,
                   "correct_and_formatted": right_fmt / n,
                   "correct_but_unformatted": right_unfmt / n,
                   "other": 1 - (right_fmt + right_unfmt) / n}
            rep["policies"][p] = row
            fmt_rows.append(row)

        # 2. Are the policies different?
        for p, q in (("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")):
            same = sum(g[p][k]["response"] == g[q][k]["response"] for k in ids)
            rep["identical_text"][f"{p}_vs_{q}"] = same / n
            pr = {}
            for name, sc in (("exact", exact), ("lenient", len_ok)):
                b = sum(sc[p][k] and not sc[q][k] for k in ids)
                c = sum(sc[q][k] and not sc[p][k] for k in ids)
                pr[name] = {f"only_{p}_correct": b, f"only_{q}_correct": c, "mcnemar_p": mcnemar_exact(b, c),
                            "accuracy_diff": (b - c) / n}
            rep["paired"][f"{p}_vs_{q}"] = pr

        # 3. Judge noise floor and judge-vs-verifier on real outputs
        pf = outdir / f"{ds}_pairwise_vs_sft.jsonl"
        if pf.exists():
            judge = {}
            for r in read_jsonl(pf):
                p, k = r["policy"], r["problem_id"]
                ident = g[p][k]["response"] == g["sft"][k]["response"]
                verdict = {"A": "policy", "B": "sft", "TIE": "tie"}.get(r["judge"], r["judge"])
                d = judge.setdefault(p, {"identical": Counter(), "different": Counter(), "separated": Counter()})
                d["identical" if ident else "different"][verdict] += 1
                if exact[p][k] != exact["sft"][k]:  # verifier separates the pair: exactly one is correct
                    right = "policy" if exact[p][k] else "sft"
                    d["separated"]["picks_correct" if verdict == right else "tie" if verdict == "tie" else "picks_wrong"] += 1
            out = {}
            for p, d in judge.items():
                ni = sum(d["identical"].values())
                out[p] = {k: dict(v) for k, v in d.items()} | {
                    "decisive_rate_on_identical_pairs": (ni - d["identical"]["tie"]) / ni if ni else None,
                    "decisive_rate_on_different_pairs": (sum(d["different"].values()) - d["different"]["tie"])
                    / max(sum(d["different"].values()), 1)}
            rep["judge"] = out
        report[ds_name] = rep

    cache = outdir / "pairwise_judge_cache.json"
    if cache.exists():
        import json
        report["judge_cache_verdicts"] = dict(Counter(json.loads(cache.read_text()).values()))

    save_json(outdir / "task5_extra.json", report)
    pd.DataFrame(fmt_rows).to_csv(outdir / "task5_format_breakdown.csv", index=False)

    pd.set_option("display.width", 160)
    print(pd.DataFrame(fmt_rows).set_index(["dataset", "policy"]).round(3).to_string())
    for ds_name, rep in report.items():
        if ds_name == "judge_cache_verdicts":
            continue
        print(f"\n{ds_name}: identical text " + ", ".join(f"{k} {v:.0%}" for k, v in rep["identical_text"].items()))
        for pair, pr in rep["paired"].items():
            print(f"  {pair}: " + "; ".join(
                f"{m} {v['accuracy_diff']:+.3f} (discordant {list(v.values())[0]}/{list(v.values())[1]}, p={v['mcnemar_p']:.2f})"
                for m, v in pr.items()))
        for p, j in rep.get("judge", {}).items():
            print(f"  judge {p}: identical {j['identical']}  different {j['different']}  verifier-separated {j['separated']}")
    if "judge_cache_verdicts" in report:
        print("\nall cached judge verdicts:", report["judge_cache_verdicts"])


if __name__ == "__main__":
    main()
