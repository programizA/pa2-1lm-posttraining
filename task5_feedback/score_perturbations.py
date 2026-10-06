from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


# Controlled pairs: (diagnostically better, worse, what the pair isolates).
# The clean response is the reference in every pair.
CONTROLLED_PAIRS = {
    "corrupt_reasoning_correct_final": ("clean_correct", "corrupt_reasoning_correct_final", "reasoning (final held correct)"),
    "good_reasoning_wrong_final": ("clean_correct", "good_reasoning_wrong_final", "outcome (reasoning held ~fixed)"),
    "persuasive_filler_correct": ("clean_correct", "persuasive_filler_correct", "style/verbosity (both correct)"),
    "gold_distractor_wrong_final": ("clean_correct", "gold_distractor_wrong_final", "outcome + parser robustness"),
}
REASONING_PAIRS = ["corrupt_reasoning_correct_final"]
OUTCOME_PAIRS = ["good_reasoning_wrong_final", "gold_distractor_wrong_final"]


def _text(row):
    for k in ("response", "text", "completion", "candidate"):
        if row.get(k) is not None:
            return str(row[k])
    raise KeyError(f"No response text in diagnostic row keys {sorted(row)}")


def _problem(row):
    for k in ("question", "problem", "prompt"):
        if isinstance(row.get(k), str) and row[k]:
            return row[k]
    msgs = row.get("messages") or []
    return str(msgs[-1]["content"]) if msgs else ""


def _gold(row):
    for k in ("gold_final", "gold", "answer_final"):
        if row.get(k) is not None:
            return str(row[k])
    raise KeyError(f"No gold answer in diagnostic row keys {sorted(row)}")


def main():
    import json

    import pandas as pd

    from common.data import repo_path, write_jsonl
    from common.logging_utils import save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = None if args.skip_judge else PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")

    records = []
    for pid, v in sorted(groups.items()):
        problem, gold = _problem(v["clean_correct"]), _gold(v["clean_correct"])
        rewards = {name: exact_reward(_text(row), _gold(row)) for name, row in v.items()}
        for cat, (good, bad, isolates) in CONTROLLED_PAIRS.items():
            rg, rb = rewards[good], rewards[bad]
            ver = "better" if rg > rb else "worse" if rg < rb else "tie"
            rec = {"problem_id": pid, "category": cat, "isolates": isolates, "verifier_reward_better": rg,
                   "verifier_reward_worse": rb, "verifier": ver}
            if judge is not None:
                # Primary protocol: one call, (better as A, worse as B); the judge's internal hash decides
                # the on-screen order. The reversed call is a position-consistency check.
                p1 = judge.compare(problem, _text(v[good]), _text(v[bad]))
                p2 = judge.compare(problem, _text(v[bad]), _text(v[good]))
                rec["judge"] = {"A": "better", "B": "worse", "TIE": "tie"}[p1]
                rec["judge_reversed"] = {"A": "worse", "B": "better", "TIE": "tie"}[p2]
                rec["judge_order_consistent"] = rec["judge"] == rec["judge_reversed"]
            records.append(rec)

    def block(recs, key):
        n = len(recs)
        return {"n": n,
                "better_rate": sum(r[key] == "better" for r in recs) / n,
                "tie_rate": sum(r[key] == "tie" for r in recs) / n,
                "wrong_rate": sum(r[key] == "worse" for r in recs) / n}

    mechanisms = ["verifier"] + ([] if judge is None else ["judge"])
    out = {"by_category": {}, "pair_definitions": {k: {"better": g, "worse": b, "isolates": i} for k, (g, b, i) in CONTROLLED_PAIRS.items()}}
    rows = []
    for cat in CONTROLLED_PAIRS:
        recs = [r for r in records if r["category"] == cat]
        out["by_category"][cat] = {m: block(recs, m) for m in mechanisms}
        if judge is not None:
            out["by_category"][cat]["judge_order_consistency"] = sum(r["judge_order_consistent"] for r in recs) / len(recs)
        for m in mechanisms:
            rows.append({"category": cat, "mechanism": m, **block(recs, m)})
    for m in mechanisms:
        rp = [r for r in records if r["category"] in REASONING_PAIRS]
        op = [r for r in records if r["category"] in OUTCOME_PAIRS]
        out[f"S_reason_{m}"] = block(rp, m)["better_rate"]
        out[f"S_reason_{m}_tie"] = block(rp, m)["tie_rate"]
        out[f"S_reason_{m}_wrong"] = block(rp, m)["wrong_rate"]
        out[f"S_outcome_{m}"] = block(op, m)["better_rate"]
        out[f"S_outcome_{m}_tie"] = block(op, m)["tie_rate"]
        out[f"S_outcome_{m}_wrong"] = block(op, m)["wrong_rate"]
        for cat in OUTCOME_PAIRS:
            out[f"S_outcome_{m}__{cat}"] = block([r for r in records if r["category"] == cat], m)["better_rate"]

    # Pointwise view of all five variants: verifier reward and judge round-robin win rate.
    pointwise = {}
    for name in sorted(EXPECTED_VARIANTS):
        pointwise[name] = {"verifier_mean_reward": sum(exact_reward(_text(v[name]), _gold(v[name])) for v in groups.values()) / len(groups)}
    if judge is not None:
        wins = {n: 0.0 for n in EXPECTED_VARIANTS}
        for pid, v in groups.items():
            problem = _problem(v["clean_correct"])
            names = sorted(EXPECTED_VARIANTS)
            rr = judge.group_rewards(problem, [_text(v[n]) for n in names])
            for n, r in zip(names, rr):
                wins[n] += r / len(groups)
        for n in EXPECTED_VARIANTS:
            pointwise[n]["judge_round_robin_win_rate"] = wins[n]
    out["pointwise_by_variant"] = pointwise

    write_jsonl(outdir / "diagnostic_pairs.jsonl", records)
    save_json(outdir / "diagnostic_summary.json", out)
    pd.DataFrame(rows).to_csv(outdir / "diagnostic_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k.startswith("S_")}, indent=1))
    print(json.dumps(pointwise, indent=1))


if __name__ == "__main__":
    main()
