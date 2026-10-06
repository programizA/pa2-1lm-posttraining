from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


POLICIES = ["sft", "dpo", "ppo", "grpo"]


def assign_policies(ids, meta, seed):
    """one-per-id scope: give each audited prompt exactly one policy's response, balanced so every
    policy gets the same number of SAFE and of UNSAFE prompts (15 each policy for 30+30 ids)."""
    rng = np.random.default_rng(seed + 1)
    cls = dict(zip(meta["xstest_id"].astype(int), meta["benchmark_class"]))
    out = {}
    for label in ["SAFE", "UNSAFE"]:
        group = [i for i in ids if cls[i] == label]
        rng.shuffle(group)
        for j, i in enumerate(group):
            out[i] = POLICIES[j % len(POLICIES)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--scope", choices=["one-per-id", "all-policies"], default="one-per-id",
                    help="one-per-id: 60 items (each fixed ID paired with one policy, balanced); "
                         "all-policies: 60 IDs x 4 policies = 240 items")
    ap.add_argument("--force", action="store_true", help="overwrite an existing audit sheet (destroys manual labels!)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    base = read_jsonl(src)
    ids = fixed_audit_ids(base, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet_path, key_path = outdir / "manual_audit_sheet.csv", outdir / "manual_audit_key.csv"
    if sheet_path.exists() and not args.force:
        print("Audit sheet already exists (not overwritten):", sheet_path)
        return
    responses = {}
    for p in POLICIES:
        f = outdir / f"generated_{p}.jsonl"
        if f.exists():
            responses[p] = {int(r["xstest_id"]): r for r in read_jsonl(f)}
    meta = pd.DataFrame(base)
    if args.scope == "one-per-id":
        pairs = [(i, p) for i, p in assign_policies(ids, meta, int(cfg["seed"])).items()]
    else:
        pairs = [(i, p) for i in ids for p in POLICIES]
    missing = sorted({p for _, p in pairs if p not in responses})
    if missing:
        raise SystemExit(f"Generate responses for {missing} first")
    rng = np.random.default_rng(int(cfg["seed"]) + 2)
    order = rng.permutation(len(pairs))
    sheet, key = [], []
    for n, k in enumerate(order):
        xid, pol = pairs[k]
        r = responses[pol][xid]
        item = f"A{n + 1:03d}"
        # Blind sheet: no policy name, no benchmark class, no AI label.
        sheet.append({"audit_item": item, "xstest_id": xid, "prompt": r["prompt"], "response": r["response"], "manual_label": "", "notes": ""})
        key.append({"audit_item": item, "xstest_id": xid, "policy": pol, "benchmark_class": r["benchmark_class"], "type": r["type"]})
    pd.DataFrame(sheet).to_csv(sheet_path, index=False)
    pd.DataFrame(key).to_csv(key_path, index=False)
    print(f"Wrote blind audit sheet ({len(sheet)} items, scope={args.scope}):", sheet_path)
    print("Fill `manual_label` with one of SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    print("before opening any judged_*.jsonl file. The key file maps items back to policies.")


if __name__ == "__main__":
    main()
