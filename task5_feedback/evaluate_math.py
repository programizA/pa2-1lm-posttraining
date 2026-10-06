"""Task 5, Steps 1 and 3: in-domain (GSM8K) and out-of-domain (SVAMP) comparison of SFT, RLVR, RLAIF.

Protocol (identical for both datasets and all three policies):
* one greedy response per problem, max `math_max_new_tokens` new tokens, the row's own messages;
* exact-answer accuracy: task5_feedback.rlvr.exact_reward (designated `#### <number>` only);
* format compliance: a designated final answer is parsed, irrespective of correctness;
* response length: generated tokens (EOS included, padding excluded);
* AI pairwise result vs SFT: fixed PairwiseAIJudge on (policy response, SFT response) with its
  deterministic orientation balancing; win=1, tie=0.5, loss=0; ties reported separately;
* verifier-judge agreement: on the same pairs, the verifier's preference is sign(r_policy - r_sft)
  (equal rewards = tie). Reported as 3-way agreement and, on pairs the verifier separates, the
  fraction where the judge picks the verified-correct response.
"""
from __future__ import annotations

import argparse
import re

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from common.rl_utils import describe, row_id
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def problem_text(row):
    if row.get("question"):
        return str(row["question"])
    msgs = row.get("messages") or []
    return str(msgs[-1]["content"]) if msgs else str(row.get("prompt", ""))


def math_messages(row):
    if isinstance(row.get("messages"), list):
        return row["messages"]
    return [{"role": "user", "content": problem_text(row) + "\n\nShow your reasoning and end your response with exactly `#### <number>`."}]


def gold_of(row):
    for k in ("gold_final", "answer_final", "gold"):
        if row.get(k) is not None:
            return str(row[k])
    ans = str(row.get("answer", ""))
    m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", ans)
    return m.group(1).replace(",", "") if m else ans


def generate_policy(cfg, name, rows, tok, batch_size):
    model = load_frozen_policy(cfg, name)
    recs = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        gen = batch_generate(model, tok, [math_messages(r) for r in chunk], max_prompt_length=int(cfg.get("math_max_prompt_length", 512)),
                             max_new_tokens=int(cfg["math_max_new_tokens"]), do_sample=False)
        for j, r in enumerate(chunk):
            resp = gen["responses"][j]
            recs.append({"problem_id": row_id(r, s + j), "policy": name, "question": problem_text(r), "gold_final": gold_of(r),
                         "response": resp, "pred_final": extract_designated_final(resp),
                         "correct": exact_reward(resp, gold_of(r)), "format_ok": extract_designated_final(resp) is not None,
                         "response_tokens": int(gen["response_lengths"][j]), "truncated": bool(gen["truncated"][j])})
        print(f"  {name}: {min(s + batch_size, len(rows))}/{len(rows)}", flush=True)
    clear_gpu(model)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-problems", type=int, help="debug only; the report uses the full fixed set")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate responses even if saved")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    if args.max_problems:
        rows = rows[: args.max_problems]
    specs = policy_specs(cfg)
    print("Rows:", len(rows), "| policies:", list(specs))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gens = {}
    for name in specs:
        path = outdir / f"{args.dataset}_generations_{name}.jsonl"
        if path.exists() and not args.force:
            gens[name] = read_jsonl(path)
            print("  loaded saved responses:", path)
            continue
        gens[name] = generate_policy(cfg, name, rows, tok, args.batch_size)
        write_jsonl(path, gens[name])

    summary = {"dataset": args.dataset, "data_path": dataset_path(cfg, args.dataset), "n_problems": len(rows),
               "decoding": "greedy", "max_new_tokens": int(cfg["math_max_new_tokens"]), "policies": {}}
    for name, recs in gens.items():
        summary["policies"][name] = {
            "exact_accuracy": sum(r["correct"] for r in recs) / len(recs),
            "format_compliance": sum(r["format_ok"] for r in recs) / len(recs),
            "accuracy_given_format": (sum(r["correct"] for r in recs) / max(sum(r["format_ok"] for r in recs), 1)),
            "truncation_rate": sum(r["truncated"] for r in recs) / len(recs),
            "response_tokens": describe(r["response_tokens"] for r in recs),
        }

    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")
        pair_rows = []
        for name in ("rlvr", "rlaif"):
            w = t = l = 0
            agree3 = sep = sep_agree = 0
            for a, b in zip(gens[name], gens["sft"]):
                assert a["problem_id"] == b["problem_id"]
                pref = judge.compare(a["question"], a["response"], b["response"])  # A = policy, B = SFT
                w += pref == "A"; t += pref == "TIE"; l += pref == "B"
                ver = "A" if a["correct"] > b["correct"] else "B" if a["correct"] < b["correct"] else "TIE"
                agree3 += pref == ver
                if ver != "TIE":
                    sep += 1
                    sep_agree += pref == ver
                pair_rows.append({"problem_id": a["problem_id"], "policy": name, "judge": pref, "verifier": ver,
                                  "policy_correct": a["correct"], "sft_correct": b["correct"]})
            n = len(gens[name])
            summary["policies"][name]["vs_sft"] = {
                "win_rate": (w + 0.5 * t) / n, "wins": w, "ties": t, "losses": l,
                "verifier_judge_agreement_3way": agree3 / n,
                "verifier_separated_pairs": sep,
                "judge_picks_verified_correct_rate": sep_agree / sep if sep else None,
            }
        summary["policies"]["sft"]["vs_sft"] = {"win_rate": 0.5, "note": "identical responses by definition"}
        write_jsonl(outdir / f"{args.dataset}_pairwise_vs_sft.jsonl", pair_rows)
        clear_gpu(judge.model)

    save_json(outdir / f"{args.dataset}_summary.json", summary)
    for name, s in summary["policies"].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "response_tokens"},
              "len=", round(s["response_tokens"]["mean"], 1))


if __name__ == "__main__":
    main()
