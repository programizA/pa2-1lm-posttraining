from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.evaluation import generate_and_score, word_limit_summary
from common.logging_utils import load_json, save_json
from common.metrics import word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.data import write_jsonl
from common.rl_utils import describe, row_id
from task1_dpo.train import dpo_batch_forward, filter_encodable, make_collate

STRATUM_KEYS = ("length_stratum", "stratum", "length_bucket", "bucket", "length_condition", "strata")


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def stratum_of(row: dict, ratio: float = 1.25):
    """Use the supplied stratum label; fall back to a word-length ratio rule if absent."""
    for k in STRATUM_KEYS:
        if k in row and row[k] is not None:
            return str(row[k]), "supplied:" + k
    yc, yr = preference_responses(row)
    lc, lr = max(word_count(yc), 1), max(word_count(yr), 1)
    if lc >= ratio * lr:
        return "preferred_longer", "derived"
    if lr >= ratio * lc:
        return "rejected_longer", "derived"
    return "length_matched", "derived"


@torch.no_grad()
def evaluate_pairs(policy, tokenizer, rows, cfg, beta, batch_size):
    max_len = int(cfg["max_sequence_length"])
    kept, dropped = filter_encodable(rows, tokenizer, max_len)
    rows_k = [rows[i] for i in kept]
    loader = DataLoader(rows_k, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tokenizer, max_len))
    per_pair, i = [], 0
    for chosen, rejected in loader:
        loss, _, out = dpo_batch_forward(policy, chosen, rejected, beta, train=False)
        m = (out["pol_c"] - out["ref_c"]) - (out["pol_r"] - out["ref_r"])
        losses = -torch.nn.functional.logsigmoid(beta * m)
        for j in range(m.shape[0]):
            row = rows_k[i]
            yc, yr = preference_responses(row)
            stratum, src = stratum_of(row)
            per_pair.append({
                "row_index": kept[i],
                "pair_id": row_id(row, kept[i]),
                "margin": float(m[j]),
                "dpo_loss": float(losses[j]),
                "correct": bool(m[j] > 0),
                "policy_logratio_chosen": float(out["pol_c"][j] - out["ref_c"][j]),
                "policy_logratio_rejected": float(out["pol_r"][j] - out["ref_r"][j]),
                # Raw sequence-likelihood preferences (length-sensitive; context for the DPO margin).
                "policy_likelihood_prefers_chosen": bool(out["pol_c"][j] > out["pol_r"][j]),
                "reference_likelihood_prefers_chosen": bool(out["ref_c"][j] > out["ref_r"][j]),
                "chosen_tokens": int(out["len_c"][j]),
                "rejected_tokens": int(out["len_r"][j]),
                "chosen_words": word_count(yc),
                "rejected_words": word_count(yr),
                "stratum": stratum,
                "stratum_source": src,
            })
            i += 1
    return per_pair, dropped


def summarise_pairs(per_pair):
    def block(items):
        return {
            "n": len(items),
            "preference_accuracy": sum(p["correct"] for p in items) / max(len(items), 1),
            "dpo_loss": sum(p["dpo_loss"] for p in items) / max(len(items), 1),
            "margin": describe(p["margin"] for p in items),
            "chosen_logratio_mean": sum(p["policy_logratio_chosen"] for p in items) / max(len(items), 1),
            "rejected_logratio_mean": sum(p["policy_logratio_rejected"] for p in items) / max(len(items), 1),
            "policy_likelihood_accuracy": sum(p["policy_likelihood_prefers_chosen"] for p in items) / max(len(items), 1),
            "reference_likelihood_accuracy": sum(p["reference_likelihood_prefers_chosen"] for p in items) / max(len(items), 1),
        }
    out = {"overall": block(per_pair)}
    by = defaultdict(list)
    for p in per_pair:
        by[p["stratum"]].append(p)
    out["by_stratum"] = {k: block(v) for k, v in sorted(by.items())}
    # Dataset-side length correlation: fraction of pairs where the preferred response is longer.
    out["dataset_preferred_longer_fraction"] = sum(p["chosen_tokens"] > p["rejected_tokens"] for p in per_pair) / max(len(per_pair), 1)
    return out


def generation_rows(rows, n):
    seen, out = set(), []
    for i, row in enumerate(rows):
        msgs = prompt_messages_from_preference(row)
        key = str(msgs)
        if key in seen:
            continue
        seen.add(key)
        out.append({"prompt_id": row_id(row, i), "messages": msgs})
        if len(out) >= n:
            break
    return out


def run_evaluation(config_path, adapter, name, pairs_path=None, beta=None, max_gen_prompts=None, skip_generation=False, greedy=False):
    cfg = load_yaml(config_path)
    adapter = None if adapter in (None, "", "none", "sft") else adapter
    if beta is None:
        meta_path = repo_path(adapter) / "run_meta.json" if adapter else None
        beta = float(load_json(meta_path)["beta"]) if meta_path is not None and meta_path.exists() else float(cfg["beta"])
    results_dir = repo_path(cfg["results_dir"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    bs = int(cfg.get("eval_batch_size", 4))
    seed = int(cfg["seed"])
    summary = {"name": name, "adapter": adapter, "beta_for_loss": beta}

    pairs_path = pairs_path or cfg["paths"]["dpo_standard_eval"]
    pair_rows = read_jsonl(pairs_path)
    per_pair, dropped = evaluate_pairs(policy, tokenizer, pair_rows, cfg, beta, bs)
    summary["pairs_path"] = str(pairs_path)
    summary["pairs_dropped_indices"] = dropped
    summary["pairs"] = summarise_pairs(per_pair)
    tag = "" if pairs_path == cfg["paths"]["dpo_standard_eval"] else "_" + repo_path(pairs_path).stem
    write_jsonl(results_dir / f"eval_{name}{tag}_pairs.jsonl", per_pair)
    print(f"[eval:{name}] pairs={len(per_pair)} acc={summary['pairs']['overall']['preference_accuracy']:.3f}")

    if not skip_generation:
        rm, rm_tok = load_reward_model(cfg)
        n = int(max_gen_prompts or cfg.get("eval_generation_prompts", 128))
        common = dict(max_prompt_length=int(cfg.get("max_prompt_length", 512)),
                      max_new_tokens=int(cfg["max_generation_tokens"]), batch_size=bs,
                      gen_cfg=cfg["generation"], seed=seed,
                      reward_max_length=int(cfg.get("reward_max_length", 1280)), greedy=greedy)
        gen_rows = generation_rows(read_jsonl(cfg["paths"]["dpo_standard_eval"]), n)
        recs, gsum = generate_and_score(policy, tokenizer, rm, rm_tok, gen_rows, **common)
        summary["generation"] = gsum
        write_jsonl(results_dir / f"eval_{name}_generations.jsonl", recs)

        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wl_recs, wl_gen = generate_and_score(policy, tokenizer, rm, rm_tok, wl_rows, **common)
        summary["word_limit"] = word_limit_summary(wl_recs) | {"reward": wl_gen["reward"], "kl_token_mean": wl_gen["kl_token_mean"]}
        write_jsonl(results_dir / f"eval_{name}_word_limit.jsonl", wl_recs)
        clear_gpu(rm)

    out = results_dir / f"eval_{name}{tag}.json"
    save_json(out, summary)
    print(f"[eval:{name}] wrote {out}")
    clear_gpu(policy)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", default="none", help="adapter dir, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--pairs", help="held-out pair file (default: dpo_standard_eval)")
    ap.add_argument("--beta", type=float, help="beta used to report DPO loss (default: the adapter's training beta)")
    ap.add_argument("--max-gen-prompts", type=int)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.pairs, args.beta, args.max_gen_prompts, args.skip_generation, args.greedy)


if __name__ == "__main__":
    main()
