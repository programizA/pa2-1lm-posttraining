"""Common held-out generation protocol used by Tasks 1-3 (written for this submission).

Every condition is evaluated with the same prompts, decoding settings, generation cap, seed
schedule, reward model and KL/entropy conventions:

* KL: released sampled-response estimator, log pi(a_t) - log pi_ref(a_t) on the policy's own
  samples. `kl_token_mean` pools all valid response tokens (common.metrics.sampled_kl);
  `kl_seq_mean` averages the per-sequence sum. Both are reported for every condition.
* Entropy: exact token-level entropy of the policy's next-token distribution, averaged over
  valid response tokens (pooled). `sampled_entropy` (-mean log p of sampled tokens) is also logged.
* Reward: mean scalar score of the fixed course reward model on (prompt, response).
* Length: generated response tokens (EOS included, padding excluded).
"""
from __future__ import annotations

import torch

from common.data import prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, word_limit_compliance, word_count
from common.models import reference_mode
from common.rl_utils import describe, prompt_fits, row_id, token_logprobs


@torch.no_grad()
def generate_and_score(
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    rows: list[dict],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int,
    gen_cfg: dict,
    seed: int,
    reward_max_length: int = 1280,
    greedy: bool = False,
    compute_kl: bool = True,
):
    """Generate one response per row and return (records, summary)."""
    policy.eval()
    fits = [prompt_fits(tokenizer, prompt_messages(r), max_prompt_length) for r in rows]
    skipped = [row_id(r, i) for i, (r, ok) in enumerate(zip(rows, fits)) if not ok]
    rows = [r for r, ok in zip(rows, fits) if ok]
    records = []
    tok_kl_num, tok_count = 0.0, 0.0
    ent_num, sent_num = 0.0, 0.0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]
        # Re-seed per batch: identical RNG schedule for every evaluated condition.
        set_seed(int(seed) + start)
        gen = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=(not greedy) and bool(gen_cfg.get("do_sample", True)),
        )
        mask = gen["response_mask"]
        rewards = score_reward_pairs(
            reward_model, reward_tokenizer, prompts, gen["responses"], max_length=reward_max_length
        ).cpu()

        lp, ent = token_logprobs(
            policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], with_entropy=True
        )
        if compute_kl:
            with reference_mode(policy):
                ref_lp, _ = token_logprobs(
                    policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
                )
        else:
            ref_lp = lp
        diff = (lp - ref_lp) * mask
        tok_kl_num += float(diff.sum().item())
        tok_count += float(mask.sum().item())
        ent_num += float((ent * mask).sum().item())
        sent_num += float((-lp * mask).sum().item())

        for j, row in enumerate(chunk):
            n = int(mask[j].sum().item())
            records.append(
                {
                    "row_index": start + j,
                    "prompt_id": row_id(row, start + j),
                    "prompt": prompts[j][-1]["content"] if prompts[j] else "",
                    "response": gen["responses"][j],
                    "response_tokens": n,
                    "response_words": word_count(gen["responses"][j]),
                    "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                    "truncated": bool(gen["truncated"][j]),
                    "reward": float(rewards[j].item()),
                    "kl_seq": float(diff[j].sum().item()),
                    "entropy_mean": float((ent[j] * mask[j]).sum().item() / max(n, 1)),
                }
            )
        print(f"  generated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    summary = {
        "n_prompts": len(records),
        "skipped_overlong_prompt_ids": skipped,
        "max_new_tokens": int(max_new_tokens),
        "decoding": "greedy" if greedy else {k: gen_cfg.get(k) for k in ("temperature", "top_p", "do_sample")},
        "seed": int(seed),
        "reward": describe(r["reward"] for r in records),
        "kl_token_mean": tok_kl_num / max(tok_count, 1.0),
        "kl_seq_mean": float(sum(r["kl_seq"] for r in records) / max(len(records), 1)),
        "entropy_token_mean": ent_num / max(tok_count, 1.0),
        "sampled_entropy_token_mean": sent_num / max(tok_count, 1.0),
        "response_tokens": describe(r["response_tokens"] for r in records),
        "response_words": describe(r["response_words"] for r in records),
        "truncation_rate": float(sum(r["truncated"] for r in records) / max(len(records), 1)),
        "eos_rate": float(sum(r["terminated_with_eos"] for r in records) / max(len(records), 1)),
    }
    return records, summary


def evaluate_rl_policy(cfg: dict, adapter: str | None, name: str, results_dir, max_prompts: int | None = None, greedy: bool = False):
    """Common held-out protocol for PPO/GRPO policies on the fixed rl_prompt_eval pool."""
    from common.data import read_jsonl, repo_path, write_jsonl
    from common.logging_utils import save_json
    from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer

    adapter = None if adapter in (None, "", "none", "sft") else adapter
    results_dir = repo_path(results_dir)
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    n = int(max_prompts or cfg.get("eval_prompts", 64))
    rows = rows[:n]
    recs, summary = generate_and_score(
        policy, tok, rm, rm_tok, rows,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg.get("eval_max_response_length", 768)),
        batch_size=int(cfg.get("eval_batch_size", 4)),
        gen_cfg=cfg["generation"], seed=int(cfg["seed"]),
        reward_max_length=int(cfg.get("reward_max_length", 1280)), greedy=greedy,
    )
    summary.update({"name": name, "adapter": adapter, "eval_prompt_pool": cfg["paths"]["rl_prompt_eval"],
                    "eval_prompt_ids": [r["prompt_id"] for r in recs]})
    write_jsonl(results_dir / f"eval_{name}_generations.jsonl", recs)
    save_json(results_dir / f"eval_{name}.json", summary)
    print(f"[eval:{name}] reward={summary['reward'].get('mean'):.3f} kl={summary['kl_token_mean']:.4f} "
          f"len={summary['response_tokens'].get('mean'):.0f} -> {results_dir / f'eval_{name}.json'}")
    clear_gpu(policy, rm)
    return summary


def word_limit_summary(records: list[dict]) -> dict:
    comp = []
    for r in records:
        c = word_limit_compliance(r["prompt"], r["response"])
        r["word_limit_compliant"] = c
        if c is not None:
            comp.append(c)
    return {
        "n_prompts_with_limit": len(comp),
        "compliance_rate": float(sum(comp) / len(comp)) if comp else None,
        "response_words": describe(r["response_words"] for r in records),
    }
