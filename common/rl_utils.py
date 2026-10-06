"""Shared helpers for the training/evaluation loops (written for this submission).

Kept deliberately small: memory-efficient log-probabilities, entropy, dropout control,
peak-VRAM bookkeeping, stable row identifiers and a fixed prompt order.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- identifiers
def row_id(row: dict, index: int) -> str:
    for key in ("prompt_id", "source_index", "id", "xstest_id", "problem_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return f"idx:{index}"


def prompt_fits(tokenizer, messages, max_prompt_length: int) -> bool:
    """True if the rendered chat prompt fits the cap.

    batch_generate truncates from the right, which would cut off the assistant generation
    header, so over-long prompts are skipped (and logged) rather than silently mangled.
    """
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(ids) <= int(max_prompt_length)


def fixed_prompt_order(n_rows: int, seed: int) -> list[int]:
    """Deterministic permutation of prompt-pool indices, shared by every run/fork."""
    rng = np.random.default_rng(int(seed))
    return rng.permutation(int(n_rows)).tolist()


# ----------------------------------------------------------------------------- model helpers
def disable_dropout(model: nn.Module) -> int:
    """Set every Dropout to p=0 (incl. LoRA dropout) so that old/new log-probs agree exactly.

    Without this the PPO/GRPO ratio is != 1 on the very first epoch purely because of dropout
    noise, which inflates clip fractions. Same convention as TRL's `disable_dropout_in_model`.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            n += 1
    return n


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def token_logprobs(model, sequences, attention_mask, prompt_width: int, response_ids, with_entropy: bool = False):
    """Log-prob of each sampled response token (teacher forcing) and optional exact entropy.

    Returns (logp [B,T], entropy [B,T] or None). Only the response positions' logits are
    materialised (logits_to_keep) and the softmax is done one sequence at a time, which keeps the
    151k-vocabulary float32 buffers small.
    """
    T = response_ids.shape[1]
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        logits_to_keep=T + 1,
    )
    logits = out.logits[:, :-1, :]  # positions prompt_width-1 ... L-2 predict response tokens
    assert logits.shape[1] == T, (logits.shape, T)
    lps, ents = [], []
    for i in range(sequences.shape[0]):
        lg = logits[i].float()
        lse = torch.logsumexp(lg, dim=-1)
        lp = lg.gather(-1, response_ids[i].unsqueeze(-1)).squeeze(-1) - lse
        lps.append(lp)
        if with_entropy:
            with torch.no_grad():
                g = lg.detach()
                p = torch.softmax(g, dim=-1)
                ents.append(lse.detach() - (p * g).sum(-1))
    logp = torch.stack(lps)
    return logp, (torch.stack(ents) if with_entropy else None)


def sequence_logprobs(model, batch: dict):
    """Sum of response-token log-probs for padded prompt+response batches (DPO / pair scoring).

    Same quantity as common.generation.response_sequence_logprobs, but without allocating a
    float32 log-softmax over the full [B, L, V] tensor.
    """
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].to(logits.device)
    seq = []
    for i in range(logits.shape[0]):
        lg = logits[i].float()
        tok = lg.gather(-1, labels[i].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)
        seq.append((tok * mask[i]).sum())
    return torch.stack(seq), mask.sum(-1)


def batch_to(batch: dict, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_device(model):
    return next(model.parameters()).device


# ----------------------------------------------------------------------------- bookkeeping
def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def describe(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "median": float(med),
        "iqr": float(q3 - q1),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run_module(module: str, args: list[str]) -> None:
    """Run `python -m module args...` in a fresh process (clean GPU memory per fork)."""
    cmd = [sys.executable, "-m", module, *[str(a) for a in args]]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]))
