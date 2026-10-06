from __future__ import annotations

import argparse
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.rl_utils import batch_to, grad_norm, model_device, peak_vram_gib, reset_peak_vram, row_id, sequence_logprobs
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_encodable(rows, tokenizer, max_length, min_response_tokens: int = 8):
    """Indices of pairs whose prompt leaves at least `min_response_tokens` of response budget.

    encode_prompt_response raises on over-long prompts; dropping them up front keeps the run
    alive, and the dropped indices are logged so the exact training set stays reproducible.
    """
    kept, dropped = [], []
    for i, row in enumerate(rows):
        n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        (kept if n < max_length - min_response_tokens else dropped).append(i)
    return kept, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    kept, dropped = filter_encodable(rows, tokenizer, max_len)
    ids = [row_id(rows[i], i) for i in kept]
    rows = [rows[i] for i in kept]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # Dedicated generator: the shuffle order depends only on the seed, so every beta fork sees the
    # identical example order regardless of how much RNG model construction consumed.
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=make_collate(tokenizer, max_len),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "row_ids": ids,
        "dropped_indices": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def dpo_batch_forward(model, chosen, rejected, beta, train: bool):
    """Policy and frozen-reference sequence log-probs for one batch, plus the DPO loss.

    The reference is the same network with the LoRA adapter disabled (the common initialisation),
    evaluated without gradients.
    """
    device = model_device(model)
    chosen, rejected = batch_to(chosen, device), batch_to(rejected, device)
    with torch.no_grad(), reference_mode(model):
        ref_c, _ = sequence_logprobs(model, chosen)
        ref_r, _ = sequence_logprobs(model, rejected)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pol_c, len_c = sequence_logprobs(model, chosen)
        pol_r, len_r = sequence_logprobs(model, rejected)
        loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
    return loss, diag, {"pol_c": pol_c, "pol_r": pol_r, "ref_c": ref_c, "ref_r": ref_r, "len_c": len_c, "len_r": len_r}


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, optimizer, beta = bundle["model"], bundle["loader"], bundle["optimizer"], bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}.jsonl"

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_norm = float(cfg["max_grad_norm"])
    params = trainable_parameters(model)
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    n_micro = len(loader) * epochs
    n_steps = math.ceil(n_micro / accum)
    every = int(cfg.get("checkpoint_every_steps", 25))
    print(f"[DPO:{run_name}] beta={beta} examples={len(bundle['rows'])} (dropped {len(bundle['dropped_indices'])}) "
          f"micro-batches={n_micro} optimizer-steps={n_steps}", flush=True)

    # Resume support: every `every` optimizer steps the LoRA weights, optimizer state, RNG state and
    # position are saved to <output>/_resume/state.pt. Because the shuffle order is fixed by its own
    # seeded generator, a restarted run replays the same batch order and skips the finished ones.
    resume_path = output / "_resume" / "state.pt"
    step, micro, done_micro, elapsed_before = 0, 0, 0, 0.0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, t in st["params"].items():
                named[n].copy_(t.to(named[n].device))
        optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng_cpu"])
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        step, done_micro, elapsed_before = st["step"], st["micro"], st["elapsed_s"]
        kept = [l for l in log_path.read_text().splitlines() if l.strip()][:step] if log_path.exists() else []
        log_path.write_text("".join(l + "\n" for l in kept))
        print(f"  resuming from optimizer step {step}/{n_steps} (skipping {done_micro} finished micro-batches)", flush=True)
    elif log_path.exists():
        log_path.unlink()

    def save_resume():
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": {n: p.detach().cpu() for n, p in named.items()}, "optimizer": optimizer.state_dict(),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "step": step, "micro": micro, "elapsed_s": elapsed()}, resume_path)

    reset_peak_vram()
    timer = wall_timer()
    elapsed = lambda: elapsed_before + timer()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    buf = []
    for epoch in range(epochs):
        for chosen, rejected in loader:
            micro += 1
            if micro <= done_micro:
                continue  # finished before the restart
            window = min(accum, n_micro - step * accum)  # last window may be shorter
            loss, diag, _ = dpo_batch_forward(model, chosen, rejected, beta, train=True)
            (loss / window).backward()
            buf.append({k: float(v) for k, v in diag.items()} | {"loss": float(loss.item())})
            if len(buf) == window:
                gnorm = grad_norm(params)
                torch.nn.utils.clip_grad_norm_(params, max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                rec = {k: sum(b[k] for b in buf) / len(buf) for k in buf[0]}
                rec.update({"step": step, "epoch": epoch, "examples_seen": micro * int(cfg["batch_size"]),
                            "grad_norm_preclip": gnorm, "elapsed_s": elapsed(), "peak_vram_gib": peak_vram_gib()})
                append_jsonl(log_path, rec)
                buf = []
                if step % 5 == 0 or step == n_steps:
                    print(f"  step {step}/{n_steps} loss={rec['loss']:.4f} acc={rec['preference_accuracy']:.3f} "
                          f"margin={rec['margin_mean']:.3f} gnorm={gnorm:.3f} t={rec['elapsed_s']:.0f}s", flush=True)
                if every > 0 and step % every == 0 and step < n_steps:
                    save_resume()

    model.save_pretrained(str(output))
    if resume_path.exists():
        resume_path.unlink()
        resume_path.parent.rmdir()
    meta = {
        "run_name": run_name,
        "beta": beta,
        "dataset_path": bundle["dataset_path"],
        "max_examples": max_examples,
        "n_train_examples": len(bundle["rows"]),
        "dropped_indices": bundle["dropped_indices"],
        "train_row_ids": bundle["row_ids"],
        "optimizer_steps": step,
        "wall_clock_s": elapsed(),
        "peak_vram_gib": peak_vram_gib(),
        "hyperparameters": {k: cfg[k] for k in ("learning_rate", "weight_decay", "batch_size", "grad_accum_steps",
                                                 "epochs", "max_sequence_length", "max_grad_norm", "seed")},
        "lora": cfg["lora"],
        "base_model": cfg["base_model"],
    }
    save_json(output / "run_meta.json", meta)
    save_json(results_dir / f"train_{run_name}_meta.json", meta)
    print(f"[DPO:{run_name}] saved adapter to {output} ({elapsed():.0f}s, peak {peak_vram_gib()} GiB)")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
