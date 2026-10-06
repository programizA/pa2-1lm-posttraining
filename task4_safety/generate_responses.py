from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def main():
    from common.data import write_jsonl
    from common.logging_utils import save_json
    from common.models import clear_gpu

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=None, help="subset of sft dpo ppo grpo (default: all four)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="regenerate even if the response file exists")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    specs = policy_specs(cfg)
    names = args.policies or list(specs)
    print("Policies:", names)
    print("XSTest rows:", len(load_xstest(cfg)))
    for name in names:
        if specs[name] is not None and not (repo_path(specs[name]) / "adapter_config.json").exists():
            raise SystemExit(f"Missing frozen {name} adapter at {specs[name]} (finish Tasks 1-3 Step 1 first)")

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    for name in names:
        path = outdir / f"generated_{name}.jsonl"
        if path.exists() and not args.force:
            print("  exists, skipping:", path)
            continue
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)  # fixed CSV order preserved
        write_jsonl(path, records)
        print(f"  {name}: {len(records)} responses -> {path}")
        clear_gpu()
    save_json(outdir / "generation_protocol.json", {
        "policies": {n: specs[n] for n in names}, "decoding": "greedy (do_sample=False)",
        "max_new_tokens": int(cfg["safety_max_new_tokens"]), "max_prompt_length": 256,
        "prompt_format": "single user turn, no system prompt beyond the chat template default"})


if __name__ == "__main__":
    main()
