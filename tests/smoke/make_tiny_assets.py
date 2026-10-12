"""Build a tiny, fully offline stand-in for every course asset, to smoke-test the pipelines on CPU.

Creates under ROOT: a small BPE tokenizer with a Qwen-style chat template, a 2-layer random Qwen2
policy/judge, random reward/value models, LoRA "midpoint" adapters, synthetic data files in the
course schemas, the PPO rollout cache and the GRPO K=8 cache, plus configs/*.yaml pointing at them.

    python -m tests.smoke.make_tiny_assets /path/to/smoke_root
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification

REPO = Path(__file__).resolve().parents[2]
CHAT = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
WORDS = ("the a model reward policy answer question math number is of to and in why how what explain safe unsafe kill "
         "process python bomb recipe cake train data loss gradient value total each has apples more than left #### "
         "one two three four five six seven eight nine ten 0 1 2 3 4 5 6 7 8 9 10 12 15 20 .").split()


def jdump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def main(root: Path):
    rng = random.Random(0)
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)
    m = root / "models"

    # --- tokenizer
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = [sentence(rng, 30) for _ in range(400)] + [open(REPO / "data/word_limit_prompts.jsonl").read()]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token="<unk>",
                                  model_input_names=["input_ids", "attention_mask"])  # like the Qwen tokenizer
    tok.chat_template = CHAT
    tok.save_pretrained(m / "tok")

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=2048, pad_token_id=tok.pad_token_id,
                      eos_token_id=tok.eos_token_id, tie_word_embeddings=False)
    for name in ("base", "judge"):
        mdl = Qwen2ForCausalLM(cfg)
        with torch.no_grad():  # make EOS likely at varied positions, so completion lengths differ
            mdl.lm_head.weight[tok.eos_token_id] = 0.6
        mdl.save_pretrained(m / name)
        tok.save_pretrained(m / name)
    for name in ("rm", "value"):
        c = Qwen2Config(**cfg.to_dict())
        c.num_labels = 1
        mdl = Qwen2ForSequenceClassification(c)
        mdl.save_pretrained(m / name)
        tok.save_pretrained(m / name)

    ck = root / "checkpoints"
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    for name in ("ppo_midpoint_policy", "grpo_midpoint_policy", "rlvr_policy", "rlaif_policy"):
        base = Qwen2ForCausalLM.from_pretrained(m / "base")
        pm = get_peft_model(base, lcfg)
        with torch.no_grad():
            for n_, p in pm.named_parameters():
                if "lora_B" in n_:
                    p.normal_(0, 0.02)  # a midpoint that has moved away from the reference
        pm.save_pretrained(ck / name)
    # Course value checkpoint is a full merged sequence-classification model.
    Qwen2ForSequenceClassification.from_pretrained(m / "value").save_pretrained(ck / "ppo_midpoint_value")

    # --- data
    d = root / "data"
    def pref(i, stratum=None):
        q = sentence(rng, 8) + "?"
        lc, lr = {"preferred_longer": (30, 8), "rejected_longer": (8, 30), "length_matched": (15, 15), None: (rng.randint(5, 30), rng.randint(5, 30))}[stratum]
        row = {"prompt_id": f"p{i}", "prompt": q,
               "chosen": [{"role": "user", "content": q}, {"role": "assistant", "content": sentence(rng, lc)}],
               "rejected": [{"role": "user", "content": q}, {"role": "assistant", "content": sentence(rng, lr)}]}
        if stratum:
            row["length_stratum"] = stratum
        return row
    jdump(d / "dpo_standard_train.jsonl", [pref(i) for i in range(40)])
    jdump(d / "dpo_standard_eval.jsonl", [pref(100 + i) for i in range(12)])
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    jdump(d / "dpo_length_balanced_train.jsonl", [pref(200 + i, strata[i % 3]) for i in range(24)])
    jdump(d / "dpo_length_stratified_eval.jsonl", [pref(300 + i, strata[i % 3]) for i in range(12)])
    (d / "word_limit_prompts.jsonl").write_text((REPO / "data/word_limit_prompts.jsonl").read_text())
    rlp = lambda i: {"source_index": f"uf:{i}", "messages": [{"role": "user", "content": sentence(rng, 10) + "?"}]}
    train_pool = [rlp(i) for i in range(30)]
    jdump(d / "rl_prompt_pool_train.jsonl", train_pool)
    jdump(d / "rl_prompt_pool_eval.jsonl", [rlp(1000 + i) for i in range(10)])
    import pandas as pd
    xs = [{"xstest_id": i, "prompt": sentence(rng, 8), "type": ["homonyms", "privacy", "contrast_homonyms", "contrast_privacy"][i % 4],
           "benchmark_class": "SAFE" if i % 4 < 2 else "UNSAFE"} for i in range(16)]
    pd.DataFrame(xs).to_csv(d / "xstest_safety_prompts.csv", index=False)
    def mathrow(i, ds):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        q = f"Tom has {a} apples and {b} more. How many total?"
        return {"source_index": f"{ds}:{i}", "question": q, "gold_final": str(a + b),
                "messages": [{"role": "user", "content": q + "\n\nShow your reasoning and end your response with exactly `#### <number>`."}]}
    jdump(d / "gsm8k_rl_train.jsonl", [mathrow(i, "gsm") for i in range(10)])
    jdump(d / "gsm8k_eval.jsonl", [mathrow(i, "gsm") for i in range(8)])
    jdump(d / "math_transfer_eval.jsonl", [mathrow(i, "svamp") for i in range(8)])
    diag = []
    for i in range(4):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        g = a + b
        q = f"Tom has {a} apples and {b} more. How many total?"
        texts = {"clean_correct": f"{a} + {b} = {g}. #### {g}",
                 "corrupt_reasoning_correct_final": f"{a} * {b} = {a*b}, so {g}. #### {g}",
                 "good_reasoning_wrong_final": f"{a} + {b} = {g}. #### {g + 1}",
                 "persuasive_filler_correct": f"{a} + {b} = {g}. This is clearly and certainly right. #### {g}",
                 "gold_distractor_wrong_final": f"Maybe {g}? No: {a} - {b}. #### {a - b}"}
        for k, t in texts.items():
            diag.append({"problem_id": f"diag{i}", "variant_type": k, "question": q, "gold_final": str(g), "response": t, "manual_validation": True})
    jdump(d / "task5_controlled_reward_diagnostics.jsonl", diag)

    # --- caches
    c = root / "cached"
    c.mkdir(parents=True, exist_ok=True)
    roll = []
    for r in train_pool[:6]:
        resp = sentence(rng, rng.randint(4, 12))
        n = len(tok(resp, add_special_tokens=False)["input_ids"]) + 1  # + EOS
        roll.append({"source_index": r["source_index"], "response": resp,
                     "old_policy_logprobs": (-6.4 + 0.3 * torch.randn(n)).tolist(),
                     "reference_logprobs": (-6.4 + 0.3 * torch.randn(n)).tolist()})
    torch.save(roll, c / "ppo_rollout.pt")
    kc = []
    for i in range(12):
        kind = i % 3
        for gi in range(8):
            rew = {0: 1.0, 1: 0.0, 2: float(rng.random() < 0.5)}[kind] if i < 6 else rng.gauss(0, 1)
            kc.append({"source_index": f"uf:{i}", "generation_index": gi, "completion": sentence(rng, 5), "reward": rew})
    jdump(c / "grpo_k_cache.jsonl", kc)

    # --- configs
    out, res = root / "outputs", root / "results"
    base = yaml.safe_load((REPO / "configs/base.yaml").read_text())
    base.update({"base_model": str(m / "base"), "reward_model": str(m / "rm"), "reward_tokenizer": str(m / "tok"),
                 "ai_judge_model": str(m / "judge"), "value_model_init": str(m / "value"), "dtype": "float32"})
    base["paths"] = {k: str(root / v) for k, v in base["paths"].items()}
    cdir = root / "configs"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "base.yaml").write_text(yaml.safe_dump(base))
    def task_cfg(name, overrides):
        t = yaml.safe_load((REPO / f"configs/{name}.yaml").read_text())
        t["base_config"] = str(cdir / "base.yaml")
        t.update(overrides)
        (cdir / f"{name}.yaml").write_text(yaml.safe_dump(t))
    task_cfg("dpo", {"short_ablation_examples": 16, "grad_accum_steps": 2, "max_sequence_length": 96, "max_generation_tokens": 12,
                     "eval_generation_prompts": 4, "standard_output": str(out / "task1_dpo/standard"),
                     "length_output": str(out / "task1_dpo/length_balanced"), "results_dir": str(res / "task1_dpo")})
    task_cfg("ppo", {"updates": 3, "fork_updates": 2, "max_response_length": 12, "eval_max_response_length": 12, "eval_prompts": 4,
                     "output": str(out / "task2_ppo/standard"), "results_dir": str(res / "task2_ppo"),
                     "cached_rollouts": str(c / "ppo_rollout.pt")})
    task_cfg("grpo", {"updates": 3, "fork_updates": 2, "max_completion_length": 12, "eval_max_response_length": 12, "eval_prompts": 4,
                      "output": str(out / "task3_grpo/standard"),
                      "results_dir": str(res / "task3_grpo"), "group_cache": str(c / "grpo_k_cache.jsonl")})
    task_cfg("feedback", {"results_dir": str(res), "safety_max_new_tokens": 12, "math_max_new_tokens": 12, "manual_audit_per_class": 4,
                          "policies": {"sft": None, "dpo": str(out / "task1_dpo/standard"), "ppo": str(out / "task2_ppo/standard"),
                                       "grpo": str(out / "task3_grpo/standard"), "rlvr": str(ck / "rlvr_policy"), "rlaif": str(ck / "rlaif_policy")}})
    print("smoke assets ready in", root)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
