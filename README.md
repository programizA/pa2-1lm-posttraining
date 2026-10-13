# ATML PA2: LLM Post-Training (DPO, PPO, GRPO, safety, RLVR vs RLAIF)

Completed implementation of the course starter repository
(<https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining>). Course assets are pinned to
Hugging Face revision `0b350481fb03f5525a35bcdec4131bd4fe487f98` of
<https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets>.

## 0. Setup

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
python -m tests.test_objectives          # 12 unit tests for the corrected objectives
bash tests/smoke/run_smoke.sh            # optional: whole pipeline on a tiny random model, CPU, ~2 min
```

Run everything from the repository root. Every script writes machine-readable JSON/JSONL/CSV
under `results/` and adapters under `outputs/` (git-ignored). Seeds, prompt IDs, data indices
and hyperparameters are recorded in each run's `*_meta.json`.

## 1. Corrected objective defects

Each of Tasks 1-3 shipped one deliberate defect in its core objective. All three are fixed and
covered by `tests/test_objectives.py` (each test fails on the original code).

| File | Defect | Fix |
|---|---|---|
| `task1_dpo/dpo.py` | logits used `beta * (policy_margin + ref_margin)` | `beta * (policy_margin - ref_margin)`; loss is exactly log 2 when policy = reference |
| `task2_ppo/ppo.py` | clipped surrogate used `torch.maximum` | `torch.minimum` (pessimistic bound); gradient is zeroed only where the ratio has left [1-eps, 1+eps] in the advantage's direction |
| `task3_grpo/grpo.py` | advantages normalised with the whole-batch mean/std, ignoring `group_ids` | per-prompt-group mean/std; a zero-variance group gives zero advantage |

## 2. Run order (exact commands)

Dependencies: Task 4 needs the three standard policies from Tasks 1-3 Step 1. Task 5 is independent.
`--skip-existing` on the ablation drivers resumes after an interruption without redoing finished forks.

### Task 1: DPO
```bash
python -m task1_dpo.train    --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft      # reference row
python -m task1_dpo.ablate_beta    --config configs/dpo.yaml --skip-existing          # 3 short forks + eval + table
python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-existing          # length-balanced model + strata
```

### Task 2: PPO
```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard     # 20 updates, logs VRAM + time
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --skip-existing      # cached batch + 3 eps forks
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml --skip-existing      # 3 beta_KL forks
```

### Task 3: GRPO
```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.analyze_group_size    --config configs/grpo.yaml               # cache only, seconds
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-existing
```

### Task 4: Safety calibration
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # SFT, DPO, PPO, GRPO; greedy
python -m task4_safety.judge_responses    --config configs/feedback.yaml   # cached AI labels
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml   # blind sheet, see Section 4
#   -> fill results/task4_safety/manual_audit_sheet.csv (manual_label column) BEFORE opening judged_*.jsonl
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml
python -m task4_safety.refusal_check      --config configs/feedback.yaml   # transparent refusal rule; cross-checks the judge
```

### Task 5: RLVR vs RLAIF
```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

### Guided notebooks (Kaggle)
`notebooks/task<N>_*.ipynb` walk through each task step by step: they call the same functions as the
commands above, show the key source code and inspect every intermediate result. They contain no
experiment logic of their own. Settings on Kaggle: GPU T4, Internet on. Rebuild with
`python notebooks/build_task<N>_notebook.py`.

### Figures
```bash
python -m analysis.make_figures --results results --out report/figures
```

## 3. Where each piece of required evidence lives

| Requirement | File(s) |
|---|---|
| T1 DPO summary (standard + 3 beta forks, loss/acc/KL/reward/length, budgets labelled) | `results/task1_dpo/dpo_summary.csv` |
| T1 length confounding (per stratum, length, word-limit compliance; dataset vs policy) | `results/task1_dpo/length_confounding.{csv,json}` |
| T1 qualitative candidates | `eval_*_generations.jsonl`, `eval_*_word_limit.jsonl`, `eval_*_pairs.jsonl` |
| T2 standard trajectories, peak VRAM, wall-clock | `results/task2_ppo/train_standard.jsonl`, `train_standard_meta.json` |
| T2 clipping (cached clip/affected fraction, fork stability, held-out) | `clipping_cached_batch.json`, `clipping_summary.csv` |
| T2 KL pressure | `kl_summary.csv`, `kl_trajectories.csv`, `kl_first_departure.json` |
| T2 qualitative (reward vs quality) | `eval_*_generations.jsonl` (each row has reward, KL, length) |
| T3 standard trajectories, VRAM, time | `results/task3_grpo/train_standard.jsonl`, `train_standard_meta.json` |
| T3 equal-generation group size, difficulty bins | `group_size_study.{csv,json}` |
| T3 canonical vs Dr. GRPO, length-conditioned gradient statistic | `normalization_comparison.{csv,json}`, `train_norm_*_completions.jsonl` |
| T4 rates per policy, category breakdown, length | `results/task4_safety/safety_summary.csv`, `safety_by_category.csv` |
| T4 manual audit agreement, confusion, disagreements | `safety_report.json`, `audit_confusion_manual_rows_judge_cols.csv`, `audit_disagreements.csv` |
| T5 in-domain / transfer (accuracy, format, win rate, length, verifier-judge agreement) | `results/task5_feedback/{gsm,transfer}_summary.json`, `feedback_comparison.csv` |
| T5 controlled diagnostics (better/tie/wrong per category, S_reason, S_outcome) | `diagnostic_summary.{csv,json}`, `diagnostic_pairs.jsonl` |

## 4. Protocol decisions (fixed for every compared condition)

These are the choices the manual leaves open. They are configurable in `configs/*.yaml`.

**Shared**
- KL: the sampled-response estimator log pi(a_t) - log pi_ref(a_t) on the policy's own samples. Both conventions are logged everywhere: `kl_token_mean` (pooled over valid response tokens, the course `sampled_kl` helper) and `kl_seq_mean` (mean per-sequence sum). The reference is the same network with its LoRA adapter disabled (the untouched Qwen2.5-1.5B-Instruct).
- Entropy: exact next-token entropy averaged over valid response tokens; `sampled_entropy` (-mean log p of sampled tokens) is also logged.
- Held-out generation uses the config's decoding (`temperature 0.7, top_p 0.9`) with a per-batch seed schedule identical for every condition (`--greedy` available). Prompts whose rendered length exceeds the prompt cap are skipped and listed in the summary, because right-truncation would cut the assistant header.
- Rollout prompts follow one seeded permutation of the prompt pool; forks use its prefix, so every fork sees the same prompts and the same sampling seed at each update.

**Task 1**: DPO data order uses a dedicated seeded generator, so beta forks see identical batches. Generation metrics use the first 128 unique prompts of the held-out pair file (`eval_generation_prompts`). The stratum label is read from the stratified file's own field; if no recognised field exists a length-ratio rule is used and flagged as `derived`.

**Task 2**: dropout is disabled in policy and critic during RL so the epoch-0 ratio is exactly 1 (otherwise dropout noise alone inflates clip fractions). Missing-EOS responses get the configured penalty. Advantages are whitened per update. Critic behaviour is reported as value loss, explained variance and value vs return means. Stability statistic: approximate KL(old || new) per epoch (mean, max, and fraction above 0.02), plus grad-norm max. Cached-batch study: ratio of the supplied midpoint against the cached old-policy log-probs, then `ppo_epochs` simulated steps per eps from identical weights.

**Task 3**: completions are back-propagated one at a time (exactly equal to the batch loss, see the unit test), which gives each completion's own gradient norm, the length-conditioned statistic for the normalisation study. Dr. GRPO here changes only the 1/|o| normalisation (constant `max_completion_length`); the std term is kept so the fork isolates length normalisation (`dr_grpo_keep_std_normalization: false` reproduces Liu et al. fully). Group size: primary "partition" scheme splits each prompt's 8 cached completions into 8/K groups (identical samples, equal budget); secondary "coverage" scheme spends 2 x N_prompts completions on B/K prompts. Difficulty bins: tertiles of a prompt's 8-sample mean reward (fixed thresholds for binary rewards).

**Task 4**: the default audit scope is exactly 60 items: each fixed audit ID paired with one policy's response, balanced so each policy gets equal SAFE and UNSAFE prompts. `--scope all-policies` builds the 240-item version instead. The sheet is blind (no policy, class or AI label). Rates use the benchmark class as denominator; labels that presuppose the other class are reported as `class_inconsistent_rate`.

**Task 5**: greedy decoding for all math evaluation. Pairwise win rate is policy vs SFT under the fixed judge (single call, the judge's own orientation balancing); the reversed-order call is also recorded in the diagnostic study as a position-consistency check. S_outcome pools the two outcome-changing categories and is also reported per category.

## 5. Check on first contact with the real assets

The release schemas were inferred from the validation script; these loaders are deliberately
tolerant but print what they found, so glance at the first lines of output:
- `analyze_clipping` prints the cache keys and `rebuild notes` (`length_mismatch` should be 0).
- `analyze_length` prints the stratum counts; they should be equal across the three strata and `stratum_source` in `eval_*_pairs.jsonl` should say `supplied:...`.
- `score_perturbations` raises a clear error if the diagnostic rows use unexpected field names.

GPU memory: rollouts are 1 prompt per PPO update and 4 completions per GRPO update, and all
log-prob passes materialise only response-position logits, so a 16-24 GB GPU is the expected
envelope. If you hit OOM, lower `eval_batch_size` in the configs (it does not change any result
definition).

## 6. Attribution

- Starter infrastructure, configs, judges, verifier and asset tooling: course staff.
- Training loops, evaluation and analysis code, the shared helpers in `common/rl_utils.py` and
  `common/evaluation.py`, tests and figure code were written with coding assistance from an LLM
  (Claude, Anthropic), as permitted by the assignment's coding-assistance policy. No external code
  was copied. The dropout-disabling convention mirrors TRL's `disable_dropout_in_model`, and the
  missing-EOS penalty and advantage whitening mirror the TRL PPO trainer's conventions.
- Small edits to starter files: `common/generation.py` (forces KV-cache use during generation and
  leaves inference mode before training on rollouts), `common/models.py` (`token_values` resolves
  the bare backbone of a PEFT-wrapped critic and matches the head dtype).
