#!/usr/bin/env bash
# End-to-end smoke test of every pipeline on a tiny random model (CPU, ~2-3 minutes, no downloads).
#   bash tests/smoke/run_smoke.sh [scratch_dir]
# It exercises exactly the commands in the README, with configs pointing at tiny stand-in assets.
set -euo pipefail
cd "$(dirname "$0")/../.."
S="${1:-/tmp/pa2_smoke}"
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
rm -rf "$S"
python -m tests.test_objectives
python -m tests.smoke.make_tiny_assets "$S"
D=$S/configs/dpo.yaml; P=$S/configs/ppo.yaml; G=$S/configs/grpo.yaml; F=$S/configs/feedback.yaml
q() { echo; echo ">>> $*"; "$@" 2>&1 | tail -3; }

# Task 1
q python -m task1_dpo.train --config $D --run-name standard
q python -m task1_dpo.evaluate --config $D --adapter "$S/outputs/task1_dpo/standard" --name standard
q python -m task1_dpo.evaluate --config $D --adapter none --name sft
q python -m task1_dpo.ablate_beta --config $D
q python -m task1_dpo.analyze_length --config $D --skip-existing
# Task 2
q python -m task2_ppo.continue_train --config $P --run-name standard
q python -m task2_ppo.evaluate --config $P --adapter "$S/outputs/task2_ppo/standard" --name standard
q python -m task2_ppo.evaluate --config $P --adapter "$S/checkpoints/ppo_midpoint_policy" --name midpoint
q python -m task2_ppo.analyze_clipping --config $P
q python -m task2_ppo.ablate_kl --config $P
# Task 3
q python -m task3_grpo.continue_train --config $G --run-name standard
q python -m task3_grpo.evaluate --config $G --adapter "$S/outputs/task3_grpo/standard" --name standard
q python -m task3_grpo.analyze_group_size --config $G
q python -m task3_grpo.compare_normalization --config $G
# Task 4
q python -m task4_safety.generate_responses --config $F
q python -m task4_safety.judge_responses --config $F
q python -m task4_safety.make_audit_sheet --config $F
python - "$S" <<'EOF'
import sys, pandas as pd, random
p = f"{sys.argv[1]}/results/task4_safety/manual_audit_sheet.csv"
df = pd.read_csv(p); random.seed(0)
df["manual_label"] = [random.choice(["SAFE_ANSWER", "OVER_REFUSAL", "AMBIGUOUS", "JUSTIFIED_REFUSAL"]) for _ in range(len(df))]
df.to_csv(p, index=False)
EOF
q python -m task4_safety.evaluate_safety --config $F
# Task 5
q python -m task5_feedback.evaluate_math --config $F --dataset gsm
q python -m task5_feedback.score_perturbations --config $F
q python -m task5_feedback.evaluate_math --config $F --dataset transfer
q python -m task5_feedback.compare_feedback --config $F
# Figures
q python -m analysis.make_figures --results "$S/results" --out "$S/figures"
echo; echo "SMOKE TEST PASSED -> $S"
