"""Generate the thin Kaggle launcher notebooks in kaggle/*.ipynb.

The notebooks contain no experiment logic: they clone this repository, install it, restore results
from earlier notebook versions attached as inputs, and run the README commands (two GPU lanes in
parallel on a "GPU T4 x2" machine). Regenerate with: python kaggle/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = r'''# ---- EDIT THIS: your public GitHub repository ----
REPO_URL = "https://github.com/<your-username>/<your-repo>.git"
BRANCH = "main"

import os, glob, shutil, subprocess, time, pathlib
WORK = "/tmp/pa2"             # big scratch disk: repo, course assets, model caches (not saved)
OUT = "/kaggle/working"       # saved with the notebook version: results/, outputs/, logs/
os.makedirs(f"{OUT}/logs", exist_ok=True)

def sh(cmd, cwd=None, log=None):
    """Run a shell command, stream its output into the notebook, stop the notebook on failure."""
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen("set -o pipefail; " + cmd, shell=True, cwd=cwd or WORK, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, executable="/bin/bash")
    with open(log or f"{OUT}/logs/main.log", "a") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    if p.wait():
        save()
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")

def save():
    """Copy results/ and outputs/ (small LoRA adapters) to the saved output folder."""
    for d in ("results", "outputs"):
        src = pathlib.Path(WORK, d)
        if src.exists():
            shutil.copytree(src, pathlib.Path(OUT, d), dirs_exist_ok=True)

def restore():
    """Merge results/ and outputs/ from any earlier notebook versions attached via 'Add Input'."""
    found = []
    for d in ("results", "outputs"):
        for src in glob.glob(f"/kaggle/input/**/{d}", recursive=True):
            if pathlib.Path(src).is_dir() and "/tmp/" not in src:
                shutil.copytree(src, pathlib.Path(WORK, d), dirs_exist_ok=True)
                found.append(src)
    print("restored from inputs:", found or "nothing attached")

def run_lanes(lanes, poll_s=120):
    """Run several command lists in parallel, one GPU each; print each lane's latest lines while waiting."""
    procs = {}
    for name, (gpu, cmds) in lanes.items():
        script = "set -euo pipefail\n" + "\n".join(cmds) + "\n"
        path = f"{OUT}/logs/{name}.sh"
        open(path, "w").write(script)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        log = open(f"{OUT}/logs/{name}.log", "a")
        procs[name] = subprocess.Popen(["bash", path], cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"started lane {name} on GPU {gpu} -> logs/{name}.log")
    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(poll_s)
        save()
        print(f"\n===== {(time.time() - t0) / 60:.0f} min =====")
        for name in procs:
            lines = open(f"{OUT}/logs/{name}.log").read().splitlines()
            state = "running" if procs[name].poll() is None else f"exit {procs[name].returncode}"
            print(f"[{name}: {state}]", *lines[-3:], sep="\n  ")
    save()
    bad = {n: p.returncode for n, p in procs.items() if p.returncode}
    if bad:
        raise RuntimeError(f"lanes failed: {bad}; see /kaggle/working/logs/<lane>.log")
    print("all lanes finished")
'''

INSTALL = r'''sh(f"rm -rf {WORK} && git clone -q --branch {BRANCH} {REPO_URL} {WORK}", cwd="/tmp")
sh("git log --oneline -1")
sh("pip install -q -r requirements.txt")
sh("python -m scripts.download_assets && python -m scripts.validate_assets")
sh("python -m scripts.check_environment && python -m tests.test_objectives")
restore()
'''

FINISH = r'''save()
sh("ls -R results | head -80")
print("Done. Download /kaggle/working/results (Output tab) and commit it to your repo.")
'''


def guard(adapter, cmd):
    """Skip a standard run whose adapter already exists (restored from an earlier version)."""
    return f"[ -f {adapter}/adapter_config.json ] || {cmd}"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def lanes_cell(lanes):
    body = "run_lanes({\n"
    for name, (gpu, cmds) in lanes.items():
        body += f"    {name!r}: ({gpu}, [\n" + "".join(f"        {c!r},\n" for c in cmds) + "    ]),\n"
    return body + "})\n"


T1 = [
    guard("outputs/task1_dpo/standard", "python -m task1_dpo.train --config configs/dpo.yaml --run-name standard"),
    "[ -f results/task1_dpo/eval_standard.json ] || python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard",
    "[ -f results/task1_dpo/eval_sft.json ] || python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft",
    "python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-existing",
    "python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-existing",
]
T5 = [
    "python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm",
    "python -m task5_feedback.score_perturbations --config configs/feedback.yaml",
    "python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer",
    "python -m task5_feedback.compare_feedback --config configs/feedback.yaml",
]
T2 = [
    guard("outputs/task2_ppo/standard", "python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard"),
    "[ -f results/task2_ppo/eval_standard.json ] || python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard",
    "[ -f results/task2_ppo/eval_midpoint.json ] || python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint",
    "python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --skip-existing",
    "python -m task2_ppo.ablate_kl --config configs/ppo.yaml --skip-existing",
]
T3 = [
    guard("outputs/task3_grpo/standard", "python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard"),
    "[ -f results/task3_grpo/eval_standard.json ] || python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard",
    "[ -f results/task3_grpo/eval_midpoint.json ] || python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint",
    "python -m task3_grpo.analyze_group_size --config configs/grpo.yaml",
    "python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-existing",
]
T4 = [
    "python -m task4_safety.generate_responses --config configs/feedback.yaml --batch-size 8",
    "python -m task4_safety.judge_responses --config configs/feedback.yaml",
    "python -m task4_safety.make_audit_sheet --config configs/feedback.yaml",
]

NOTEBOOKS = {
    "00_check_cpu.ipynb": [
        md("""# PA2 launcher 00: environment and asset check (CPU, no GPU quota)
Settings: **Accelerator: None**, **Internet: On**. Clones the repo, installs pinned packages, downloads and
validates the course assets, runs the unit tests and the tiny-model smoke test, and prints dataset sizes."""),
        code(SETUP), code(INSTALL),
        code('sh("CUDA_VISIBLE_DEVICES= bash tests/smoke/run_smoke.sh /tmp/smoke 2>&1 | tail -25")'),
        code('sh("wc -l data/*.jsonl; python -c \\"import pandas as pd; print(\'xstest rows\', len(pd.read_csv(\'data/xstest_safety_prompts.csv\')))\\"")'),
    ],
    "01_task1_dpo_and_task5.ipynb": [
        md("""# PA2 launcher 01: Task 1 (DPO) on GPU 0 and Task 5 (RLVR vs RLAIF) on GPU 1, in parallel
Settings: **Accelerator: GPU T4 x2**, **Internet: On**. Run with *Save Version → Save & Run All (Commit)*.
To resume after a failure, attach this notebook's previous version as an input: finished steps are skipped."""),
        code(SETUP), code(INSTALL), code(lanes_cell({"task1": (0, T1), "task5": (1, T5)})), code(FINISH),
    ],
    "02_task2_ppo_and_task3_grpo.ipynb": [
        md("""# PA2 launcher 02: Task 2 (PPO) on GPU 0 and Task 3 (GRPO) on GPU 1, in parallel
Settings: **Accelerator: GPU T4 x2**, **Internet: On**. Run with *Save Version → Save & Run All (Commit)*.
Peak VRAM and wall-clock for the standard runs are in results/task2_ppo|task3_grpo/train_standard_meta.json."""),
        code(SETUP), code(INSTALL), code(lanes_cell({"task2": (0, T2), "task3": (1, T3)})), code(FINISH),
    ],
    "03_task4_safety.ipynb": [
        md("""# PA2 launcher 03: Task 4 safety generation, AI judging and the blind audit sheet
Settings: **Accelerator: GPU T4 x2**, **Internet: On**.
**Before running:** *Add Input → Your Work* → add the latest versions of launcher **01** and **02**, so the standard
DPO, PPO and GRPO adapters are restored. Afterwards: download `results/task4_safety/manual_audit_sheet.csv`,
label it by hand, then run `python -m task4_safety.evaluate_safety` (CPU, seconds) locally or in launcher 04."""),
        code(SETUP), code(INSTALL),
        code('for p in ("outputs/task1_dpo/standard", "outputs/task2_ppo/standard", "outputs/task3_grpo/standard"):\n'
             '    assert os.path.exists(f"{WORK}/{p}/adapter_config.json"), f"missing {p}: attach launcher 01 and 02 outputs"'),
        code(lanes_cell({"task4": (0, T4)})), code(FINISH),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1))
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
