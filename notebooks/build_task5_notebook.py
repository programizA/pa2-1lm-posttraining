"""Build notebooks/task5_feedback.ipynb, the guided Task 5 walkthrough. Run: python notebooks/build_task5_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# Task 5: RLVR vs RLAIF (verifiable rewards vs AI feedback)

No training here: the course supplies two frozen policies trained from the same starting point with the same
group-based RL recipe, differing only in **where the reward came from**. This notebook evaluates them against the
untouched SFT model, probes the two reward mechanisms directly on controlled answer variants, and checks transfer.

**Kaggle settings:** Accelerator **GPU T4 x2** (the code uses one), Internet **On**. No inputs need attaching.

| Section | What happens | GPU time (rough) |
|---|---|---|
| 0 | Setup: clone, install, download the math data and the two frozen policies | minutes |
| 1 | RLVR vs RLAIF from the ground up (reading) | none |
| 2 | The data: GSM8K, the SVAMP transfer set, the controlled diagnostics | none |
| 3 | The verifier, up close | none |
| 4 | The pairwise AI judge, up close | ~1 min |
| 5 | **Step 1**: in-domain (GSM8K): accuracy, format, length, judge win rate | the long one |
| 6 | **Step 2**: controlled diagnostics: what each reward actually rewards | minutes |
| 7 | **Step 3**: transfer (SVAMP) | ~20 min |
| 8 | The combined comparison | none |
| 9 | Candidates for the qualitative evidence | none |
| 10 | Figures and a zip of all results | none |

Generated responses and judge verdicts are cached on disk, so a re-run after a session drop skips finished work
(attach this notebook's previous version as an input to carry the cache over).
""")

md(r"""
## 0. Setup
""")
code(r'''
REPO_URL = "https://github.com/programizA/pa2-1lm-posttraining.git"
WORK = "/kaggle/working/pa2"
CONFIG = "configs/feedback.yaml"

import os, subprocess, sys, json, shutil, glob
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", "-q", REPO_URL, WORK], check=True)
else:
    subprocess.run(["git", "-C", WORK, "pull", "-q"], check=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
''')
code(r'''
%pip install -q -r requirements.txt
''')
code(r'''
import transformers, peft, huggingface_hub, torch
print(transformers.__version__, peft.__version__, huggingface_hub.__version__, torch.__version__, torch.cuda.is_available())
''')
code(r'''
from huggingface_hub import snapshot_download
from scripts.download_assets import HF_ASSET_REPO, HF_ASSET_REVISION
from common.data import load_yaml, repo_path, read_jsonl
cfg = load_yaml(CONFIG)
# Resume: carry over cached responses/verdicts from an attached earlier version of this notebook.
for src in sorted(glob.glob("/kaggle/input/**/pa2/results/task5_feedback", recursive=True)):
    shutil.copytree(src, os.path.join(WORK, "results/task5_feedback"), dirs_exist_ok=True); print("restored", src)
need = {"gsm_eval": cfg["paths"]["gsm_eval"], "math_transfer_eval": cfg["paths"]["math_transfer_eval"],
        "task5_diagnostics": cfg["paths"]["task5_diagnostics"], "rlvr_policy": cfg["policies"]["rlvr"], "rlaif_policy": cfg["policies"]["rlaif"]}
if not all(repo_path(p).exists() for p in need.values()):
    snapshot_download(repo_id=HF_ASSET_REPO, repo_type="dataset", revision=HF_ASSET_REVISION, local_dir=WORK,
                      allow_patterns=["data/gsm8k_*", "data/math_transfer_eval.jsonl", "data/task5_*",
                                      "checkpoints/rlvr_policy*", "checkpoints/rlaif_policy*"])
if not repo_path(cfg["paths"]["math_transfer_eval"]).exists():
    # The course installer builds the fixed 100-example SVAMP subset from the official source when it is absent.
    subprocess.run([sys.executable, "-m", "scripts.prepare_transfer_eval"], check=True)
for k, p in need.items():
    print(f"{k:20s} {'OK' if repo_path(p).exists() else 'MISSING'}  {p}")
subprocess.run([sys.executable, "-m", "tests.test_objectives"], check=True, stdout=subprocess.DEVNULL)
print({k: cfg[k] for k in ["math_max_new_tokens", "ai_judge_model", "seed"]})
''')

md(r"""
## 1. RLVR vs RLAIF from the ground up

Both policies were trained with the same group-based RL (Task 3's machinery): sample several solutions to a math
problem, score them, push up the better ones. Only the **scorer** differs.

**RLVR (verifiable rewards): a strict answer key.** The reward is 1 if the solution's designated final answer
(`#### <number>`) equals the gold answer, else 0. It cannot be talked into anything, but it is blind: it ignores the
reasoning completely, and it gives 0 to a perfect solution that forgot the `####` format. Think of a multiple-choice
scanner: fast, objective, and unable to tell lucky guesses from real understanding.

**RLAIF (AI feedback): a human-like grader.** A judge model (Qwen2.5-3B) reads two solutions to the same problem and
says which is better, or TIE. A completion's reward is its round-robin win rate within its group. It *can* read the
reasoning, but it can also be fooled: by confident, polished filler, by a wrong final answer wrapped in good-looking
working, or by which candidate happens to be shown first (the judge is orientation-balanced to reduce that).

**What the task measures.**
1. **In-domain** (GSM8K, the training distribution): exact accuracy, format compliance, length, and the judge's
   preference against the SFT model, plus how often the verifier and the judge agree.
2. **Controlled diagnostics:** for each problem, five hand-validated variants: a clean correct solution; corrupt
   reasoning that still lands on the right number; good reasoning with the wrong final; persuasive filler around a correct
   answer; and a solution that mentions the right number but ends on a wrong one. Each pair tests one thing:
   - **S_reason**: does the reward prefer clean reasoning over corrupt reasoning *when the final answer is the same*?
   - **S_outcome**: does it prefer the right final answer over a wrong one *when the reasoning looks the same*?
   - **filler susceptibility** and **distractor robustness**: style over substance, and parsing the right number.
3. **Transfer** (SVAMP, a differently-built word-problem set): does what each reward taught carry over?
""")

md(r"""
## 2. The data
""")
code(r'''
import pandas as pd, numpy as np
gsm = read_jsonl(cfg["paths"]["gsm_eval"]); svamp = read_jsonl(cfg["paths"]["math_transfer_eval"])
diag = read_jsonl(cfg["paths"]["task5_diagnostics"])
print(f"GSM8K eval: {len(gsm)} problems | SVAMP transfer: {len(svamp)} | diagnostics: {len(diag)} rows")
print("GSM row keys:", sorted(gsm[0])); print("SVAMP row keys:", sorted(svamp[0])); print("diagnostic row keys:", sorted(diag[0]))
from task5_feedback.evaluate_math import problem_text, gold_of
print("\nGSM example:", problem_text(gsm[0])[:300], "| gold:", gold_of(gsm[0]))
print("SVAMP example:", problem_text(svamp[0])[:300], "| gold:", gold_of(svamp[0]))
''')
code(r'''
# All five variants of one diagnostic problem, side by side.
from task5_feedback.score_perturbations import load_diagnostic_groups, _text, _problem, _gold
groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
pid, v = next(iter(sorted(groups.items())))
print(f"{len(groups)} problems x {len(v)} variants | problem {pid}: gold = {_gold(v['clean_correct'])}")
print("PROBLEM:", _problem(v["clean_correct"])[:400], "\n")
for name in ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final", "persuasive_filler_correct", "gold_distractor_wrong_final"]:
    print(f"--- {name}:\n{_text(v[name])[:500]}\n")
''')

md(r"""
## 3. The verifier, up close

`exact_reward` (supplied, in `task5_feedback/rlvr.py`) finds the **last** `#### <number>` in the text and compares it
numerically with the gold answer. Numbers that appear only in the working are ignored.
""")
code(r'''
import inspect
from task5_feedback.rlvr import exact_reward, extract_designated_final
print(inspect.getsource(extract_designated_final)); print(inspect.getsource(exact_reward))
display(pd.DataFrame([{"variant": n, "parsed final": extract_designated_final(_text(v[n])), "gold": _gold(v[n]),
                       "verifier reward": exact_reward(_text(v[n]), _gold(v[n]))} for n in sorted(v)]))
''')
md(r"""
Read the reward column for this problem against Section 1: the verifier should give the corrupt-reasoning variant the
same reward as the clean one (it never reads the reasoning), and 0 to both wrong-final variants.
""")

md(r"""
## 4. The pairwise AI judge, up close

`PairwiseAIJudge` (supplied, in `task5_feedback/rlaif.py`) asks the judge which of two solutions is better, or TIE,
with a fixed rubric that tells it to ignore length and confidence. To cancel position bias it **deterministically
swaps** the two candidates for about half the pairs (decided by a hash of the pair) and un-swaps the verdict. Every
verdict is cached on disk. A completion's RLAIF reward is its round-robin score in its group:
(wins + 0.5 x ties) / (K - 1).
""")
code(r'''
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
print(PAIRWISE_RUBRIC)
OUT = repo_path(cfg["results_dir"]) / "task5_feedback"; OUT.mkdir(parents=True, exist_ok=True)
judge = PairwiseAIJudge(cfg, OUT / "pairwise_judge_cache.json")
prob = _problem(v["clean_correct"])
for a, b in [("clean_correct", "good_reasoning_wrong_final"), ("good_reasoning_wrong_final", "clean_correct")]:
    print(f"A = {a:28s} B = {b:28s} -> judge says {judge.compare(prob, _text(v[a]), _text(v[b]))}")
print("A consistent judge names the same solution both times (A first, then B).")
from common.models import clear_gpu
del judge; clear_gpu()
''')

md(r"""
## 5. Step 1: in-domain evaluation on GSM8K

`evaluate_math --dataset gsm` generates one **greedy** solution per problem from SFT, RLVR and RLAIF (512-token cap),
scores each with the verifier, measures format compliance and length, and asks the judge to compare each RL policy's
solution with SFT's on the same problem. It also records where the judge and the verifier agree. Responses are saved
per policy, so they are reused if this runs again.
""")
code(r'''
_ = subprocess.run([sys.executable, "-m", "task5_feedback.evaluate_math", "--config", CONFIG, "--dataset", "gsm"], check=True)
''')
code(r'''
from common.logging_utils import load_json
def summary_table(name):
    s = load_json(OUT / f"{name}_summary.json")
    rows = []
    for pol, d in s["policies"].items():
        vs = d.get("vs_sft", {})
        rows.append({"policy": pol, "exact acc": d["exact_accuracy"], "format ok": d["format_compliance"],
                     "acc | format ok": d["accuracy_given_format"], "truncated": d["truncation_rate"],
                     "mean tokens": d["response_tokens"]["mean"], "win rate vs SFT": vs.get("win_rate"),
                     "W/T/L": f"{vs.get('wins', '-')}/{vs.get('ties', '-')}/{vs.get('losses', '-')}",
                     "judge = verifier (3-way)": vs.get("verifier_judge_agreement_3way"),
                     "judge picks correct when verifier separates": vs.get("judge_picks_verified_correct_rate")})
    print(f"{s['n_problems']} problems, greedy decoding, {s['max_new_tokens']}-token cap")
    return pd.DataFrame(rows).set_index("policy").round(3)
display(summary_table("gsm"))
''')
md(r"""
**Reading it.** "W/T/L" is the judge's verdict on each RL policy's solution against SFT's, problem by problem. The last
two columns test the judge *against* the verifier on real outputs: when exactly one of the two solutions is correct, how
often does the judge pick that one? Accuracy differences of a couple of points on a few hundred problems are within
noise; compute an interval before calling one a win.
""")

md(r"""
## 6. Step 2: controlled diagnostics
""")
code(r'''
subprocess.run([sys.executable, "-m", "task5_feedback.score_perturbations", "--config", CONFIG], check=True, stdout=subprocess.DEVNULL)
d = load_json(OUT / "diagnostic_summary.json")
display(pd.read_csv(OUT / "diagnostic_summary.csv").set_index(["category", "mechanism"]).round(3))
print({k: round(v, 3) for k, v in d.items() if k.startswith("S_") and isinstance(v, float)})
print("judge position consistency per category:", {c: round(x.get("judge_order_consistency", float("nan")), 3) for c, x in d["by_category"].items()})
display(pd.DataFrame(d["pointwise_by_variant"]).T.round(3))
''')
md(r"""
**Reading it.** Each row is one controlled pair (the clean correct solution vs a degraded one): the share of problems
where the mechanism prefers the clean one (`better_rate`), calls it a tie, or prefers the degraded one (`wrong_rate`).
Expect the verifier to tie on corrupt-reasoning pairs (S_reason near 0: it cannot see reasoning) and to be perfect on
wrong-final pairs (S_outcome = 1). The judge is the interesting one: does it notice broken reasoning, does it catch a
wrong final answer, and is it swayed by filler? The reversed-order consistency shows whether its verdicts depend on
which solution it sees first.
""")

md(r"""
## 7. Step 3: transfer to SVAMP
""")
code(r'''
_ = subprocess.run([sys.executable, "-m", "task5_feedback.evaluate_math", "--config", CONFIG, "--dataset", "transfer"], check=True)
display(summary_table("transfer"))
''')

md(r"""
## 8. The combined comparison
""")
code(r'''
subprocess.run([sys.executable, "-m", "task5_feedback.compare_feedback", "--config", CONFIG], check=True, stdout=subprocess.DEVNULL)
display(pd.read_csv(OUT / "feedback_comparison.csv").set_index("policy").T.round(3))
cmp = load_json(OUT / "feedback_comparison.json")
print("judge generations performed (cached, unique pairs):", cmp.get("judge_calls_total"))
print(cmp.get("cost_note"))
''')

md(r"""
## 9. Candidates for the qualitative evidence

Problems where the judge and the verifier **disagree** on an RL policy vs SFT: the clearest evidence of what each
reward misses. Read a few and choose honest examples.
""")
code(r'''
pairs = pd.DataFrame(read_jsonl(OUT / "gsm_pairwise_vs_sft.jsonl"))
gens = {p: {r["problem_id"]: r for r in read_jsonl(OUT / f"gsm_generations_{p}.jsonl")} for p in ("sft", "rlvr", "rlaif")}
dis = pairs[(pairs.judge != pairs.verifier) & (pairs.verifier != "TIE")]
print(f"{len(dis)} of {len(pairs)} comparisons where the verifier separates the two and the judge disagrees with it")
for _, r in dis.head(3).iterrows():
    a, b = gens[r.policy][r.problem_id], gens["sft"][r.problem_id]
    print(f"\n=== {r.problem_id}: {r.policy} correct={a['correct']} vs SFT correct={b['correct']} | verifier prefers "
          f"{'policy' if r.verifier == 'A' else 'SFT'}, judge prefers {'policy' if r.judge == 'A' else 'SFT' if r.judge == 'B' else 'neither (TIE)'}")
    print("QUESTION:", a["question"][:300]); print(f"{r.policy.upper()}:", a["response"][:400]); print("SFT:", b["response"][:400])
''')

md(r"""
## 10. Figures and a results zip
""")
code(r'''
FIGS = repo_path("report/figures")
subprocess.run([sys.executable, "-m", "analysis.make_figures", "--results", str(repo_path(cfg["results_dir"])), "--out", str(FIGS)], check=True)
shutil.make_archive("/kaggle/working/task5_results", "zip", repo_path(cfg["results_dir"]), "task5_feedback")
shutil.make_archive("/kaggle/working/task5_figures", "zip", FIGS.parent, FIGS.name)
print("wrote /kaggle/working/task5_results.zip and task5_figures.zip")
''')

md(r"""
## 11. Follow-up on your laptop (no GPU)

After unzipping `task5_results.zip` into `results/`:

```bash
python3 -m task5_feedback.analyze_outputs --config configs/feedback.yaml
python3 -m analysis.make_figures
```

`analyze_outputs` splits the verifier's zeros into "right answer but no `####` line" and "actually wrong" (a lenient
last-number check), runs paired McNemar tests against SFT on the same problems, and measures the judge's noise floor:
how often it picks a winner between two byte-identical solutions.
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "task5_feedback.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(CELLS)} cells)")
