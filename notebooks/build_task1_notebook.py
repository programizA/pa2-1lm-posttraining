"""Build notebooks/task1_dpo.ipynb, the guided Task 1 walkthrough. Run: python notebooks/build_task1_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


# =====================================================================================
md(r"""
# Task 1: Direct Preference Optimization (DPO)

This notebook walks through Task 1 one step at a time. The experiment code itself lives in the repo
(`task1_dpo/` and `common/`), because the manual requires scripts rather than a notebook pipeline. Each
cell here calls one of those functions, shows its source where it matters, and inspects what it produced.

**Kaggle settings:** Accelerator **GPU T4** (one GPU is enough), Internet **On**.

**Every section is idempotent.** Training sections skip themselves if their saved adapter already exists,
and evaluation sections skip if their result file exists. After a session drop, just *Run All* again.

| Section | What happens | GPU time (rough) |
|---|---|---|
| 0 | Setup: clone repo, install, download Task 1 data, unit tests | minutes |
| 1 | DPO from the ground up (reading, no code) | none |
| 2 | Look at the preference data, including its length bias | none |
| 3 | The planted bug, and why the fix is correct | none |
| 4 | One real example through the model: tokens, masks, log-probs, loss | ~1 min |
| 5 | **Step 1**: train standard DPO for one epoch | the big one |
| 6 | Evaluate standard DPO and the untouched SFT model | ~10 min each |
| 7 | **Step 2**: beta forks {0.03, 0.10, 0.30} | ~3 short runs + evals |
| 8 | **Step 3**: length-balanced DPO and per-stratum evaluation | 1 run + evals |
| 9 | Candidates for the qualitative evidence | none |
| 10 | Figures and a zip of all results | none |
""")

# =====================================================================================
md(r"""
## 0. Setup

Edit `REPO_URL` to your public GitHub repo. The repo is cloned into `/kaggle/working/pa2`, so trained
adapters and results survive kernel restarts within a session and are saved with the notebook version.
""")
code(r'''
REPO_URL = "https://github.com/sramiza/<your-repo>.git"   # <- EDIT
WORK = "/kaggle/working/pa2"
CONFIG = "configs/dpo.yaml"

import os, subprocess, sys
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", "-q", REPO_URL, WORK], check=True)
else:
    subprocess.run(["git", "-C", WORK, "pull", "-q"], check=True)   # pick up any code fixes
os.chdir(WORK)
sys.path.insert(0, WORK)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
''')
code(r'''
%pip install -q -r requirements.txt
''')
code(r'''
# Download only the Task 1 files from the pinned course asset release (small JSONL files).
from huggingface_hub import snapshot_download
from scripts.download_assets import HF_ASSET_REPO, HF_ASSET_REVISION
from common.data import load_yaml, repo_path

cfg = load_yaml(CONFIG)
needed = ["dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"]
if not all(repo_path(cfg["paths"][k]).exists() for k in needed):
    snapshot_download(repo_id=HF_ASSET_REPO, repo_type="dataset", revision=HF_ASSET_REVISION,
                      local_dir=WORK, allow_patterns=["data/dpo_*"])
for k in needed + ["word_limit_prompts"]:
    p = repo_path(cfg["paths"][k])
    print(f"{k:22s} {'OK' if p.exists() else 'MISSING'}  {p}")
''')
code(r'''
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (enable a GPU accelerator!)")
# The 12 unit tests that pin the corrected objectives to the manual's equations (3 are about DPO).
subprocess.run([sys.executable, "-m", "tests.test_objectives"], check=True)
print({k: cfg[k] for k in ["beta", "betas", "learning_rate", "batch_size", "grad_accum_steps", "epochs",
                           "short_ablation_examples", "max_sequence_length", "max_generation_tokens", "seed"]})
print("LoRA:", cfg["lora"])
''')

# =====================================================================================
md(r"""
## 1. DPO from the ground up

**A language model scores whole responses by adding up token scores.** At each step the model gives a
probability to every possible next token. The probability of a whole response is the product of the
probabilities of its tokens, so its **log**-probability is a *sum*:

$$\log \pi(y \mid x) = \sum_{t} \log \pi(y_t \mid x, y_{<t})$$

Each term is negative (a probability is below 1), so a 200-token response has a much more negative
log-probability than a 20-token one. Keep that in mind; it comes back in Step 3.

**The data are comparisons, not labels.** Each row has a prompt $x$, a preferred response $y^+$ and a
rejected one $y^-$. Nobody says how good either one is in absolute terms, only which one won.

**The reference model is a photograph of the starting point.** Before training we freeze a copy of the
model, $\pi_{\text{ref}}$. In this repo that copy is the same network with the LoRA adapter switched off,
so it costs no extra memory.

**DPO grades the model on how much its opinion changed, not on its opinion.** For one pair,

$$m = \underbrace{\big[\log\pi_\theta(y^+|x) - \log\pi_{\text{ref}}(y^+|x)\big]}_{\text{how much more it now likes the winner}}
 - \underbrace{\big[\log\pi_\theta(y^-|x) - \log\pi_{\text{ref}}(y^-|x)\big]}_{\text{how much more it now likes the loser}}$$

Think of a teacher who grades students on improvement: a student who already knew the answer gets no
credit for knowing it, only for what changed. If the reference already preferred $y^+$, that head start
cancels out.

**The loss turns the margin into a probability of winning.**
$L = -\log \sigma(\beta m)$, where $\sigma$ squashes any number into (0, 1). Read $\sigma(\beta m)$ as "the
probability the model's change puts the winner ahead" (the Bradley-Terry model of comparisons).
At the start the policy equals the photograph, so $m = 0$, $\sigma(0) = 0.5$ and the loss is
$\log 2 \approx 0.693$. **Any correct implementation starts there.**

**Beta is an exchange rate.** The loss only sees $\beta m$. With a large $\beta$, a small change in
log-probabilities already looks decisive, the sigmoid flattens out sooner, and the push stops early, so
the model stays close to the reference. With a small $\beta$ the model has to move its log-probabilities
much further before the loss is satisfied, so it can drift further from the photograph. That is the
fit-versus-drift trade-off Step 2 measures. (One thing to watch: AdamW rescales gradients, so in the very
first steps the three beta runs can look almost identical; differences grow as margins grow.)

**Why length matters.** Because log-probabilities are sums, a long response offers many tokens whose
probabilities can each be nudged a little, and the nudges add up. If preferred responses in the data are
usually longer, one cheap way to grow $m$ is to favour long, verbose answers. Step 3 tests how much of
DPO's apparent success is that shortcut.
""")

# =====================================================================================
md(r"""
## 2. The preference data

First, what one row looks like, then the size of each split, and finally the length structure of the
data. The length numbers describe the **dataset**; whether the trained **policy** becomes verbose is a
separate question answered in Sections 6 to 8.
""")
code(r'''
from common.data import read_jsonl, prompt_messages_from_preference, preference_responses
from common.metrics import word_count

splits = {k: read_jsonl(cfg["paths"][k]) for k in needed}
for k, rows in splits.items():
    print(f"{k:22s} {len(rows):6d} pairs")

row = splits["dpo_standard_train"][0]
prompt = prompt_messages_from_preference(row)
y_pos, y_neg = preference_responses(row)
print("\nkeys in a row:", sorted(row))
print("\nPROMPT:", prompt[-1]["content"][:400])
print(f"\nPREFERRED ({word_count(y_pos)} words):", y_pos[:400])
print(f"\nREJECTED  ({word_count(y_neg)} words):", y_neg[:400])
''')
code(r'''
import numpy as np, matplotlib.pyplot as plt

def length_profile(rows):
    lc = np.array([word_count(preference_responses(r)[0]) for r in rows], float)
    lr = np.array([word_count(preference_responses(r)[1]) for r in rows], float)
    return lc, lr

lc, lr = length_profile(splits["dpo_standard_train"])
print(f"standard train: preferred response is longer in {np.mean(lc > lr):.1%} of pairs")
print(f"  median words  preferred {np.median(lc):.0f} | rejected {np.median(lr):.0f}")
lcb, lrb = length_profile(splits["dpo_length_train"])
print(f"length-balanced train: preferred longer in {np.mean(lcb > lrb):.1%} of pairs")

fig, ax = plt.subplots(figsize=(5, 2.4))
ax.hist(np.log((lc + 1) / (lr + 1)), bins=40, color="#2a78d6")
ax.axvline(0, color="#52514e", lw=1)
ax.set_xlabel("log(preferred words / rejected words)   (>0: preferred is longer)")
ax.set_ylabel("pairs"); ax.set_title("Length structure of the standard training set")
plt.show()
''')
code(r'''
# The stratified held-out file used in Step 3: which field carries the stratum label?
from task1_dpo.evaluate import stratum_of
from collections import Counter
strat = splits["dpo_length_eval"]
print("keys:", sorted(strat[0]))
print("strata:", Counter(stratum_of(r) for r in strat))
# Expect equal counts and a 'supplied:<field>' source. 'derived' means no stratum field was found and the
# code fell back to a length-ratio rule; tell me if you see that.
''')

# =====================================================================================
md(r"""
## 3. The planted bug, and why the fix is right

The starter computed `logits = beta * (policy_margin + ref_margin)`. The manual's equation rearranges to
`beta * (policy_margin - ref_margin)`: the reference margin is **subtracted** (the "grade on improvement"
idea from Section 1). Below is the corrected function, then both versions on the same numbers.
""")
code(r'''
import inspect
from task1_dpo.dpo import dpo_loss
print(inspect.getsource(dpo_loss))
''')
code(r'''
import torch.nn.functional as F

def starter_dpo_loss(pc, pr, rc, rr, beta):          # the original, buggy version
    return -F.logsigmoid(beta * ((pc - pr) + (rc - rr))).mean()

# Two pairs at INITIALISATION: policy == reference. Pair A: reference already prefers the winner by 30 nats.
# Pair B: reference prefers the loser by 20 nats.
ref_c = torch.tensor([-120.0, -80.0]); ref_r = torch.tensor([-150.0, -60.0])
pol_c, pol_r = ref_c.clone(), ref_r.clone()
fixed, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
print(f"fixed loss at init  : {fixed.item():.4f}   (log 2 = {np.log(2):.4f})")
print(f"starter loss at init: {starter_dpo_loss(pol_c, pol_r, ref_c, ref_r, 0.1).item():.4f}")
for i in range(2):
    s = starter_dpo_loss(pol_c[i:i+1], pol_r[i:i+1], ref_c[i:i+1], ref_r[i:i+1], 0.1).item()
    print(f"  pair {'AB'[i]} under the starter: {s:.4f}")
''')
md(r"""
With the bug, pair A is already "solved" (loss near 0) and pair B looks terrible, before any training,
purely because of what the frozen reference happened to think. The loss rewards the reference's opinion
and then double-counts it as the policy learns, instead of measuring the change. With the fix, every pair
starts at exactly log 2, and the gradient raises the winner's log-prob and lowers the loser's:
""")
code(r'''
pc = torch.tensor([-5.0], requires_grad=True); pr = torch.tensor([-5.0], requires_grad=True)
loss, _ = dpo_loss(pc, pr, torch.tensor([-5.0]), torch.tensor([-5.0]), 0.1)
loss.backward()
print(f"d loss / d logp(winner) = {pc.grad.item():+.4f}  (negative: gradient descent RAISES the winner)")
print(f"d loss / d logp(loser)  = {pr.grad.item():+.4f}  (positive: gradient descent LOWERS the loser)")
''')

# =====================================================================================
md(r"""
## 4. One real example through the model

Now the same calculation on a real pair with the real model. Three things to see:
1. **The response mask**: only response tokens count towards $\log\pi(y|x)$; prompt tokens are context.
2. **Policy vs reference** at initialisation: identical, because the fresh LoRA adapter starts at zero.
3. **The loss on this pair is log 2**, which confirms the whole pipeline (not just the formula) is right.
""")
code(r'''
from common.data import encode_prompt_response
from common.models import load_tokenizer
tok = load_tokenizer(cfg["base_model"])
ids, rmask = encode_prompt_response(tok, prompt, y_pos, int(cfg["max_sequence_length"]))
first = rmask.index(1)
print(f"{len(ids)} tokens: {first} prompt + {sum(rmask)} response (EOS included)")
print("\ntokens around the prompt/response boundary (mask 0 = prompt, 1 = response):")
for i in range(max(0, first - 6), min(len(ids), first + 6)):
    print(f"  {rmask[i]}  {tok.decode([ids[i]])!r}")
''')
code(r'''
from task1_dpo.train import make_collate, dpo_batch_forward
from common.models import load_policy, clear_gpu

model = load_policy(cfg, trainable=False, fresh_lora=True)      # base model + fresh (zero) LoRA
chosen, rejected = make_collate(tok, int(cfg["max_sequence_length"]))([row])
loss, diag, out = dpo_batch_forward(model, chosen, rejected, cfg["beta"], train=False)
print(f"log pi(y+|x): policy {out['pol_c'].item():9.2f} | reference {out['ref_c'].item():9.2f}  over {int(out['len_c'])} tokens")
print(f"log pi(y-|x): policy {out['pol_r'].item():9.2f} | reference {out['ref_r'].item():9.2f}  over {int(out['len_r'])} tokens")
print(f"margin m = {diag['margin_mean'].item():.4f}   loss = {loss.item():.4f}   (expect 0 and 0.6931)")
del model; clear_gpu()
''')

# =====================================================================================
md(r"""
## 5. Step 1: train standard DPO (one epoch)

`run_training` (in `task1_dpo/train.py`) does, for every batch of two pairs:
1. compute the **reference** log-probs of both responses with the adapter switched off, no gradients;
2. compute the **policy** log-probs with the adapter on, with gradients;
3. compute the DPO loss, divide by the accumulation window, and back-propagate;
4. every `grad_accum_steps` = 8 batches (16 pairs): clip the gradient norm to 1.0 and take one AdamW step.

Only the LoRA matrices on the attention `q_proj` and `v_proj` layers train (a few million of 1.5 billion
weights). Each optimiser step appends one line to `results/task1_dpo/train_standard.jsonl`. The step
printout shows the loss, preference accuracy, mean margin, gradient norm and elapsed seconds; use the
elapsed time after the first few steps to estimate the full run.

**If the session drops mid-epoch,** just Run All again. Every 25 optimiser steps the run saves its LoRA
weights, optimiser state and position to `outputs/task1_dpo/standard/_resume/`, and a restarted run picks
up from there, replaying the same batch order. (Tested: an interrupted-and-resumed run ends with weights
identical to an uninterrupted one.)
""")
code(r'''
from task1_dpo.train import run_training
from common.data import repo_path

std_out = repo_path(cfg["standard_output"])
if (std_out / "adapter_config.json").exists():
    print("Standard DPO adapter found, skipping training:", std_out)
else:
    run_training(CONFIG, run_name="standard")
clear_gpu()
''')
md(r"""
**Reading the training curves.** Loss should fall from about 0.693. Preference accuracy is measured on
the training batches, so it is optimistic. The last panel shows the two halves of the margin separately:
$\beta[\log\pi_\theta - \log\pi_{\text{ref}}]$ for the winner and for the loser. DPO often lowers *both*,
the loser faster; check whether yours does. That is a good viva question, because it means "the model
prefers the winner more" can happen while the winner itself becomes less likely.
""")
code(r'''
import pandas as pd
log = pd.read_json(repo_path(cfg["results_dir"]) / "train_standard.jsonl", lines=True)
fig, axes = plt.subplots(1, 4, figsize=(11, 2.3))
for ax, (k, t) in zip(axes, [("loss", "DPO loss"), ("preference_accuracy", "train-batch pref. accuracy"),
                             ("margin_mean", "margin m")]):
    ax.plot(log["step"], log[k], color="#2a78d6"); ax.set_title(t); ax.set_xlabel("optimizer step")
axes[0].axhline(np.log(2), color="#52514e", ls="--", lw=1)
axes[3].plot(log["step"], log["chosen_reward_mean"], color="#2a78d6", label="winner")
axes[3].plot(log["step"], log["rejected_reward_mean"], color="#eb6834", label="loser")
axes[3].set_title("beta * (log pi - log ref)"); axes[3].legend(frameon=False); axes[3].set_xlabel("optimizer step")
plt.tight_layout(); plt.show()
meta = pd.read_json(repo_path(cfg["results_dir"]) / "train_standard_meta.json", typ="series")
print(f"{meta['n_train_examples']} pairs, {meta['optimizer_steps']} optimizer steps, "
      f"{meta['wall_clock_s'] / 60:.1f} min, peak VRAM {meta['peak_vram_gib']} GiB")
''')

# =====================================================================================
md(r"""
## 6. Evaluate standard DPO and the untouched SFT model

`run_evaluation` (in `task1_dpo/evaluate.py`) measures, for one model:
- **held-out preference accuracy and DPO loss**: the margin $m$ on every held-out pair; accuracy is the
  fraction with $m > 0$. For the SFT model the policy *is* the reference, so $m = 0$ everywhere and its
  accuracy is 0 by definition. That row is only a reference point for the metrics below.
- **generation metrics** on the first 128 held-out prompts, sampled with the config's decoding settings
  and a fixed seed schedule: reward-model score, KL from the reference, and response length.
- **word-limit compliance** on the 10 fixed prompts that state a word limit.

The **KL** here is the manual's sampled estimator: on the model's own responses, the average of
$\log\pi_\theta(a_t) - \log\pi_{\text{ref}}(a_t)$ over response tokens (`kl_token_mean`), and the
per-response sum (`kl_seq_mean`). Read it as "how surprised the photograph would be by what the trained
model now says".
""")
code(r'''
from task1_dpo.evaluate import run_evaluation
from common.logging_utils import load_json
RES = repo_path(cfg["results_dir"])

for name, adapter in [("standard", str(std_out)), ("sft", "none")]:
    if (RES / f"eval_{name}.json").exists():
        print(f"eval_{name}.json found, skipping")
    else:
        run_evaluation(CONFIG, adapter, name)
    clear_gpu()
''')
code(r'''
from task1_dpo.ablate_beta import summary_row
rows = [summary_row(n, load_json(RES / f"eval_{n}.json"), b)
        for n, b in [("sft", "reference"), ("standard", "1 epoch, full set")]]
pd.DataFrame(rows).set_index("condition").T
''')

# =====================================================================================
md(r"""
## 7. Step 2: the beta forks

Three short runs, each from the **original** model with a fresh adapter, on the same first 600 training
pairs in the same order (the shuffle has its own fixed seed). Only beta changes, so any difference is
caused by beta. These forks use a much smaller budget than the standard run, which the summary table
labels; do not compare a fork's numbers with the standard run as if they were matched.
""")
code(r'''
from task1_dpo.ablate_beta import fmt
out_root = std_out.parent
n_short = int(cfg["short_ablation_examples"])
for b in cfg["betas"]:
    name = f"beta_{fmt(b)}"
    out = out_root / name
    if (out / "adapter_config.json").exists():
        print(name, "adapter found, skipping training")
    else:
        run_training(CONFIG, run_name=name, output_path=str(out), beta=float(b), max_examples=n_short)
    clear_gpu()
    if not (RES / f"eval_{name}.json").exists():
        run_evaluation(CONFIG, str(out), name)
    clear_gpu()
''')
code(r'''
# Same table the script writes (results/task1_dpo/dpo_summary.csv).
subprocess.run([sys.executable, "-m", "task1_dpo.ablate_beta", "--config", CONFIG, "--summary-only"],
               check=True, stdout=subprocess.DEVNULL)   # errors still print
summary = pd.read_csv(RES / "dpo_summary.csv").set_index("condition")
display(summary.T)

fk = summary[summary.index.str.startswith("beta_")].sort_values("beta")
fig, axes = plt.subplots(1, 4, figsize=(11, 2.3))
for ax, (k, t) in zip(axes, [("heldout_pref_acc", "held-out pref. accuracy"), ("kl_token_mean", "KL to reference"),
                             ("reward_mean", "reward-model score"), ("len_tokens_mean", "response length (tokens)")]):
    ax.plot(fk["beta"], fk[k], marker="o", color="#2a78d6"); ax.set_xscale("log")
    ax.set_xticks(fk["beta"], [f"{b:g}" for b in fk["beta"]]); ax.minorticks_off(); ax.set_title(t); ax.set_xlabel("beta")
plt.tight_layout(); plt.show()
''')
md(r"""
**Research question 1** asks whether the relationship is monotonic over the tested range. Three points can
show a trend or a turn, but not much more; say how big the differences are relative to the run-to-run
noise you would expect from 600 pairs before calling a trend real.
""")

# =====================================================================================
md(r"""
## 8. Step 3: the length-confounding study

One more model, trained exactly like the standard run but on the supplied **length-balanced** subset:
equal numbers of pairs where the winner is clearly longer, roughly the same length, and clearly shorter.
Then both models are scored on the **length-stratified** held-out set, separately per stratum.

How to read it: if standard DPO learned "longer wins", it should do well on *preferred-longer* pairs and
badly on *rejected-longer* pairs, where length points the wrong way. If balancing the data removes that
shortcut, the gap between those two strata should shrink. Generated length and word-limit compliance show
whether the shortcut also shows up in the model's own outputs.
""")
code(r'''
bal_out = repo_path(cfg["length_output"])
strat_path = cfg["paths"]["dpo_length_eval"]
tag = "_" + repo_path(strat_path).stem

if (bal_out / "adapter_config.json").exists():
    print("length-balanced adapter found, skipping training")
else:
    run_training(CONFIG, run_name="length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=str(bal_out))
clear_gpu()
jobs = [("standard", std_out, strat_path, True), ("length_balanced", bal_out, strat_path, True),
        ("length_balanced", bal_out, None, False)]
for name, adapter, pairs, skip_gen in jobs:
    fname = f"eval_{name}{tag if pairs else ''}.json"
    if (RES / fname).exists():
        print(fname, "found, skipping")
    else:
        run_evaluation(CONFIG, str(adapter), name, pairs_path=pairs, skip_generation=skip_gen)
    clear_gpu()
''')
code(r'''
subprocess.run([sys.executable, "-m", "task1_dpo.analyze_length", "--config", CONFIG, "--summary-only"],
               check=True, stdout=subprocess.DEVNULL)
lc_tab = pd.read_csv(RES / "length_confounding.csv").set_index("model")
display(lc_tab.T)

# Stratum names come from the data file itself (e.g. preferred_longer / length_matched / rejected_longer).
strata = sorted((c[4:] for c in lc_tab.columns if c.startswith("acc_") and c != "acc_overall_stratified"),
                key=lambda s: 0 if s.startswith("pref") else 2 if s.startswith("rej") else 1)
x = np.arange(len(strata)); w = 0.38
fig, ax = plt.subplots(figsize=(5.5, 2.6))
for i, (m, c) in enumerate([("standard", "#2a78d6"), ("length_balanced", "#eb6834")]):
    ax.bar(x + (i - 0.5) * w, [lc_tab.loc[m, f"acc_{s}"] for s in strata], w * 0.94, color=c, label=m)
ax.set_xticks(x, [s.replace("_", "-") for s in strata]); ax.set_ylim(0, 1); ax.set_ylabel("held-out pref. accuracy")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5)); plt.show()
''')

# =====================================================================================
md(r"""
## 9. Candidates for the qualitative evidence

The manual asks for brief examples of (i) a case where the preference or reward signal and the actual
quality disagree, and (ii) behaviour tied to length bias or instruction following. These cells only
**surface candidates**; reading them and choosing honest examples is your call.
""")
code(r'''
def gens(name):
    p = RES / f"eval_{name}_generations.jsonl"
    return {r["prompt_id"]: r for r in read_jsonl(p)} if p.exists() else {}

std_g, sft_g = gens("standard"), gens("sft")
both = [k for k in std_g if k in sft_g]
cand = sorted(both, key=lambda k: std_g[k]["reward"] - sft_g[k]["reward"], reverse=True)
print("(i) Biggest reward-model GAINS of standard DPO over SFT. Check whether the response is actually better,")
print("    or just longer / more confident:\n")
for k in cand[:3]:
    s, b = std_g[k], sft_g[k]
    print(f"--- {k}: reward {b['reward']:.2f} -> {s['reward']:.2f} | tokens {b['response_tokens']} -> {s['response_tokens']}")
    print("PROMPT:", s["prompt"][:250]); print("SFT:", b["response"][:350]); print("DPO:", s["response"][:350], "\n")
''')
code(r'''
from common.metrics import parse_word_limit
print("(ii) Word-limit prompts: words written vs the stated limit, per model\n")
wl = {}
for name in ("sft", "standard", "length_balanced"):
    p = RES / f"eval_{name}_word_limit.jsonl"
    if p.exists():
        for r in read_jsonl(p):
            wl.setdefault(r["prompt_id"], {"limit": parse_word_limit(r["prompt"]), "prompt": r["prompt"][:70]})[name] = r["response_words"]
display(pd.DataFrame(wl).T)
''')

# =====================================================================================
md(r"""
## 10. Figures and a results zip

The repo's figure script renders the Task 1 figures as PDF for the report. Everything in `results/` is
small JSON/CSV and belongs in your GitHub repo (the manual requires machine-readable logs for every
reported number). Download the zip from the Output panel and commit its contents.
""")
code(r'''
import shutil
FIGS = repo_path("report/figures")
subprocess.run([sys.executable, "-m", "analysis.make_figures", "--results", str(RES.parent), "--out", str(FIGS)], check=True)
OUTDIR = "/kaggle/working"
shutil.make_archive(f"{OUTDIR}/task1_results", "zip", RES.parent, RES.name)
shutil.make_archive(f"{OUTDIR}/task1_figures", "zip", FIGS.parent, FIGS.name)
print(f"wrote {OUTDIR}/task1_results.zip and {OUTDIR}/task1_figures.zip")
''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "task1_dpo.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(CELLS)} cells)")
