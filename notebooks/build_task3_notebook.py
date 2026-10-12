"""Build notebooks/task3_grpo.ipynb, the guided Task 3 walkthrough. Run: python notebooks/build_task3_notebook.py"""
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
# Task 3: Group Relative Policy Optimization (GRPO)

Same structure as the Task 1 and 2 notebooks: each cell calls functions from the repo (`task3_grpo/`,
`common/`), shows the important source, and inspects the result.

**Kaggle settings:** Accelerator **GPU T4 x2** (the code uses one), Internet **On**.

**Every section is idempotent:** training skips if the adapter exists, evaluation skips if its result exists.

| Section | What happens | GPU time (rough) |
|---|---|---|
| 0 | Setup: clone, install, download the GRPO assets, unit tests | minutes |
| 1 | GRPO from the ground up (reading, no code) | none |
| 2 | The supplied K=8 completion cache | none |
| 3 | The planted bug, and why the fix is correct | none |
| 4 | One group under the hood: 4 completions, rewards, advantages, masks, weights | ~1 min |
| 5 | **Step 1**: the standard 20-update continuation (VRAM + time recorded) | ~10-15 min |
| 6 | Evaluate the midpoint and the standard run on held-out prompts | ~7 min each |
| 7 | **Step 2**: group size K in {2, 4, 8} at equal generations (cache only) | none |
| 8 | **Step 3**: canonical GRPO vs Dr. GRPO forks, length-conditioned | 2 forks + evals |
| 9 | Candidates for the qualitative evidence | none |
| 10 | Figures and a zip of all results | none |
""")

# =====================================================================================
md(r"""
## 0. Setup
""")
code(r'''
REPO_URL = "https://github.com/programizA/pa2-1lm-posttraining.git"
WORK = "/kaggle/working/pa2"
CONFIG = "configs/grpo.yaml"

import os, subprocess, sys, json, shutil
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
# OPTIONAL resume: if an earlier version of this notebook was attached via "Add Input -> Your Work",
# copy its finished adapters and results here, so training/evaluation that already ran is skipped.
import glob
for sub in ("outputs/task3_grpo", "results/task3_grpo"):
    for src in sorted(glob.glob(f"/kaggle/input/**/pa2/{sub}", recursive=True)):
        shutil.copytree(src, os.path.join(WORK, sub), dirs_exist_ok=True)
        print("restored", src)
''')
code(r'''
import transformers, peft, huggingface_hub, torch
print(transformers.__version__, peft.__version__, huggingface_hub.__version__, torch.__version__, torch.cuda.is_available())
# Expect 4.57.1 0.17.1 0.36.2 ... True. The pip "dependency conflict" warnings about gradio/diffusers are harmless.
''')
code(r'''
from huggingface_hub import snapshot_download
from scripts.download_assets import HF_ASSET_REPO, HF_ASSET_REVISION
from common.data import load_yaml, repo_path

cfg = load_yaml(CONFIG)
needed = ["rl_prompt_train", "rl_prompt_eval", "grpo_midpoint_policy"]
if not (all(repo_path(cfg["paths"][k]).exists() for k in needed) and repo_path(cfg["group_cache"]).exists()):
    snapshot_download(repo_id=HF_ASSET_REPO, repo_type="dataset", revision=HF_ASSET_REVISION, local_dir=WORK,
                      allow_patterns=["data/rl_prompt_pool_*", "cached/grpo_k_cache.jsonl", "checkpoints/grpo_midpoint_*"])
for k in needed:
    p = repo_path(cfg["paths"][k]); print(f"{k:22s} {'OK' if p.exists() else 'MISSING'}  {p}")
print(f"{'grpo_k_cache':22s} {'OK' if repo_path(cfg['group_cache']).exists() else 'MISSING'}")
subprocess.run([sys.executable, "-m", "tests.test_objectives"], check=True, stdout=subprocess.DEVNULL)
print("12 unit tests passed")
print({k: cfg[k] for k in ["updates", "fork_updates", "prompts_per_update", "policy_epochs", "learning_rate", "clip_epsilon",
                           "kl_beta", "num_generations", "group_sizes", "max_completion_length", "mask_truncated_completions",
                           "dr_grpo_keep_std_normalization", "eval_prompts", "eval_max_response_length"]})
''')

# =====================================================================================
md(r"""
## 1. GRPO from the ground up

**PPO needed a critic to say what "expected" means.** GRPO drops the critic. Instead, for each prompt it
samples a **group** of $K$ completions and uses the group itself as the yardstick.

**Grading on a curve.** Think of $K$ students answering the same exam question. Rather than asking "is this
score good in absolute terms?", you ask "is it better than the others who got the same question?". Each
completion's advantage is its reward relative to its group:

$$A_k = \frac{r_k - \text{mean}(r_1..r_K)}{\text{std}(r_1..r_K) + \epsilon}$$

Every token of completion $k$ gets the same advantage $A_k$. Above-average answers are pushed up, below-average
ones pushed down, and the group's average cancels out, so an easy prompt and a hard prompt both produce a useful
"which answer was better" signal.

**When the group gives no signal.** If all $K$ completions get the same reward, every advantage is zero and the
prompt teaches nothing: an **uninformative group**. Bigger groups make this less likely (more chances for one
answer to differ), but cost more generations per prompt. That is the trade-off Step 2 measures.

**The rest is PPO's machinery:** the same ratio $\rho_t = \pi_\theta / \pi_{\text{old}}$, the same clipped
surrogate, and a KL penalty. One difference: here the KL penalty sits **in the loss** (estimator
$e^{x} - x - 1$ with $x = \log\pi_{\text{ref}} - \log\pi_\theta$, always $\ge 0$), not in the reward.

**Truncated completions are masked.** A completion that hits the length cap never finished, so its reward
judges a cut-off answer. With masking on, such completions still count towards the group's mean and std,
but contribute no gradient themselves.

**How a sequence's tokens are averaged (Step 3).**
- **Canonical GRPO** averages over the completion's own length: each token weighs $|A_k| / |o_k|$. A
  long wrong answer has its blame spread thinly over many tokens, so each token is barely penalised; a short
  wrong answer is penalised hard per token. Over time this can quietly favour longer wrong answers.
- **Dr. GRPO** divides by a constant (the generation cap) instead, so every token weighs the same no matter how
  long its completion is. Long completions then carry proportionally more total weight.
""")

code(r'''
# How far is the supplied midpoint from the base model? (adapter on vs off on a few training prompts)
from common.rl_utils import adapter_effect
from common.models import load_policy, load_tokenizer, clear_gpu
from common.data import read_jsonl, prompt_messages
_tok = load_tokenizer(cfg["base_model"])
_mid = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=False)
print(adapter_effect(_mid, _tok, [prompt_messages(r) for r in read_jsonl(cfg["paths"]["rl_prompt_train"])[:5]]))
print("A mean change of ~0.001 nats per token means the midpoint is almost the base model, i.e. KL to the reference starts near 0.")
del _mid; clear_gpu()
''')

# =====================================================================================
md(r"""
## 2. The supplied K=8 cache

The course supplies 8 completions per prompt, already scored. Step 2 uses only this file, so no generation
is needed. A quick look at what is in it:
""")
code(r'''
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from task3_grpo.analyze_group_size import load_k8_cache
by_prompt = load_k8_cache(cfg["group_cache"])
first = next(iter(by_prompt.values()))
print(f"{len(by_prompt)} prompts x 8 completions | row keys: {sorted(first[0].keys())}")
r = np.array([[float(x["reward"]) for x in rows[:8]] for rows in by_prompt.values()])
print(f"reward range {r.min():.2f} .. {r.max():.2f} | binary reward: {set(np.unique(r)) <= {0.0, 1.0}}")
print(f"within-prompt std of 8 rewards: median {np.median(r.std(1)):.3f}  |  across-prompt std of means: {r.mean(1).std():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(9, 2.4))
axes[0].hist(r.mean(1), bins=30, color="#2a78d6"); axes[0].set_title("prompt difficulty (mean of 8 rewards)")
axes[1].hist(r.std(1), bins=30, color="#2a78d6"); axes[1].set_title("within-prompt reward std")
plt.tight_layout(); plt.show()
''')
md(r"""
Compare the two stds printed above. If prompts differ much more from each other than completions differ within a
prompt, a global baseline would mostly learn "which prompt was easy". That is exactly the planted bug.
""")

# =====================================================================================
md(r"""
## 3. The planted bug, and why the fix is right

The starter normalised every reward in the batch with **one** mean and std, ignoring `group_ids`. Below, the
corrected function and both versions on the same numbers: one easy prompt and one hard prompt, two completions each.
""")
code(r'''
import inspect, torch
from task3_grpo.grpo import group_relative_advantages
print(inspect.getsource(group_relative_advantages))

rewards = torch.tensor([9.0, 11.0, -11.0, -9.0]); gids = torch.tensor([0, 0, 1, 1])
starter = (rewards - rewards.mean()) / rewards.std(unbiased=False)
fixed = group_relative_advantages(rewards, gids)
display(pd.DataFrame({"prompt": ["easy", "easy", "hard", "hard"], "reward": rewards.tolist(),
                      "starter (global)": starter.numpy().round(3), "fixed (per group)": fixed.numpy().round(3)}))
''')
md(r"""
With the global baseline, **both** answers to the easy prompt are pushed up and **both** answers to the hard prompt
are pushed down, including the *better* answer to the hard prompt. The update learns "be the kind of response
that follows easy prompts", not "be the better answer". Per-group normalisation gives +1 to the better answer and
-1 to the worse one in each group, whatever the prompt's difficulty.
""")

# =====================================================================================
md(r"""
## 4. One group under the hood

One prompt, $K$ = 4 completions from the supplied midpoint: their rewards, group-relative advantages, lengths,
whether each was truncated (and so masked), and how much weight each token gets under the two normalisations.
""")
code(r'''
from common.data import read_jsonl, prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.models import load_policy, load_tokenizer, load_reward_model, clear_gpu
from common.logging_utils import set_seed
from common.rl_utils import disable_dropout
from task3_grpo.grpo import mask_truncated_sequences

tok = load_tokenizer(cfg["base_model"])
policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=False); disable_dropout(policy)
rm, rm_tok = load_reward_model(cfg)
prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
K, L = int(cfg["num_generations"]), int(cfg["max_completion_length"])
msgs = [prompt_messages(prompts[1])] * K
set_seed(int(cfg["seed"]))
g = batch_generate(policy, tok, msgs, int(cfg["max_prompt_length"]), L,
                   cfg["generation"]["temperature"], cfg["generation"]["top_p"], True)
with torch.no_grad():
    rew = score_reward_pairs(rm, rm_tok, msgs, g["responses"], int(cfg.get("reward_max_length", 1280))).cpu()
adv = group_relative_advantages(rew, torch.zeros(K, dtype=torch.long))
train_mask = mask_truncated_sequences(g["response_mask"], g["truncated"]).cpu()
n_trained = train_mask.sum(-1)
print("PROMPT:", msgs[0][-1]["content"][:200].replace("\n", " "), "\n")
display(pd.DataFrame({
    "length": g["response_lengths"], "truncated (masked)": g["truncated"], "reward": rew.numpy().round(3),
    "advantage": adv.numpy().round(3),
    "per-token weight, GRPO |A|/|o|": [round(abs(a) / max(int(n), 1) / K, 5) if n > 0 else 0 for a, n in zip(adv.tolist(), n_trained)],
    f"per-token weight, Dr.GRPO |A|/{L}": [round(abs(a) / L / K, 5) if n > 0 else 0 for a, n in zip(adv.tolist(), n_trained)],
    "start of response": [t[:90].replace("\n", " ") for t in g["responses"]]}))
del policy, rm; clear_gpu()
''')
md(r"""
**What to look at.** The advantages sum to zero within the group. Truncated completions show a weight of 0: they
shaped the group's mean and std but receive no gradient. And for the same $|A|$, canonical GRPO gives a shorter
completion a larger per-token weight, while Dr. GRPO's per-token weight does not depend on length.
""")

# =====================================================================================
md(r"""
## 5. Step 1: the standard 20-update continuation

`run_grpo` (in `task3_grpo/continue_train.py`): each update takes the next prompt in the fixed seeded order,
samples $K$ completions (seeded per update, so forks share them), scores them, computes group-relative advantages,
masks truncated completions, then takes one clipped policy step with the KL term. Completions are back-propagated
**one at a time**, which bounds memory and records each completion's own gradient norm (used in Step 3); the unit
tests check that the per-completion pieces add up exactly to the batch loss.

Logged per update: reward, within-group reward std, uninformative-group fraction, KL, entropy, length, truncation,
fraction of completions trained, clip fraction, gradient norm, time and peak VRAM.
""")
code(r'''
from task3_grpo.continue_train import run_grpo
RES = repo_path(cfg["results_dir"]); OUT_ROOT = repo_path(cfg["output"]).parent
std_out = repo_path(cfg["output"])
if (std_out / "adapter_config.json").exists():
    print("standard GRPO adapter found, skipping:", std_out)
else:
    run_grpo(CONFIG, run_name="standard")
clear_gpu()
''')
code(r'''
log = pd.read_json(RES / "train_standard.jsonl", lines=True)
panels = [("reward_mean", "reward-model score"), ("group_reward_std_mean", "within-group reward std"),
          ("uninformative_group_fraction", "uninformative-group fraction"), ("kl_token_mean", "KL to reference"),
          ("entropy_token_mean", "entropy"), ("response_tokens_mean", "response length"),
          ("truncated_fraction", "truncated fraction"), ("trained_completion_fraction", "completions trained"),
          ("grad_norm", "grad norm"), ("clip_fraction", "clip fraction")]
fig, axes = plt.subplots(2, 5, figsize=(14, 4.6))
for ax, (k, t) in zip(axes.flat, panels):
    ax.plot(log["update"], log[k], marker="o", ms=3, color="#2a78d6"); ax.set_title(t, fontsize=9); ax.set_xlabel("update")
plt.tight_layout(); plt.show()
meta = pd.read_json(RES / "train_standard_meta.json", typ="series")
print(f"wall-clock {meta['wall_clock_s'] / 60:.1f} min | peak VRAM {meta['peak_vram_gib']} GiB | "
      f"{meta['cumulative_generated_tokens']} generated tokens   <- the manual asks you to report these")
print(f"completions masked as truncated: {(1 - log['trained_completion_fraction']).mean():.0%} on average")
''')
md(r"""
**If many completions are masked,** most updates train on fewer than $K$ completions, and an update where all
$K$ hit the cap trains on nothing. Note that fraction in the report: it is a direct cost of the masking rule.
""")

# =====================================================================================
md(r"""
## 6. Held-out evaluation: midpoint vs standard

The same protocol as Task 2 (`common/evaluation.py`): the same 64-prompt held-out pool (prompts over the 256-token
prompt cap are skipped and listed), the same 768-token generation cap, sampling seeds, reward model and KL estimator,
so PPO and GRPO numbers are directly comparable.
""")
code(r'''
from common.evaluation import evaluate_rl_policy
from common.logging_utils import load_json
for name, adapter in [("midpoint", cfg["paths"]["grpo_midpoint_policy"]), ("standard", str(std_out))]:
    if (RES / f"eval_{name}.json").exists():
        print(f"eval_{name}.json found, skipping")
    else:
        evaluate_rl_policy(cfg, adapter, name, cfg["results_dir"])
    clear_gpu()
def ev_row(name):
    e = load_json(RES / f"eval_{name}.json")
    return {"condition": name, "n_prompts": e["n_prompts"], "reward": e["reward"]["mean"], "reward_std": e["reward"]["std"],
            "kl_token": e["kl_token_mean"], "kl_seq": e["kl_seq_mean"], "entropy": e["entropy_token_mean"],
            "len_mean": e["response_tokens"]["mean"], "len_std": e["response_tokens"]["std"], "truncation_rate": e["truncation_rate"]}
display(pd.DataFrame([ev_row("midpoint"), ev_row("standard")]).set_index("condition").round(4))
''')

# =====================================================================================
md(r"""
## 7. Step 2: group size at an equal generation budget (cache only)

No GPU. `analyze_group_size` re-slices the K=8 cache two ways, so every group size uses the same total number of
generations:

- **partition (primary):** each prompt's 8 completions are split into 8/K groups of size K. Same samples, same
  budget, only the grouping changes.
- **coverage (secondary):** a fixed budget of 2 x N_prompts completions is spent on fewer prompts as K grows (K=2:
  every prompt; K=8: a quarter of them). This is the real choice you face: more prompts, or more samples per prompt.

Metrics: informative-group rate, mean within-group reward std, the variance of the relative signal, and how far each
group's mean (the baseline) is from the prompt's 8-sample mean. Difficulty bins come from each prompt's 8-sample mean
reward (the rule is printed).
""")
code(r'''
if not (RES / "group_size_study.json").exists():
    subprocess.run([sys.executable, "-m", "task3_grpo.analyze_group_size", "--config", CONFIG], check=True, stdout=subprocess.DEVNULL)
gs = load_json(RES / "group_size_study.json")
print("difficulty rule:", gs["difficulty_rule"], "| reward is binary:", gs["reward_is_binary"])
display(pd.read_csv(RES / "group_size_study.csv").set_index("K").T.round(4))
ks = sorted(gs["partition"], key=int)
fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))
for i, (b, c) in enumerate(zip(("hard", "medium", "easy"), ("#2a78d6", "#eb6834", "#1baf7a"))):
    axes[0].plot([int(k) for k in ks], [gs["by_difficulty"][k].get(b, {}).get("informative_group_rate", np.nan) for k in ks],
                 marker="o", color=c, label=b)
    axes[1].plot([int(k) for k in ks], [gs["by_difficulty"][k].get(b, {}).get("mean_within_group_reward_std", np.nan) for k in ks],
                 marker="o", color=c, label=b)
axes[2].plot([int(k) for k in ks], [gs["partition"][k]["baseline_mse_vs_8sample_mean"] for k in ks], marker="o", color="#2a78d6")
for ax, t in zip(axes, ["informative-group rate", "within-group reward std", "baseline error vs 8-sample mean"]):
    ax.set_title(t, fontsize=9); ax.set_xticks([int(k) for k in ks]); ax.set_xlabel("group size K")
axes[0].legend(title="difficulty", frameon=False); plt.tight_layout(); plt.show()
''')
md(r"""
**Reading it.** With a continuous reward-model score, two completions almost never tie exactly, so the raw
informative rate can sit near 1 for every K. The `informative_rate_reltol` row uses a stricter threshold (5% of the
overall reward spread) that counts near-ties as uninformative, which is usually the more meaningful comparison.
Small groups also give a noisier baseline: the baseline error shrinks as K grows.
""")

# =====================================================================================
md(r"""
## 8. Step 3: canonical GRPO vs Dr. GRPO

Two forks of `fork_updates` = 8 updates from the **identical** midpoint, sharing prompts, sampling seeds, K, reward,
KL, epsilon and generation cap. Only the sequence normalisation in the loss changes. The per-completion logs record
each completion's own gradient norm, so the two can be compared at matched completion lengths.
""")
code(r'''
for lt, name in [("grpo", "norm_grpo"), ("dr_grpo", "norm_dr_grpo")]:
    out = OUT_ROOT / name
    if (out / "adapter_config.json").exists():
        print(name, "adapter found, skipping training")
    else:
        run_grpo(CONFIG, output=str(out), updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name)
    clear_gpu()
    if not (RES / f"eval_{name}.json").exists():
        evaluate_rl_policy(cfg, str(out), name, cfg["results_dir"])
    clear_gpu()
''')
code(r'''
subprocess.run([sys.executable, "-m", "task3_grpo.compare_normalization", "--config", CONFIG, "--summary-only"],
               check=True, stdout=subprocess.DEVNULL)
display(pd.read_csv(RES / "normalization_comparison.csv").set_index("loss_type").T.round(4))
nc = load_json(RES / "normalization_comparison.json")
print("length bins (tokens):", nc["length_bin_edges_tokens"])
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
for i, (lt, c) in enumerate(nc["conditions"].items()):
    tab = pd.DataFrame(c["length_conditioned"])
    col = ["#2a78d6", "#eb6834"][i]
    if not tab.empty:
        axes[0].plot(tab["length_mean"], tab["grad_norm_policy_mean"], marker="o", color=col, label=lt)
        axes[1].plot(tab["length_mean"], tab["per_token_weight_mean"], marker="o", color=col, label=lt)
    axes[2].plot(range(1, len(c["train_length_trajectory"]) + 1), c["train_length_trajectory"], marker="o", color=col, label=lt)
for ax, t in zip(axes, ["per-completion policy grad norm", "per-token weight |A|/denominator", "training response length"]):
    ax.set_title(t, fontsize=9)
axes[0].set_xlabel("completion length (tokens)"); axes[1].set_xlabel("completion length (tokens)"); axes[2].set_xlabel("update")
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
for lt, c in nc["conditions"].items():
    print(lt, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c["correlations"].items()})
''')
md(r"""
**Reading it.** Under canonical GRPO the per-token weight falls with length (dividing by $|o|$); under Dr. GRPO it is
flat. 8 updates is far too short to see a length *drift* reliably, so treat the training-length curve as a check, and
rest the comparison on the length-conditioned gradient statistics, which are measured exactly on every completion.
""")

# =====================================================================================
md(r"""
## 9. Candidates for the qualitative evidence
""")
code(r'''
def gens(name):
    p = RES / f"eval_{name}_generations.jsonl"
    return {r["prompt_id"]: r for r in read_jsonl(p)} if p.exists() else {}
mid, std_g = gens("midpoint"), gens("standard")
both = [k for k in std_g if k in mid]
for title, keys in [("largest reward GAINS over the midpoint", sorted(both, key=lambda k: std_g[k]["reward"] - mid[k]["reward"], reverse=True)[:2]),
                    ("highest-reward truncated responses (reward for an unfinished answer?)",
                     sorted([k for k in both if std_g[k]["truncated"]], key=lambda k: std_g[k]["reward"], reverse=True)[:2])]:
    print("===================", title)
    for k in keys:
        a, b = mid[k], std_g[k]
        print(f"--- {k}: reward {a['reward']:.2f} -> {b['reward']:.2f} | tokens {a['response_tokens']} -> {b['response_tokens']} | truncated {b['truncated']}")
        print("PROMPT:", b["prompt"][:250]); print("MIDPOINT:", a["response"][:350]); print("GRPO:", b["response"][:350], "\n")
''')

# =====================================================================================
md(r"""
## 10. Figures and a results zip
""")
code(r'''
FIGS = repo_path("report/figures")
subprocess.run([sys.executable, "-m", "analysis.make_figures", "--results", str(RES.parent), "--out", str(FIGS)], check=True)
OUTDIR = "/kaggle/working"
shutil.make_archive(f"{OUTDIR}/task3_results", "zip", RES.parent, RES.name)
shutil.make_archive(f"{OUTDIR}/task3_figures", "zip", FIGS.parent, FIGS.name)
print(f"wrote {OUTDIR}/task3_results.zip and {OUTDIR}/task3_figures.zip")
''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "task3_grpo.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(CELLS)} cells)")
