"""Build notebooks/task2_ppo.ipynb, the guided Task 2 walkthrough. Run: python notebooks/build_task2_notebook.py"""
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
# Task 2: Proximal Policy Optimization (PPO)

Same structure as the Task 1 notebook: each cell calls functions from the repo (`task2_ppo/`, `common/`),
shows the important source, and inspects what came out. Nothing here is experiment logic of its own.

**Kaggle settings:** Accelerator **GPU T4 x2** (the code uses one), Internet **On**.

**Every section is idempotent.** Training skips if the saved adapter exists; evaluation skips if its result
file exists. After a session drop, *Run All* again.

| Section | What happens | GPU time (rough) |
|---|---|---|
| 0 | Setup: clone, install, download the PPO assets, unit tests | minutes |
| 1 | PPO from the ground up (reading, no code) | none |
| 2 | The four models and the prompt pool | ~1 min |
| 3 | The planted bug, and why the fix is correct | none |
| 4 | One rollout under the hood: every quantity, token by token | ~1 min |
| 5 | **Step 1**: the standard 20-update continuation (VRAM + time recorded) | the long one |
| 6 | Evaluate the midpoint and the standard run on held-out prompts | ~10 min each |
| 7 | **Step 2a**: clipping on the fixed cached batch | minutes |
| 8 | **Step 2b**: matched short forks over epsilon | 3 forks + evals |
| 9 | **Step 3**: matched short forks over the KL coefficient | 2 new forks + evals |
| 10 | Candidates for the qualitative evidence | none |
| 11 | Figures and a zip of all results | none |
""")

# =====================================================================================
md(r"""
## 0. Setup
""")
code(r'''
REPO_URL = "https://github.com/programizA/pa2-1lm-posttraining.git"
WORK = "/kaggle/working/pa2"
CONFIG = "configs/ppo.yaml"

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
import transformers, peft, huggingface_hub, torch
print(transformers.__version__, peft.__version__, huggingface_hub.__version__, torch.__version__, torch.cuda.is_available())
# Expect 4.57.1 0.17.1 0.36.2 ... True. The pip "dependency conflict" warnings about gradio/diffusers are harmless.
''')
code(r'''
# Download only what Task 2 needs from the pinned course asset release.
from huggingface_hub import snapshot_download
from scripts.download_assets import HF_ASSET_REPO, HF_ASSET_REVISION
from common.data import load_yaml, repo_path

cfg = load_yaml(CONFIG)
needed = ["rl_prompt_train", "rl_prompt_eval", "ppo_midpoint_policy", "ppo_midpoint_value"]
if not (all(repo_path(cfg["paths"][k]).exists() for k in needed) and repo_path(cfg["cached_rollouts"]).exists()):
    snapshot_download(repo_id=HF_ASSET_REPO, repo_type="dataset", revision=HF_ASSET_REVISION, local_dir=WORK,
                      allow_patterns=["data/rl_prompt_pool_*", "cached/ppo_rollout.pt", "checkpoints/ppo_midpoint_*"])
for k in needed:
    p = repo_path(cfg["paths"][k]); print(f"{k:22s} {'OK' if p.exists() else 'MISSING'}  {p}")
print(f"{'ppo_rollout_cache':22s} {'OK' if repo_path(cfg['cached_rollouts']).exists() else 'MISSING'}")
subprocess.run([sys.executable, "-m", "tests.test_objectives"], check=True, stdout=subprocess.DEVNULL)
print("12 unit tests passed")
print({k: cfg[k] for k in ["updates", "fork_updates", "prompts_per_update", "ppo_epochs", "policy_learning_rate",
                           "clip_epsilon", "clip_values", "kl_beta", "kl_values", "gamma", "gae_lambda", "value_coef",
                           "missing_eos_penalty", "max_response_length", "eval_max_response_length", "eval_prompts"]})
''')

# =====================================================================================
md(r"""
## 1. PPO from the ground up

**DPO learned from a fixed pile of comparisons. PPO learns from its own attempts.** Each update, the model
(the *policy*) writes a response, a judge scores it, and the policy is nudged towards whatever scored well.
Because the data come from the current model, this is *online* or *on-policy* learning.

**Four models take part, and only two of them learn.**

| Model | Role | Trains? |
|---|---|---|
| Policy (1.5B + LoRA) | the student writing answers | yes |
| Reward model (1.5B) | the examiner: one score for a whole finished answer | no |
| Reference (the policy with LoRA off) | the "photograph" of where the student started | no |
| Critic / value model (0.5B) | the student's own running guess of the final score | yes |

**A response is a sequence of decisions.** Each token is one action. The reward model only scores the end,
so the hard question is *credit assignment*: which of the 300 tokens deserve the credit or blame?

**The critic answers "how well am I doing so far?"** At each token $t$ it predicts $V(s_t)$, the reward it
expects the finished response to get from here. Think of a sports commentator who updates the predicted
final score after every play.

**The advantage is "better or worse than expected".** A token is credited by how much the expected outcome
*improved* after choosing it:
$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. If the commentator's forecast jumps after a play, the play was
good. **GAE** blends these one-step surprises over the following tokens,
$A_t = \delta_t + \gamma\lambda\,\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots$, with $\lambda = 0.95$
trading a little bias for much less noise. The **return** $A_t + V(s_t)$ is what the critic is trained to predict.

**The KL leash.** Left alone, the policy can find odd text the reward model happens to love. So each token
also pays a small toll for drifting from the reference: $r_t = -\beta_{KL}\,[\log\pi(a_t) - \log\pi_{\text{ref}}(a_t)]$,
and the reward model's score is added at the final token. Larger $\beta_{KL}$ means a shorter leash.

**The ratio and the speed limit.** After collecting a response we take a few gradient steps on it. Each step
compares the current policy with the one that wrote the response:
$\rho_t = \pi_\theta(a_t)/\pi_{\text{old}}(a_t)$. The surrogate objective is $\rho_t A_t$: make good tokens
more likely, bad ones less likely. But these samples were only valid for the *old* policy, so PPO caps how
far one batch may push: it uses $\min(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t)$. Once a
token's probability has already moved more than $\epsilon$ in the direction its advantage wants, that token
stops contributing gradient. Like a speed limit that only applies when you are already speeding in the
direction you are being pushed: slowing down or reversing is always allowed.

**Two measures of clipping, kept separate.** The *clip fraction* counts tokens whose ratio is outside
$[1-\epsilon, 1+\epsilon]$. The *affected fraction* counts tokens whose gradient was actually switched off,
which is only those outside the interval *in the direction their advantage pushes*. The second is always at
most the first.
""")

# =====================================================================================
md(r"""
## 2. The models and the prompt pool
""")
code(r'''
from common.data import read_jsonl, prompt_messages
prompts = read_jsonl(cfg["paths"]["rl_prompt_train"]); heldout = read_jsonl(cfg["paths"]["rl_prompt_eval"])
print(f"train prompt pool: {len(prompts)} | held-out pool: {len(heldout)} (the evaluation uses the first {cfg['eval_prompts']})")
print("keys:", sorted(prompts[0]))
for r in prompts[:3]:
    print("-", prompt_messages(r)[-1]["content"][:200].replace("\n", " "))
''')
code(r'''
from common.models import load_policy, load_tokenizer, load_reward_model, load_value_model, clear_gpu, count_parameters
from common.generation import score_reward_pairs
tok = load_tokenizer(cfg["base_model"])
rm, rm_tok = load_reward_model(cfg)
# The reward model gives ONE number per (prompt, response). Same prompt, two responses:
p = prompt_messages(prompts[0])
pair = ["I don't know.", "Here is a clear, step-by-step answer to your question, with the reasoning shown."]
print([round(float(s), 3) for s in score_reward_pairs(rm, rm_tok, [p, p], pair)])
print("(Only the difference between scores matters; the scale is arbitrary.)")
critic = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
print("critic parameters: %.0fM" % (count_parameters(critic)[0] / 1e6))
del rm, critic; clear_gpu()
''')

# =====================================================================================
md(r"""
## 3. The planted bug, and why the fix is right

The starter used `torch.maximum` where the manual's surrogate uses `min`. The cell below evaluates four
tokens, one in each corner of the clipping picture, under both versions, and shows which tokens still get
a gradient.
""")
code(r'''
import inspect, torch, pandas as pd
from task2_ppo.ppo import ppo_policy_loss, clip_diagnostics
print(inspect.getsource(ppo_policy_loss))

eps = 0.2
ratio = torch.tensor([[1.5, 1.5, 0.5, 0.5]]); adv = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
rows = []
for name, op in [("fixed (min)", torch.minimum), ("starter (max)", torch.maximum)]:
    r = ratio.clone().requires_grad_(True)
    obj = op(r * adv, r.clamp(1 - eps, 1 + eps) * adv)
    (-obj.mean()).backward()
    rows.append([name] + [f"{g:+.2f}" for g in r.grad[0].tolist()])
cols = ["version", "rho=1.5, A>0", "rho=1.5, A<0", "rho=0.5, A<0", "rho=0.5, A>0"]
display(pd.DataFrame(rows, columns=cols).set_index("version"))
print(clip_diagnostics(ratio, adv, torch.ones_like(adv), eps))
''')
md(r"""
Read the table as "gradient of the loss for each token" (0 means the token is switched off).

- **Fixed:** the two tokens that already moved too far *in the direction their advantage wants* (columns 1
  and 3) get zero gradient; the two moving the other way still get pulled back. Clip fraction 1.0 (all four
  are outside the interval), affected fraction 0.5 (only two were actually switched off).
- **Starter:** exactly the wrong tokens are switched off. The token that is already far too likely and is
  being pushed to become *more* likely (column 1) keeps its full gradient, so there is no speed limit at all.
""")

# =====================================================================================
md(r"""
## 4. One rollout under the hood

One prompt, one response from the supplied midpoint policy, and every quantity from Section 1 laid out per
token: the log-probabilities, the KL toll, the reward (zero until the final token), the critic's
forecast, the advantage, and the return. This is exactly the computation inside each training update.
""")
code(r'''
from common.generation import batch_generate
from common.models import reference_mode
from common.rl_utils import token_logprobs, disable_dropout
from common.logging_utils import set_seed
from task2_ppo.ppo import shaped_rewards, compute_gae, normalize_advantages
from task2_ppo.continue_train import critic_values, explained_variance

policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False); disable_dropout(policy)
critic = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
rm, rm_tok = load_reward_model(cfg)

msgs = [prompt_messages(prompts[0])]
set_seed(int(cfg["seed"]))
g = batch_generate(policy, tok, msgs, int(cfg["max_prompt_length"]), int(cfg["max_response_length"]),
                   cfg["generation"]["temperature"], cfg["generation"]["top_p"], True)
seq, attn, pw, resp, mask = g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"], g["response_mask"]
with torch.no_grad():
    old_lp, ent = token_logprobs(policy, seq, attn, pw, resp, with_entropy=True)
    with reference_mode(policy):
        ref_lp, _ = token_logprobs(policy, seq, attn, pw, resp)
    score = score_reward_pairs(rm, rm_tok, msgs, g["responses"], int(cfg["reward_max_length"])).to(seq.device)
    eos_pen = cfg["missing_eos_penalty"] * (0.0 if g["terminated_with_eos"][0] else 1.0)
    values = critic_values(critic, seq.to(critic.device), attn.to(critic.device), pw, resp.shape[1]).to(seq.device) * mask
    rewards = shaped_rewards(score - eos_pen, old_lp, ref_lp, mask, cfg["kl_beta"])
    adv, ret = compute_gae(rewards, values, mask, cfg["gamma"], cfg["gae_lambda"])

print("PROMPT:", msgs[0][-1]["content"][:200])
print(f"\nRESPONSE ({g['response_lengths'][0]} tokens, ended with EOS: {g['terminated_with_eos'][0]}):\n", g["responses"][0][:600])
print(f"\nreward-model score {score.item():.3f}, missing-EOS penalty {eos_pen}")
n = int(mask.sum())
table = pd.DataFrame({
    "token": [tok.decode([t]) for t in resp[0, :n].tolist()],
    "log pi(a)": old_lp[0, :n].cpu().numpy(), "log ref(a)": ref_lp[0, :n].cpu().numpy(),
    "reward r_t": rewards[0, :n].cpu().numpy(), "value V(s_t)": values[0, :n].cpu().numpy(),
    "advantage A_t": adv[0, :n].cpu().numpy(), "return": ret[0, :n].cpu().numpy()})
display(pd.concat([table.head(8), table.tail(4)]).round(3))
''')
md(r"""
**What to check in the table.**
- `reward r_t` is a small KL toll on every token, plus the reward-model score on the **last** token only.
- `value V(s_t)` is the critic's running forecast. The manual warns this critic is imperfectly calibrated;
  if its forecasts are far from the final score, every advantage inherits that error. That is why critic
  behaviour is an analysis variable in this task.
- The advantage is large where the forecast changes a lot and where the final reward lands.

And the property every PPO update relies on: on the first epoch, the policy being trained *is* the policy
that wrote the response, so every ratio is exactly 1 and nothing is clipped. Dropout is switched off in
training precisely so this holds.
""")
code(r'''
with torch.no_grad():
    again, _ = token_logprobs(policy, seq, attn, pw, resp)
ratio = torch.exp(again - old_lp)
print(f"max |ratio - 1| on epoch 0: {(ratio - 1).abs().max().item():.2e}   (expect ~0)")
print(f"critic explained variance on this one response: {explained_variance(values, ret, mask):.3f} (1 = perfect, 0 = no better than a constant)")
del policy, critic, rm; clear_gpu()
''')

# =====================================================================================
md(r"""
## 5. Step 1: the standard 20-update continuation

`run_ppo` (in `task2_ppo/continue_train.py`) starts from the supplied midpoint policy **and** critic. Each update:
1. takes the next prompt from a fixed, seeded order and samples one response (seeded per update, so every fork
   later sees the same prompt and the same random numbers at update *u*);
2. computes old and reference log-probs, the reward-model score, the critic's values, the KL-shaped rewards, GAE
   advantages and returns, and whitens the advantages;
3. runs `ppo_epochs` = 2 passes: one clipped policy step and one critic (value-loss) step per pass.

Every update logs reward, KL, entropy, length, policy and value loss, clip and affected fractions, gradient
norms, explained variance, approximate KL between old and new policy, elapsed time and peak VRAM, to
`results/task2_ppo/train_standard.jsonl`.
""")
code(r'''
from task2_ppo.continue_train import run_ppo
RES = repo_path(cfg["results_dir"]); OUT_ROOT = repo_path(cfg["output"]).parent
std_out = repo_path(cfg["output"])
if (std_out / "adapter_config.json").exists():
    print("standard PPO adapter found, skipping:", std_out)
else:
    run_ppo(CONFIG, run_name="standard")
clear_gpu()
''')
code(r'''
import matplotlib.pyplot as plt, numpy as np
log = pd.read_json(RES / "train_standard.jsonl", lines=True)
panels = [("rm_score_mean", "reward-model score"), ("kl_token_mean", "KL to reference"), ("policy_loss", "policy loss"),
          ("value_loss", "value loss"), ("entropy_token_mean", "entropy"), ("clip_fraction", "clip fraction"),
          ("policy_grad_norm", "policy grad norm"), ("response_tokens_mean", "response length"),
          ("explained_variance", "critic explained variance"), ("approx_kl_old_new", "approx KL(old||new)")]
fig, axes = plt.subplots(2, 5, figsize=(14, 4.6))
for ax, (k, t) in zip(axes.flat, panels):
    ax.plot(log["update"], log[k], marker="o", ms=3, color="#2a78d6"); ax.set_title(t, fontsize=9); ax.set_xlabel("update")
plt.tight_layout(); plt.show()
meta = pd.read_json(RES / "train_standard_meta.json", typ="series")
print(f"wall-clock {meta['wall_clock_s'] / 60:.1f} min | peak VRAM {meta['peak_vram_gib']} GiB | "
      f"{meta['cumulative_generated_tokens']} generated tokens   <- the manual asks you to report these")
''')
md(r"""
**Reading these.** With one prompt per update, every curve is noisy; look at trends, not single points.
Reward and KL should drift upward together. The critic's explained variance tells you whether its forecasts
help: near or below zero means the advantages are mostly noise from the critic, not signal.
""")

# =====================================================================================
md(r"""
## 6. Held-out evaluation: midpoint vs standard

Both evaluated by the same protocol (`common/evaluation.py`): the first 64 held-out prompts, the larger
generation cap from the config (768 tokens, because the midpoint is known to ramble past 512), fixed sampling
seeds, the same reward model, and the sampled KL estimator against the reference.
""")
code(r'''
from common.evaluation import evaluate_rl_policy
from common.logging_utils import load_json
for name, adapter in [("midpoint", cfg["paths"]["ppo_midpoint_policy"]), ("standard", str(std_out))]:
    if (RES / f"eval_{name}.json").exists():
        print(f"eval_{name}.json found, skipping")
    else:
        evaluate_rl_policy(cfg, adapter, name, cfg["results_dir"])
    clear_gpu()
def ev_row(name):
    e = load_json(RES / f"eval_{name}.json")
    return {"condition": name, "reward": e["reward"]["mean"], "reward_std": e["reward"]["std"], "kl_token": e["kl_token_mean"],
            "kl_seq": e["kl_seq_mean"], "entropy": e["entropy_token_mean"], "len_mean": e["response_tokens"]["mean"],
            "len_std": e["response_tokens"]["std"], "truncation_rate": e["truncation_rate"]}
display(pd.DataFrame([ev_row("midpoint"), ev_row("standard")]).set_index("condition").round(4))
''')

# =====================================================================================
md(r"""
## 7. Step 2a: clipping on the fixed cached batch

No new sampling here. The course supplies a cached batch of rollouts with the log-probs of the policy that
generated them. `cached_batch_study` rebuilds it, computes rewards, GAE advantages and returns, and the
ratio of the supplied midpoint against those cached log-probs. For each epsilon it reports the clipped
surrogate, the clip fraction and the affected fraction under the same mask, then takes `ppo_epochs` real
gradient steps on that batch from identical weights, to show how quickly tokens hit the boundary.
""")
code(r'''
from types import SimpleNamespace
from task2_ppo.analyze_clipping import cached_batch_study
from common.logging_utils import save_json
cache_json = RES / "clipping_cached_batch.json"
if cache_json.exists():
    print("cached-batch study found, skipping")
else:
    save_json(cache_json, cached_batch_study(cfg, SimpleNamespace(sim_steps=int(cfg["ppo_epochs"]), micro_batch=4)))
clear_gpu()
c = load_json(cache_json)
print({k: c[k] for k in ["n_rollouts", "valid_tokens", "reward_source", "value_source", "rebuild_notes"]})
print("ratio of midpoint vs cached old policy:", {k: round(v, 4) if isinstance(v, float) else v
                                               for k, v in c["midpoint_vs_cached_old_ratio"].items()})
rows = []
for e, d in c["per_epsilon"].items():
    rows.append({"epsilon": float(e), **{k: d["at_midpoint"][k] for k in ["clip_fraction", "affected_fraction", "clipped_surrogate", "unclipped_surrogate"]},
                 "affected_after_steps": d["simulated_steps"][-1]["affected_fraction"]})
display(pd.DataFrame(rows).set_index("epsilon").round(4))
# How the affected fraction evolves over the simulated steps (0 = supplied midpoint, before any update).
display(pd.DataFrame({f"eps={e}": {s["updates_applied"]: s["affected_fraction"] for s in d["simulated_steps"]}
                      for e, d in c["per_epsilon"].items()}).rename_axis("updates applied").round(4))
''')
md(r"""
With the small policy learning rate (3e-6), two steps may barely move the ratios; if the affected fraction
hardly changes across updates, that is itself the finding (clipping only bites once ratios have drifted, which
takes many updates or a larger step size). The first row compares the midpoint with the policy that generated
the cache, which may differ from it.

**Check:** `rebuild_notes` should show `length_mismatch: 0` (each cached response re-tokenises to exactly as
many tokens as there are cached log-probs). If it isn't 0, tell me before trusting this table.
""")

# =====================================================================================
md(r"""
## 8. Step 2b: matched short forks over epsilon

Three forks of `fork_updates` = 8 updates, each from the **identical** supplied midpoint (policy and critic),
with identical prompts, sampling seeds, reward and KL settings. Only epsilon changes.
""")
code(r'''
def fork(name, **kw):
    out = OUT_ROOT / name
    if (out / "adapter_config.json").exists():
        print(name, "adapter found, skipping training")
    else:
        run_ppo(CONFIG, output=str(out), updates=int(cfg["fork_updates"]), run_name=name, **kw)
    clear_gpu()
    if not (RES / f"eval_{name}.json").exists():
        evaluate_rl_policy(cfg, str(out), name, cfg["results_dir"])
    clear_gpu()

for e in cfg["clip_values"]:
    fork(f"clip_{float(e):g}", clip_epsilon=float(e))
''')
code(r'''
subprocess.run([sys.executable, "-m", "task2_ppo.analyze_clipping", "--config", CONFIG, "--part", "summary"],
               check=True, stdout=subprocess.DEVNULL)
clip_tab = pd.read_csv(RES / "clipping_summary.csv").set_index("epsilon")
display(clip_tab.T.round(4))
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
for i, e in enumerate(cfg["clip_values"]):
    lg = pd.read_json(RES / f"train_clip_{float(e):g}.jsonl", lines=True)
    for ax, k in zip(axes, ["approx_kl_old_new", "clip_fraction", "policy_grad_norm", "rm_score_mean"]):
        ax.plot(lg["update"], lg[k], marker="o", ms=3, color=colors[i], label=f"eps={e:g}"); ax.set_title(k, fontsize=9)
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
''')
md(r"""
The stability statistic is the approximate KL between the policy before and after each update epoch
(`approx_kl_mean`, `approx_kl_max`, and `unstable_epoch_fraction` = share of epochs above 0.02), alongside the
maximum gradient norm. A looser epsilon should let the policy move further per batch.
""")

# =====================================================================================
md(r"""
## 9. Step 3: matched short forks over the KL coefficient

Forks for beta_KL in {0, 0.10, 0.20}, same 8-update budget, same everything else. One shortcut: the
beta_KL = 0.10 fork has exactly the same settings as the epsilon = 0.20 fork from Section 8 (both use the
default eps 0.20 and beta_KL 0.10, the same prompts, seeds and budget). It is the same experimental
condition, so its files are reused rather than trained twice, which saves about half an hour of GPU.
""")
code(r'''
default_eps, default_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
for b in cfg["kl_values"]:
    name = f"kl_{float(b):g}"
    if float(b) == default_kl and default_eps in [float(e) for e in cfg["clip_values"]]:
        src = f"clip_{default_eps:g}"
        if not (OUT_ROOT / name / "adapter_config.json").exists():
            shutil.copytree(OUT_ROOT / src, OUT_ROOT / name, dirs_exist_ok=True)
            for f in (f"train_{src}.jsonl", f"train_{src}_meta.json", f"eval_{src}.json", f"eval_{src}_generations.jsonl"):
                shutil.copy(RES / f, RES / f.replace(src, name))
        print(f"{name}: identical condition to {src}, files reused")
    else:
        fork(name, kl_beta=float(b))
''')
code(r'''
subprocess.run([sys.executable, "-m", "task2_ppo.ablate_kl", "--config", CONFIG, "--summary-only"],
               check=True, stdout=subprocess.DEVNULL)
display(pd.read_csv(RES / "kl_summary.csv").set_index("kl_beta").T.round(4))
traj = pd.read_csv(RES / "kl_trajectories.csv")
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
for i, (b, d) in enumerate(traj.groupby("kl_beta")):
    for ax, k in zip(axes, ["rm_score_mean", "kl_token_mean", "entropy_token_mean", "response_tokens_mean"]):
        ax.plot(d["update"], d[k], marker="o", ms=3, color=colors[i], label=f"beta_KL={b:g}"); ax.set_title(k, fontsize=9)
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
print(json.dumps(load_json(RES / "kl_first_departure.json")["first_update"], indent=1))
''')
md(r"""
**Research question 3** asks which observable moves first as KL pressure drops. `kl_first_departure.json`
gives, for each fork, the first update where a metric differs from the beta_KL = 0.20 fork by more than one
standard deviation of that metric over the standard run. With 8 noisy single-prompt updates treat it as a
hint to check against the curves, not as a precise timing.
""")

# =====================================================================================
md(r"""
## 10. Candidates for the qualitative evidence

The manual wants at least one case where a high learned reward does **not** match better quality. Below are
the held-out responses where the reward model's score rose most from the midpoint to each trained model,
with lengths, so you can judge whether the gain is real.
""")
code(r'''
def gens(name):
    p = RES / f"eval_{name}_generations.jsonl"
    return {r["prompt_id"]: r for r in read_jsonl(p)} if p.exists() else {}
mid = gens("midpoint")
for name in ("standard", "kl_0"):
    g = gens(name)
    both = [k for k in g if k in mid]
    if not both:
        continue
    top = sorted(both, key=lambda k: g[k]["reward"] - mid[k]["reward"], reverse=True)[:2]
    print(f"=================== {name}: largest reward gains over the midpoint")
    for k in top:
        a, b = mid[k], g[k]
        print(f"--- {k}: reward {a['reward']:.2f} -> {b['reward']:.2f} | tokens {a['response_tokens']} -> {b['response_tokens']} "
              f"| truncated {a['truncated']} -> {b['truncated']}")
        print("PROMPT:", b["prompt"][:250]); print("MIDPOINT:", a["response"][:400]); print(f"{name.upper()}:", b["response"][:400], "\n")
''')

# =====================================================================================
md(r"""
## 11. Figures and a results zip
""")
code(r'''
FIGS = repo_path("report/figures")
subprocess.run([sys.executable, "-m", "analysis.make_figures", "--results", str(RES.parent), "--out", str(FIGS)], check=True)
OUTDIR = "/kaggle/working"
shutil.make_archive(f"{OUTDIR}/task2_results", "zip", RES.parent, RES.name)
shutil.make_archive(f"{OUTDIR}/task2_figures", "zip", FIGS.parent, FIGS.name)
print(f"wrote {OUTDIR}/task2_results.zip and {OUTDIR}/task2_figures.zip")
''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "task2_ppo.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(CELLS)} cells)")
