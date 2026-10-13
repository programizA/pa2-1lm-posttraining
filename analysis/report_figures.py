"""Figures for the PDF report, drawn at their printed size so the text stays readable.

    python -m analysis.report_figures            # writes report/figures/fig_*.pdf (+ .png previews)

Every figure is exactly the NeurIPS text width (5.5 in) and is included at width=\\linewidth, so the 8 pt
labels and 7.5 pt tick labels below are the sizes that appear on paper. Inputs are the saved files under
results/ only; no model calls. (analysis/make_figures.py draws the larger exploratory versions.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

W = 5.5  # NeurIPS \textwidth in inches
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3"
PPO_C, GRPO_C = BLUE, ORANGE

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "lines.markersize": 3,
    "axes.titlepad": 3, "axes.labelpad": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "legend.handlelength": 1.4, "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("  wrote", out / f"{name}.pdf")


def panel_label(ax, s):
    ax.set_title(s, loc="left", color=INK)


# ---------------------------------------------------------------- Task 1
def fig_dpo(R, out, nums):
    st = nums["t1_strata"]
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    names = ["preferred\nlonger", "length\nmatched", "rejected\nlonger"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(3)
    for i, (cond, lab, col) in enumerate([("standard", "standard DPO", BLUE), ("length_balanced", "length-balanced DPO", ORANGE)]):
        acc = np.array([st[cond][s]["acc"] for s in strata])
        lo = acc - np.array([st[cond][s]["ci95"][0] for s in strata])
        hi = np.array([st[cond][s]["ci95"][1] for s in strata]) - acc
        a.bar(x + (i - 0.5) * 0.38, acc, 0.36, color=col, label=lab, yerr=[lo, hi],
              error_kw={"ecolor": INK, "elinewidth": 0.7, "capsize": 1.5})
    a.axhline(0.5, color=INK2, ls=":", lw=0.8)
    a.set_xticks(x, names)
    a.set_ylim(0, 1.18)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_ylabel("held-out preference accuracy")
    a.grid(axis="x", visible=False)
    a.legend(frameon=False, loc="upper left", ncol=2, fontsize=7, columnspacing=0.8)
    panel_label(a, "(a) accuracy by length stratum")

    w = 0.2
    for j, (cond, hatch) in enumerate([("standard", None), ("length_balanced", "////")]):
        for k, (key, lab, col) in enumerate([("chosen_logratio", "preferred", GREEN), ("rejected_logratio", "rejected", GOLD)]):
            v = [st[cond][s][key] for s in strata]
            pos = x + (j * 2 + k - 1.5) * w
            b.bar(pos, v, w * 0.95, color=col, hatch=hatch, edgecolor="white", linewidth=0.3,
                  label=f"{lab}, {'standard' if j == 0 else 'length-balanced'}")
    b.axhline(0, color=INK2, lw=0.6)
    b.set_xticks(x, names)
    b.set_ylabel(r"$\log\pi_\theta-\log\pi_{\mathrm{ref}}$ (nats)")
    b.grid(axis="x", visible=False)
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=7, columnspacing=0.8)
    panel_label(b, "(b) change in sequence log-probability")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig_dpo_length")


# ---------------------------------------------------------------- Tasks 2 and 3: continuation diagnostics
def fig_continuations(R, out):
    ppo = jl(R / "task2_ppo" / "train_standard.jsonl")
    grpo = jl(R / "task3_grpo" / "train_standard.jsonl")
    up = np.arange(1, 21)
    fig, axes = plt.subplots(3, 4, figsize=(W, 3.35), sharex=True)
    ax = axes.flat

    def two(axh, kp, kg, title, log=False):
        axh.plot(up, [r[kp] for r in ppo], color=PPO_C, marker="o", label="PPO")
        axh.plot(up, [r[kg] for r in grpo], color=GRPO_C, marker="s", label="GRPO")
        if log:
            axh.set_yscale("symlog", linthresh=1e-2)
        panel_label(axh, title)

    two(ax[0], "rm_score_mean", "reward_mean", "(a) reward-model score")
    two(ax[1], "kl_token_mean", "kl_token_mean", "(b) KL / token")
    two(ax[2], "entropy_token_mean", "entropy_token_mean", "(c) entropy / token")
    two(ax[3], "response_tokens_mean", "response_tokens_mean", "(d) length (tokens)")
    two(ax[4], "policy_loss", "policy_loss", "(e) policy loss")
    two(ax[5], "policy_grad_norm", "grad_norm", "(f) policy grad norm", log=True)
    # clip fraction is 0 at every update for both; show the ratio range that explains it
    lo = [min(e["ratio_min"] for e in r["epochs"]) for r in ppo]
    hi = [max(e["ratio_max"] for e in r["epochs"]) for r in ppo]
    ax[6].fill_between(up, lo, hi, color=PPO_C, alpha=0.35, lw=0, label="PPO ratio range")
    ax[6].plot(up, np.ones(20), color=GRPO_C, lw=1.2, label="GRPO ratio (=1)")
    for e, ls in ((0.05, "--"), (0.2, ":")):
        ax[6].axhline(1 + e, color=INK2, ls=ls, lw=0.7)
        ax[6].axhline(1 - e, color=INK2, ls=ls, lw=0.7)
    ax[6].set_ylim(0.75, 1.25)
    panel_label(ax[6], "(g) policy ratio range")
    ax[7].plot(up, [r["value_loss"] for r in ppo], color=PPO_C, marker="o")
    panel_label(ax[7], "(h) PPO value loss")
    ax[8].plot(up, [r["task_reward_mean"] for r in ppo], color=PPO_C, marker="o", label="reward")
    ax[8].plot(up, [r["value_first_token_mean"] for r in ppo], color=INK2, marker="x", ls="--", label=r"critic $V(s_0)$")
    panel_label(ax[8], r"(i) $V(s_0)$ vs reward")
    ax[9].plot(up, [r["group_reward_std_mean"] for r in grpo], color=GRPO_C, marker="s")
    panel_label(ax[9], "(j) within-group std")
    ax[10].plot(up, [r["uninformative_group_fraction"] for r in grpo], color=GRPO_C, marker="s")
    ax[10].set_ylim(-0.05, 1.05)
    panel_label(ax[10], "(k) uninformative groups")
    ax[11].plot(up, [r["trained_completion_fraction"] for r in grpo], color=GRPO_C, marker="s")
    ax[11].set_ylim(-0.05, 1.05)
    panel_label(ax[11], "(l) completions trained")
    for a in axes[2]:
        a.set_xlabel("update")
        a.set_xticks([1, 5, 10, 15, 20])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0.03, 1, 1))
    save(fig, out, "fig_continuations")


# ---------------------------------------------------------------- Task 2 ablations
def fig_ppo_ablations(R, out):
    clip = json.loads((R / "task2_ppo" / "clipping_cached_batch.json").read_text())["per_epsilon"]
    kl = pd.read_csv(R / "task2_ppo" / "kl_trajectories.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    eps = ["0.05", "0.2", "0.5"]
    x = np.arange(3)
    a.bar(x - 0.19, [clip[e]["at_midpoint"]["clip_fraction"] for e in eps], 0.36, color=BLUE, label="ratio outside band")
    a.bar(x + 0.19, [clip[e]["at_midpoint"]["affected_fraction"] for e in eps], 0.36, color=ORANGE, label="gradient zeroed")
    a.set_xticks(x, [r"$\epsilon$=" + e for e in eps])
    a.set_ylabel("fraction of tokens")
    a.grid(axis="x", visible=False)
    a.set_ylim(0, 0.24)
    a.legend(frameon=False, fontsize=7, loc="upper right")
    panel_label(a, "(a) cached batch")
    for (beta, g), col, m in zip(kl.groupby("kl_beta"), [BLUE, ORANGE, GREEN], ["o", "s", "^"]):
        b.plot(g["update"], g["kl_token_mean"], color=col, marker=m, label=r"$\beta_{KL}$=" + f"{beta:g}")
        c.plot(g["update"], g["rm_score_mean"], color=col, marker=m)
    b.legend(frameon=False, fontsize=7, loc="lower right")
    panel_label(b, "(b) KL / token")
    panel_label(c, "(c) reward-model score")
    for axh in (b, c):
        axh.set_xlabel("update")
        axh.set_xticks(range(1, 9))
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_ppo_ablations")


# ---------------------------------------------------------------- Task 3 ablations
def fig_grpo_ablations(R, out):
    g = json.loads((R / "task3_grpo" / "group_size_study.json").read_text())
    n = json.loads((R / "task3_grpo" / "normalization_comparison.json").read_text())["conditions"]
    ks = ["2", "4", "8"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.8))
    for d, col, m in (("hard", BLUE, "o"), ("medium", ORANGE, "s"), ("easy", GREEN, "^")):
        a.plot([2, 4, 8], [g["by_difficulty"][k][d]["informative_group_rate"] for k in ks], color=col, marker=m, label=d)
        b.plot([2, 4, 8], [g["by_difficulty"][k][d]["mean_within_group_reward_std"] for k in ks], color=col, marker=m, label=d)
    a.plot([2, 4, 8], [g["partition"][k]["informative_group_rate"] for k in ks], color=INK2, ls="--", label="all")
    b.plot([2, 4, 8], [g["partition"][k]["mean_within_group_reward_std"] for k in ks], color=INK2, ls="--", label="all")
    for axh in (a, b):
        axh.set_xticks([2, 4, 8])
        axh.set_xlabel("group size $K$")
    a.set_ylim(0.8, 1.01)
    a.legend(frameon=False, fontsize=7, loc="lower right")
    panel_label(a, "(a) informative-group rate")
    panel_label(b, "(b) mean within-group std")
    for cond, lab, col, m in (("grpo", r"canonical $1/|o|$", BLUE, "o"), ("dr_grpo", r"Dr. GRPO $1/L$", ORANGE, "s")):
        lc = n[cond]["length_conditioned"]
        c.plot([r["length_mean"] for r in lc], [r["grad_norm_policy_mean"] for r in lc], color=col, marker=m, label=lab)
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlabel("completion length (tokens)")
    c.legend(frameon=False, fontsize=7, loc="center left")
    panel_label(c, "(c) per-completion grad norm")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_grpo_ablations")


# ---------------------------------------------------------------- Task 4
SHORT = {"definitions": "definitions", "figurative_language": "figurative language", "historical_events": "historical events",
         "homonyms": "homonyms", "nons_group_real_discr": "nonsense group, real discr.", "privacy_fictional": "privacy (fictional)",
         "privacy_public": "privacy (public figure)", "real_group_nons_discr": "real group, nonsense discr.",
         "safe_contexts": "safe contexts", "safe_targets": "safe targets"}


def fig_safety(R, out):
    t4 = R / "task4_safety"
    cat = pd.read_csv(t4 / "safety_by_category.csv")
    rule = pd.read_csv(t4 / "refusal_check_labels.csv")
    pols = ["sft", "dpo", "ppo", "grpo"]
    judge = cat.pivot_table(index=["class", "type"], columns="policy", values="frac_JUSTIFIED_REFUSAL")[pols]
    refused = rule.groupby(["cls", "type", "policy"])["refused"].mean().unstack()[pols]
    refused.index = refused.index.set_names(["class", "type"])
    refused = refused.reindex(judge.index)

    def label(cls, t):
        if cls == "SAFE":
            return SHORT.get(t, t)
        return "contrast: " + SHORT.get(t.replace("contrast_", ""), t.replace("contrast_", "").replace("_", " "))

    rows = [label(c, t) for c, t in judge.index]
    nsafe = int((judge.index.get_level_values(0) == "SAFE").sum())
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.25), sharey=True)
    for axh, M, title in ((axes[0], judge, "AI judge: labelled JUSTIFIED_REFUSAL"), (axes[1], refused, "refusal rule: response refused")):
        im = axh.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.iat[i, j]
                axh.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.6 else INK)
        axh.set_xticks(range(4), [p.upper() for p in pols])
        axh.xaxis.tick_top()
        axh.tick_params(axis="x", length=0)
        axh.axhline(nsafe - 0.5, color=INK, lw=1.0)
        axh.grid(False)
        axh.set_title(title, color=INK, pad=14)
    axes[0].set_yticks(range(len(rows)), rows)
    axes[0].text(-0.5, nsafe / 2 - 0.5, "", transform=axes[0].transData)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_safety_categories")


# ---------------------------------------------------------------- Task 5
def fig_feedback(R, out):
    t5 = R / "task5_feedback"
    fb = pd.read_csv(t5 / "task5_format_breakdown.csv")
    d = json.loads((t5 / "diagnostic_summary.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw={"width_ratios": [1.1, 1.3]})
    xs, labels, x = [], [], 0.0
    for ds, grp in fb.groupby("dataset", sort=False):
        for _, r in grp.iterrows():
            xs.append(x)
            labels.append(r["policy"].upper())
            x += 1
        a.text(np.mean(xs[-3:]), 0.97, ds, ha="center", va="top", fontsize=7.5, color=INK)
        x += 0.5
    bottom = np.zeros(len(fb))
    for col, lab, c in (("correct_and_formatted", "correct, has ####", BLUE),
                        ("correct_but_unformatted", "correct, no ####", GOLD), ("other", "wrong", GREY)):
        v = fb[col].to_numpy()
        a.bar(xs, v, 0.8, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=0.5)
        bottom += v
    a.set_xticks(xs, labels, fontsize=6.5, rotation=30)
    a.set_ylim(0, 1)
    a.set_ylabel("fraction of problems")
    a.grid(axis="x", visible=False)
    a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, handlelength=1.0, columnspacing=0.7)
    panel_label(a, "(a) what the exact verifier scores")

    cats = ["corrupt_reasoning_correct_final", "persuasive_filler_correct", "good_reasoning_wrong_final", "gold_distractor_wrong_final"]
    names = ["corrupt reasoning,\ncorrect final", "persuasive filler,\ncorrect final", "good reasoning,\nwrong final", "gold distractor,\nwrong final"]
    y = np.arange(len(cats))
    for off, mech, hatch in ((-0.2, "verifier", None), (0.2, "judge", "////")):
        left = np.zeros(len(cats))
        for key, col in (("better_rate", BLUE), ("tie_rate", GREY), ("wrong_rate", ORANGE)):
            v = np.array([d["by_category"][c][mech][key] for c in cats])
            b.barh(y + off, v, 0.36, left=left, color=col, hatch=hatch, edgecolor="white", linewidth=0.3)
            left += v
        for yi in y:
            b.text(1.02, yi + off, mech, va="center", fontsize=6.5, color=INK2)
    b.set_yticks(y, names, fontsize=7)
    b.invert_yaxis()
    b.set_xlim(0, 1)
    b.set_xlabel("fraction of 20 controlled pairs")
    b.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    b.legend([Patch(color=BLUE), Patch(color=GREY), Patch(color=ORANGE)], ["prefers clean", "tie", "prefers perturbed"],
             frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, handlelength=1.0, columnspacing=0.7)
    panel_label(b, "(b) controlled pairs: verifier vs judge")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_feedback")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    nums = json.loads((R / "report_numbers.json").read_text())
    fig_dpo(R, out, nums)
    fig_continuations(R, out)
    fig_ppo_ablations(R, out)
    fig_grpo_ablations(R, out)
    fig_safety(R, out)
    fig_feedback(R, out)


if __name__ == "__main__":
    main()
