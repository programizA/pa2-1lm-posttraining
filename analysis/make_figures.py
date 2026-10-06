"""Render report figures from the saved machine-readable results (no model calls).

    python -m analysis.make_figures --results results --out report/figures

Every figure is skipped quietly if its input files are missing, so this can be re-run as results
arrive. Output: one PDF (vector, for the LaTeX report) and one PNG per figure.

Palette: fixed categorical order (validated for colour-vision deficiency). Every multi-series plot
also uses distinct marker shapes and a legend so identity never depends on colour alone.
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
from matplotlib.ticker import MaxNLocator  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("  wrote", out / f"{name}.pdf")


def trajectory_grid(log, panels, title, out, name):
    """Small multiples of one run (single series -> no legend; panel titles name each metric)."""
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 1.6 * rows + 0.3), squeeze=False)
    x = [r["update"] for r in log]
    for ax, (key, label) in zip(axes.flat, panels):
        y = [r.get(key) for r in log]
        ax.plot(x, y, color=SERIES[0], marker="o")
        ax.set_title(label, color=INK)
        ax.set_xlabel("update")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, out, name)


def multi_series(series, panels, title, out, name, legend_title):
    """series: list of (label, log). Same metric panels, one line per condition."""
    fig, axes = plt.subplots(1, len(panels), figsize=(1.8 * len(panels) + 0.9, 1.9), squeeze=False)
    for ax, (key, label) in zip(axes.flat, panels):
        for i, (lab, log) in enumerate(series):
            x = [r["update"] for r in log]
            y = [r.get(key) for r in log]
            ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], label=lab)
        ax.set_title(label, color=INK)
        ax.set_xlabel("update")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title, color=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, title=legend_title, frameon=False, loc="center left", bbox_to_anchor=(0.88, 0.5))
    save(fig, out, name)


def grouped_bars(categories, groups, values, ylabel, title, out, name, legend_title=None, ylim=None):
    """values[g][c]; groups are series (<=4), categories on the x axis. 2px-style gaps between bars."""
    x = np.arange(len(categories))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(3.2, 0.9 * len(categories) + 1.5), 2.2))
    for i, g in enumerate(groups):
        v = [values[g].get(c, np.nan) for c in categories]
        bars = ax.bar(x + (i - (len(groups) - 1) / 2) * w, v, w * 0.92, color=SERIES[i], label=g, edgecolor="white", linewidth=0.8)
        for b, val in zip(bars, v):
            if val == val and len(groups) * len(categories) <= 8:  # selective labels only on sparse charts
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, [c.replace(": ", ":\n") for c in categories])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(title=legend_title, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title, color=INK)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)

    # ---------------- Task 1
    t1 = R / "task1_dpo"
    if (t1 / "dpo_summary.csv").exists():
        df = pd.read_csv(t1 / "dpo_summary.csv")
        fk = df[df.condition.str.startswith("beta_")].sort_values("beta")
        if len(fk):
            panels = [("heldout_pref_acc", "held-out pref. accuracy"), ("kl_token_mean", "KL to reference (token mean)"),
                      ("reward_mean", "reward-model score"), ("len_tokens_mean", "response length (tokens)")]
            fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
            for ax, (k, lab) in zip(axes, panels):
                ax.plot(fk["beta"], fk[k], color=SERIES[0], marker="o")
                if k == "len_tokens_mean" and "len_tokens_std" in fk:
                    ax.errorbar(fk["beta"], fk[k], yerr=fk["len_tokens_std"], color=SERIES[0], fmt="none", elinewidth=0.8, capsize=2)
                ax.set_xscale("log")
                ax.set_xticks(fk["beta"], [f"{b:g}" for b in fk["beta"]])
                ax.minorticks_off()
                ax.set_xlabel(r"DPO $\beta$ (short-run forks)")
                ax.set_title(lab, color=INK)
            fig.tight_layout()
            save(fig, out, "t1_beta_study")
    if (t1 / "length_confounding.csv").exists():
        df = pd.read_csv(t1 / "length_confounding.csv").set_index("model")
        strata = [s for s in ("preferred_longer", "matched", "rejected_longer") if f"acc_{s}" in df]
        vals = {m: {s.replace("_", "-"): df.loc[m, f"acc_{s}"] for s in strata} for m in df.index}
        grouped_bars([s.replace("_", "-") for s in strata], list(df.index), vals, "held-out preference accuracy",
                     "Preference accuracy by length stratum", out, "t1_length_strata", "DPO training set", (0, 1.05))

    # ---------------- Task 2
    t2 = R / "task2_ppo"
    if (t2 / "train_standard.jsonl").exists():
        trajectory_grid(jl(t2 / "train_standard.jsonl"), [
            ("rm_score_mean", "reward-model score"), ("kl_token_mean", "KL to reference"), ("policy_loss", "policy loss"),
            ("value_loss", "value loss"), ("entropy_token_mean", "entropy"), ("clip_fraction", "clip fraction"),
            ("policy_grad_norm", "policy grad norm"), ("response_tokens_mean", "response length"),
            ("explained_variance", "critic explained variance"), ("affected_fraction", "affected-token fraction")],
            "Standard PPO continuation from the supplied midpoint", out, "t2_ppo_standard")
    for prefix, vals_key, legend, name, panels in [
        ("kl_", "kl_values", r"$\beta_{KL}$", "t2_kl_forks",
         [("rm_score_mean", "reward"), ("kl_token_mean", "KL to reference"), ("entropy_token_mean", "entropy"), ("response_tokens_mean", "length")]),
        ("clip_", "clip_values", r"$\epsilon$", "t2_clip_forks",
         [("approx_kl_old_new", "approx KL(old||new)"), ("clip_fraction", "clip fraction"), ("policy_grad_norm", "grad norm"), ("rm_score_mean", "reward")]),
    ]:
        logs = sorted(t2.glob(f"train_{prefix}*.jsonl")) if t2.exists() else []
        logs = [p for p in logs if not p.name.endswith("_meta.json")]
        if logs:
            series = sorted(((p.stem.replace(f"train_{prefix}", ""), jl(p)) for p in logs), key=lambda s: float(s[0]))
            multi_series(series[:4], panels, "Matched short PPO forks from the identical midpoint", out, name, legend)
    if (t2 / "clipping_cached_batch.json").exists():
        c = json.loads((t2 / "clipping_cached_batch.json").read_text())
        eps = list(c["per_epsilon"])
        vals = {"clip fraction": {e: c["per_epsilon"][e]["at_midpoint"]["clip_fraction"] for e in eps},
                "affected-token fraction": {e: c["per_epsilon"][e]["at_midpoint"]["affected_fraction"] for e in eps}}
        grouped_bars(eps, list(vals), vals, "fraction of response tokens", r"Cached rollout batch: clipping vs $\epsilon$",
                     out, "t2_clip_cached", None, (0, 1.05))

    # ---------------- Task 3
    t3 = R / "task3_grpo"
    if (t3 / "train_standard.jsonl").exists():
        trajectory_grid(jl(t3 / "train_standard.jsonl"), [
            ("reward_mean", "reward-model score"), ("kl_token_mean", "KL to reference"), ("group_reward_std_mean", "within-group reward std"),
            ("uninformative_group_fraction", "uninformative-group fraction"), ("policy_loss", "policy loss"), ("grad_norm", "grad norm"),
            ("entropy_token_mean", "entropy"), ("response_tokens_mean", "response length")],
            "Standard GRPO continuation from the supplied midpoint", out, "t3_grpo_standard")
    if (t3 / "group_size_study.json").exists():
        g = json.loads((t3 / "group_size_study.json").read_text())
        ks = sorted(g["partition"], key=int)
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0))
        for i, b in enumerate(("hard", "medium", "easy")):
            y = [g["by_difficulty"][k].get(b, {}).get("informative_group_rate", np.nan) for k in ks]
            axes[0].plot([int(k) for k in ks], y, color=SERIES[i], marker=MARKERS[i], label=b)
        axes[0].plot([int(k) for k in ks], [g["partition"][k]["informative_group_rate"] for k in ks], color=INK2, ls="--", label="all prompts")
        axes[0].set_title("informative-group rate (equal generations)", color=INK)

        axes[1].plot([int(k) for k in ks], [g["partition"][k]["baseline_mse_vs_8sample_mean"] for k in ks], color=SERIES[0], marker="o")
        axes[1].set_title("group-baseline error vs 8-sample mean", color=INK)
        for ax in axes:
            ax.set_xticks([int(k) for k in ks])
            ax.set_xlabel("group size K")
        fig.tight_layout(rect=(0, 0, 0.8, 1))
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, title="difficulty bin", frameon=False, loc="center left", bbox_to_anchor=(0.8, 0.5))
        save(fig, out, "t3_group_size")
    if (t3 / "normalization_comparison.json").exists():
        n = json.loads((t3 / "normalization_comparison.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
        for i, (lt, c) in enumerate(n["conditions"].items()):
            tab = pd.DataFrame(c["length_conditioned"])
            if tab.empty:
                continue
            axes[0].plot(tab["length_mean"], tab["grad_norm_policy_mean"], color=SERIES[i], marker=MARKERS[i], label=lt)
            axes[1].plot(tab["length_mean"], tab["per_token_weight_mean"], color=SERIES[i], marker=MARKERS[i], label=lt)
        axes[0].set_title("per-completion policy grad norm", color=INK)
        axes[1].set_title("per-token loss weight |A|/denominator", color=INK)
        for ax in axes:
            ax.set_xlabel("completion length (tokens, quartile bins)")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        save(fig, out, "t3_normalization")

    # ---------------- Task 4
    t4 = R / "task4_safety"
    if (t4 / "safety_summary.csv").exists():
        df = pd.read_csv(t4 / "safety_summary.csv", index_col=0)
        metrics = {"safe: answer": "safe_answer_rate", "safe: over-refusal": "safe_over_refusal_rate",
                   "unsafe: compliance": "unsafe_compliance_rate", "unsafe: justified refusal": "justified_refusal_rate"}
        pols = list(df.index)[:4]
        vals = {p: {m: float(df.loc[p, c]) for m, c in metrics.items()} for p in pols}
        grouped_bars(list(metrics), pols, vals, "rate (AI judge)", "XSTest safety calibration", out, "t4_safety", "policy", (0, 1.1))

    # ---------------- Task 5
    t5 = R / "task5_feedback"
    if (t5 / "diagnostic_summary.json").exists():
        d = json.loads((t5 / "diagnostic_summary.json").read_text())
        cats = list(d["by_category"])
        mechs = [m for m in ("verifier", "judge") if m in d["by_category"][cats[0]]]
        fig, axes = plt.subplots(1, len(mechs), figsize=(3.2 * len(mechs), 2.2), squeeze=False)
        for ax, m in zip(axes.flat, mechs):
            y = np.arange(len(cats))
            left = np.zeros(len(cats))
            for i, (k, lab) in enumerate([("better_rate", "prefers better"), ("tie_rate", "tie"), ("wrong_rate", "prefers worse")]):
                v = np.array([d["by_category"][c][m][k] for c in cats])
                ax.barh(y, v, left=left, color=SERIES[i], label=lab, edgecolor="white", linewidth=0.8)
                left += v
            ax.set_yticks(y, [c.replace("_", " ") for c in cats])
            ax.set_xlim(0, 1)
            ax.set_title(m, color=INK)
            ax.grid(axis="y", visible=False)
        fig.tight_layout()
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.02))
        save(fig, out, "t5_diagnostics")
    if (t5 / "feedback_comparison.csv").exists():
        df = pd.read_csv(t5 / "feedback_comparison.csv").set_index("policy")
        vals = {"GSM8K": df["gsm_acc"].to_dict(), "SVAMP": df["svamp_acc"].to_dict()}
        grouped_bars(list(df.index), list(vals), vals, "exact-answer accuracy", "In-domain vs transfer accuracy", out, "t5_accuracy", "eval set", (0, 1.1))


if __name__ == "__main__":
    main()
