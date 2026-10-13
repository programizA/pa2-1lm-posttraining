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
            # Means with bootstrap 95% CIs from the per-example files, so noise is not mistaken for a trend.
            # Unscaled margin m (comparable across beta) replaces the held-out DPO loss, which is not.
            rng = np.random.default_rng(0)

            def ci(x):
                x = np.asarray(x, float)
                b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
                return x.mean(), *np.percentile(b, [2.5, 97.5])

            def per_example(name):
                pairs = pd.DataFrame(jl(t1 / f"eval_{name}_pairs.jsonl")) if (t1 / f"eval_{name}_pairs.jsonl").exists() else None
                gens = pd.DataFrame(jl(t1 / f"eval_{name}_generations.jsonl")) if (t1 / f"eval_{name}_generations.jsonl").exists() else None
                return pairs, gens

            panels = [("acc", "held-out pref. accuracy"), ("margin", "held-out margin $m$ (unscaled)"),
                      ("reward", "reward-model score"), ("len", "response length (tokens)")]
            stats = {p: [] for p, _ in panels}
            for cond in fk["condition"]:
                pairs, gens = per_example(cond)
                stats["acc"].append(ci(pairs["correct"].astype(float)) if pairs is not None else (np.nan,) * 3)
                stats["margin"].append(ci(pairs["margin"]) if pairs is not None else (np.nan,) * 3)
                stats["reward"].append(ci(gens["reward"]) if gens is not None else (np.nan,) * 3)
                stats["len"].append(ci(gens["response_tokens"]) if gens is not None else (np.nan,) * 3)
            sft_pairs, sft_gens = per_example("sft")
            fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
            for ax, (k, lab) in zip(axes, panels):
                m, lo, hi = (np.array(v) for v in zip(*stats[k]))
                ax.errorbar(fk["beta"], m, yerr=[m - lo, hi - m], color=SERIES[0], marker="o", capsize=2, elinewidth=0.8)
                if k == "reward" and sft_gens is not None:
                    ax.axhline(sft_gens["reward"].mean(), color=INK2, ls="--", lw=0.8, label="SFT")
                    ax.legend(frameon=False, loc="lower left")
                if k == "len" and sft_gens is not None:
                    ax.axhline(sft_gens["response_tokens"].mean(), color=INK2, ls="--", lw=0.8)
                if k == "acc":
                    ax.axhline(0.5, color=INK2, ls=":", lw=0.8)
                if k == "margin":
                    ax.axhline(0.0, color=INK2, ls=":", lw=0.8)
                ax.set_xscale("log")
                ax.set_xticks(fk["beta"], [f"{b:g}" for b in fk["beta"]])
                ax.minorticks_off()
                ax.set_xlabel(r"DPO $\beta$ (short-run forks)")
                ax.set_title(lab, color=INK)
            fig.tight_layout()
            save(fig, out, "t1_beta_study")
    strat_files = {m: t1 / f"eval_{m}_dpo_length_stratified_eval_pairs.jsonl" for m in ("standard", "length_balanced")}
    if all(f.exists() for f in strat_files.values()):
        rng = np.random.default_rng(0)
        data = {m: pd.DataFrame(jl(f)) for m, f in strat_files.items()}
        strata = sorted(data["standard"]["stratum"].unique(),
                        key=lambda s: 0 if s.startswith("pref") else 2 if s.startswith("rej") else 1)
        labels = [s.replace("_", "-").replace("length-", "") for s in strata]
        x = np.arange(len(strata))
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
        # (a) accuracy per stratum with bootstrap 95% CI, both training sets
        w = 0.38
        for i, m in enumerate(data):
            means, errs = [], [[], []]
            for s in strata:
                v = data[m].loc[data[m]["stratum"] == s, "correct"].astype(float).to_numpy()
                b = np.percentile([rng.choice(v, len(v)).mean() for _ in range(2000)], [2.5, 97.5])
                means.append(v.mean()); errs[0].append(v.mean() - b[0]); errs[1].append(b[1] - v.mean())
            axes[0].bar(x + (i - 0.5) * w, means, w * 0.92, yerr=errs, capsize=2, color=SERIES[i], label=m.replace("_", "-"),
                        edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 0.8})
        axes[0].axhline(0.5, color=INK2, ls=":", lw=0.8)
        axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1); axes[0].set_ylabel("held-out pref. accuracy")
        axes[0].set_title("(a) accuracy by length stratum", color=INK); axes[0].grid(axis="x", visible=False)
        axes[0].legend(title="DPO training set", frameon=False, loc="upper left", fontsize=6, title_fontsize=6)
        # (b) mechanism: change in summed log-prob of winner and loser (standard model)
        d = data["standard"]
        for i, (col, lab) in enumerate([("policy_logratio_chosen", "preferred"), ("policy_logratio_rejected", "rejected")]):
            vals = [d.loc[d["stratum"] == s, col].mean() for s in strata]
            axes[1].bar(x + (i - 0.5) * w, vals, w * 0.92, color=SERIES[i + 2], label=lab, edgecolor="white", linewidth=0.8)
        axes[1].axhline(0, color=INK2, lw=0.8)
        axes[1].set_xticks(x, labels); axes[1].set_ylabel(r"$\Delta \log \pi$ vs reference (nats)")
        axes[1].set_title("(b) standard DPO: log-prob change per response", color=INK); axes[1].grid(axis="x", visible=False)
        axes[1].legend(frameon=False, loc="lower left", fontsize=6)
        fig.tight_layout()
        save(fig, out, "t1_length_strata")

    # ---------------- Task 2
    t2 = R / "task2_ppo"
    if (t2 / "train_standard.jsonl").exists():
        log2 = jl(t2 / "train_standard.jsonl")
        # Per-update explained variance is computed within ONE response, whose returns barely vary, so it is
        # ill-conditioned (values of -500); critic quality is shown across updates in t2_critic_ratio instead.
        trajectory_grid(log2, [
            ("rm_score_mean", "reward-model score"), ("kl_token_mean", "KL to reference"), ("policy_loss", "policy loss"),
            ("value_loss", "value loss"), ("entropy_token_mean", "entropy"), ("clip_fraction", "clip fraction"),
            ("policy_grad_norm", "policy grad norm"), ("response_tokens_mean", "response length")],
            "Standard PPO continuation from the supplied midpoint", out, "t2_ppo_standard")
        upd = [r["update"] for r in log2]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
        axes[0].plot(upd, [r["task_reward_mean"] for r in log2], color=SERIES[0], marker="o", label="reward received")
        axes[0].plot(upd, [r["value_first_token_mean"] for r in log2], color=SERIES[1], marker="s", label=r"critic forecast $V(s_0)$")
        r_ = np.array([r["task_reward_mean"] for r in log2]); v_ = np.array([r["value_first_token_mean"] for r in log2])
        axes[0].set_title(f"(a) critic forecast vs outcome (corr {abs(np.corrcoef(r_, v_)[0, 1]):.2f})"
                          if abs(np.corrcoef(r_, v_)[0, 1]) < 0.005 else
                          f"(a) critic forecast vs outcome (corr {np.corrcoef(r_, v_)[0, 1]:+.2f})", color=INK)
        axes[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
        lo = [min(e["ratio_min"] for e in r["epochs"]) for r in log2]
        hi = [max(e["ratio_max"] for e in r["epochs"]) for r in log2]
        axes[1].fill_between(upd, lo, hi, color=SERIES[0], alpha=0.35, linewidth=0, label="range of ratios in update")
        for eps, ls in ((0.05, "--"), (0.2, ":")):
            axes[1].axhline(1 + eps, color=INK2, ls=ls, lw=0.8, label=rf"$1\pm{eps:g}$")
            axes[1].axhline(1 - eps, color=INK2, ls=ls, lw=0.8)
        axes[1].set_ylim(0.75, 1.25)
        axes[1].set_title(r"(b) ratio $\pi_{new}/\pi_{old}$ vs clipping bands", color=INK)
        axes[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
        for ax in axes:
            ax.set_xlabel("update"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        save(fig, out, "t2_critic_ratio")
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
            ("uninformative_group_fraction", "uninformative-group fraction"),
            ("trained_completion_fraction", "completions trained (not masked)"), ("grad_norm", "grad norm"),
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

        # The K=8 baseline IS the 8-sample mean it is compared with, so its error is 0 by construction: not plotted.
        kb = [k for k in ks if int(k) < 8]
        axes[1].plot([int(k) for k in kb], [g["partition"][k]["baseline_mse_vs_8sample_mean"] for k in kb], color=SERIES[0], marker="o")
        axes[1].set_title("baseline error vs 8-sample mean\n(K=8 is the reference itself)", color=INK)
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.set_xticks([int(k) for k in ks])
            ax.set_xlabel("group size K")
        fig.tight_layout(rect=(0, 0, 0.8, 1))
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, title="difficulty bin", frameon=False, loc="center left", bbox_to_anchor=(0.8, 0.5))
        save(fig, out, "t3_group_size")
    if (t3 / "normalization_comparison.json").exists():
        n = json.loads((t3 / "normalization_comparison.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
        for i, (lt, c) in enumerate(n["conditions"].items()):
            tab = pd.DataFrame(c["length_conditioned"])
            if tab.empty:
                continue
            tab = tab.sort_values("length_mean")  # bins are labels; order them by length, not alphabetically
            label = {"grpo": "canonical GRPO (1/|o|)", "dr_grpo": "Dr. GRPO (1/L)"}.get(lt, lt)
            axes[0].plot(tab["length_mean"], tab["grad_norm_policy_mean"], color=SERIES[i], marker=MARKERS[i], label=label)
            axes[1].plot(tab["length_mean"], tab["per_token_weight_mean"], color=SERIES[i], marker=MARKERS[i], label=label)
        # Log scales: the two losses differ ~10x in overall gradient size, and the question is the SLOPE with length.
        axes[0].set_title("per-completion policy gradient norm", color=INK)
        axes[1].set_title(r"per-token weight $|A|$ / denominator", color=INK)
        for ax in axes:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("completion length (tokens, quartile-bin mean)")
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
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
    if (t4 / "refusal_check_summary.csv").exists():
        # Left: over-refusal on benign prompts, judge vs refusal rule (Wilson 95% CI).
        # Right: which benign XSTest types get refused, per policy.
        s = pd.read_csv(t4 / "refusal_check_summary.csv", index_col=0)
        bt = pd.read_csv(t4 / "refusal_check_over_refusal_by_type.csv", index_col=0)
        pols = list(s.index)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
        x = np.arange(len(pols))
        # Neutral greys here: the series colours on the right panel mean policies.
        jb = a1.bar(x - 0.2, s["judge_over_refusal"], 0.37, color="#b9b8b3", label="AI judge", edgecolor="white")
        for b, v in zip(jb, s["judge_over_refusal"]):
            a1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0f}" if v == 0 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6, color=INK2)
        lo = s["check_over_refusal"] - s["check_over_refusal_lo"]
        hi = s["check_over_refusal_hi"] - s["check_over_refusal"]
        a1.bar(x + 0.2, s["check_over_refusal"], 0.37, color=INK2, label="refusal rule", edgecolor="white",
               yerr=[lo, hi], error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
        a1.set_xticks(x, pols)
        a1.set_ylabel("over-refusal rate (benign prompts)")
        a1.set_ylim(0, max(0.1, float(s["check_over_refusal_hi"].max()) * 1.35))
        a1.grid(axis="x", visible=False)
        a1.legend(frameon=False, loc="upper left")
        a1.set_title("the judge never reports over-refusal", color=INK)
        y = np.arange(len(bt))
        h = 0.8 / len(pols)
        for i, p in enumerate(pols):
            a2.barh(y + (i - (len(pols) - 1) / 2) * h, bt[p], h * 0.92, color=SERIES[i], label=p,
                    edgecolor="white", linewidth=0.5)
        a2.set_yticks(y, [t.replace("_", " ") for t in bt.index])
        a2.invert_yaxis()
        a2.set_xlabel("refusal rate (refusal rule)")
        a2.set_xlim(0, 1)
        a2.grid(axis="y", visible=False)
        a2.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), title="policy")
        a2.set_title("over-refusal by benign XSTest type", color=INK)
        fig.tight_layout()
        save(fig, out, "t4_over_refusal")

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
            # tie is "no signal", so it gets a neutral grey; better/worse get the two strongest series colours
            for k, lab, col in [("better_rate", "prefers better", SERIES[0]), ("tie_rate", "tie", "#c9c8c3"),
                                ("wrong_rate", "prefers worse", SERIES[1])]:
                v = np.array([d["by_category"][c][m][k] for c in cats])
                ax.barh(y, v, left=left, color=col, label=lab, edgecolor="white", linewidth=0.8)
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
    if (t5 / "task5_format_breakdown.csv").exists():
        # What the verifier's 0s are made of: right answer without a `####` line vs actually wrong.
        fb = pd.read_csv(t5 / "task5_format_breakdown.csv")
        fig, ax = plt.subplots(figsize=(5.2, 2.4))
        labels, xs, x = [], [], 0.0
        for ds, grp in fb.groupby("dataset", sort=False):
            for _, r in grp.iterrows():
                xs.append(x); labels.append(f"{r['policy']}\n{ds}"); x += 1
            x += 0.6
        parts = [("correct_and_formatted", "correct, with #### (reward 1)", SERIES[0]),
                 ("correct_but_unformatted", "correct, no #### (reward 0)", SERIES[3]),
                 ("other", "wrong or no number (reward 0)", "#c9c8c3")]
        bottom = np.zeros(len(fb))
        for col, lab, c in parts:
            v = fb[col].to_numpy()
            ax.bar(xs, v, 0.8, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=0.8)
            for xi, b0, vi in zip(xs, bottom, v):
                if vi >= 0.08:
                    ax.text(xi, b0 + vi / 2, f"{vi:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if c != "#c9c8c3" else INK)
            bottom += v
        ax.set_xticks(xs, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of problems")
        ax.grid(axis="x", visible=False)
        ax.set_title("what the exact verifier sees", color=INK)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        save(fig, out, "t5_format")


if __name__ == "__main__":
    main()
