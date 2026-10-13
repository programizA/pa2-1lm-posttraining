"""LaTeX tables for the PDF report, filled directly from the saved result files (no hand-copied numbers).

    python -m analysis.report_numbers && python -m analysis.report_tables     # writes report/tables/*.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def f(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def ci(c, d=2):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def write(out, name, body):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.tex").write_text(body)
    print("  wrote", out / f"{name}.tex")


def t_dpo(R, N, out):
    s = pd.read_csv(R / "task1_dpo" / "dpo_summary.csv").set_index("condition")
    lc = pd.read_csv(R / "task1_dpo" / "length_confounding.csv").set_index("model")
    rows = []
    spec = [("sft", "SFT (ref.)", "", None), ("standard", "Standard", "", "standard"),
            ("beta_0.03", r"$\beta=0.03^\ast$", "", "beta_0.03"), ("beta_0.1", r"$\beta=0.10^\ast$", "", "beta_0.1"),
            ("beta_0.3", r"$\beta=0.30^\ast$", "", "beta_0.3")]
    for key, name, budget, nk in spec:
        r = s.loc[key]
        if nk is None:
            acc, d = "--", "--"
            ident = "--"
        else:
            a = N["t1_heldout_acc"][nk]
            acc = f"{f(a['acc'])} {ci(a['ci95'])}"
            p = N["t1_vs_sft"][nk]
            d = f"{f(p['reward_diff'])} {ci(p['ci95'])}"
            ident = pct(p["identical_text"])
        rows.append(f"{name} & {f(r.heldout_dpo_loss, 3)} & {acc} & {f(r.heldout_margin_mean)} & {f(r.kl_seq_mean, 3)} & "
                    f"{f(r.reward_mean)} & {d} & {r.len_tokens_mean:.0f} ({r.len_tokens_std:.0f}) & {ident} \\\\")
    lb = N["t1_heldout_acc"]["length_balanced"]
    p = N["t1_vs_sft"]["length_balanced"]
    ev = json.loads((R / "task1_dpo" / "eval_length_balanced.json").read_text())
    g = ev["generation"]
    rows.append(r"\midrule")
    rows.append(f"Length-bal. & {f(ev['pairs']['overall']['dpo_loss'], 3)} & {f(lb['acc'])} {ci(lb['ci95'])} & "
                f"{f(ev['pairs']['overall']['margin']['mean'])} & {f(g['kl_seq_mean'], 3)} & {f(g['reward']['mean'])} & {f(p['reward_diff'])} {ci(p['ci95'])} & "
                f"{g['response_tokens']['mean']:.0f} ({g['response_tokens']['std']:.0f}) & {pct(p['identical_text'])} \\\\")
    body = r"""\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Held-out pairs ($n=290$)} & \multicolumn{5}{c}{Generations on 116 held-out prompts} \\
\cmidrule(lr){2-4}\cmidrule(l){5-9}
 & loss & accuracy [95\% CI] & $m$ & KL/seq & RM & $\Delta$RM [95\% CI] & tokens (sd) & same (\%) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_dpo", body)
    wl = (f"word-limit compliance: SFT {pct(s.loc['sft'].word_limit_compliance)}\\%, standard {pct(lc.loc['standard'].word_limit_compliance)}\\%, "
          f"length-balanced {pct(lc.loc['length_balanced'].word_limit_compliance)}\\%")
    write(out, "txt_dpo_wordlimit", wl)


def t_ppo(R, N, out):
    c = pd.read_csv(R / "task2_ppo" / "clipping_summary.csv").set_index("epsilon")
    k = pd.read_csv(R / "task2_ppo" / "kl_summary.csv").set_index("kl_beta")
    rows = []
    for e in [0.05, 0.2, 0.5]:
        r = c.loc[e]
        p = N["t2_vs_midpoint"][f"clip_{e:g}"]
        rows.append(rf"$\epsilon={e:g}$ & {pct(r.cache_clip_fraction, 1)} & {pct(r.cache_affected_fraction, 1)} & {r.approx_kl_max:.1e} & "
                    rf"{f(r.heldout_reward)} & {f(p['reward_diff'])} {ci(p['ci95'])} & {f(r.heldout_kl_seq, 3)} & {f(r.heldout_entropy)} & {r.heldout_len_mean:.0f} \\")
    rows.append(r"\midrule")
    for b in [0.0, 0.1, 0.2]:
        r = k.loc[b]
        p = N["t2_vs_midpoint"][f"kl_{b:g}"]
        rows.append(rf"$\beta_{{KL}}={b:g}$ & -- & -- & -- & {f(r.heldout_reward_mean)} & {f(p['reward_diff'])} {ci(p['ci95'])} & "
                    rf"{f(r.heldout_kl_seq, 3)} & {f(r.heldout_entropy)} & {r.heldout_len_mean:.0f} \\")
    m = json.loads((R / "task2_ppo" / "eval_midpoint.json").read_text())
    rows.append(r"\midrule")
    rows.append(rf"midpoint (start) & -- & -- & -- & {f(m['reward']['mean'])} & -- & {f(m['kl_seq_mean'], 3)} & {f(m['entropy_token_mean'])} & {m['response_tokens']['mean']:.0f} \\")
    body = r"""\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{2}{c}{Cached batch (\%)} & Stability & \multicolumn{5}{c}{Held-out (54 prompts) after 8 updates} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(l){5-9}
Fork & clip & affected & max $\widehat{\mathrm{KL}}_{old,new}$ & RM & $\Delta$RM [95\% CI] & KL/seq & entropy & tokens \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_ppo", body)


def t_grpo(R, N, out):
    g = json.loads((R / "task3_grpo" / "group_size_study.json").read_text())
    rows = []
    for k in ["2", "4", "8"]:
        p, pr, bd = g["partition"][k], g["partition_rel_tol"][k], g["by_difficulty"][k]
        rows.append(rf"{k} & {p['n_groups']} & {f(p['informative_group_rate'])} & {f(pr['informative_group_rate'])} & {f(p['mean_within_group_reward_std'])} & "
                    rf"{f(p['var_centered_advantage'])} & {f(p['baseline_mse_vs_8sample_mean'])} & {f(bd['hard']['informative_group_rate'])} / {f(bd['easy']['informative_group_rate'])} & "
                    rf"{f(bd['hard']['mean_within_group_reward_std'])} / {f(bd['easy']['mean_within_group_reward_std'])} \\")
    body = r"""\begin{tabular}{@{}cccccccc c@{}}
\toprule
$K$ & groups & inf. & inf.\ (5\%) & sd & var$(r-\bar r)$ & base.\ MSE & hard/easy inf. & hard/easy sd \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_grpo_k", body)
    n = pd.read_csv(R / "task3_grpo" / "normalization_comparison.csv").set_index("loss_type")
    rows = []
    for key, name in (("grpo", r"canonical ($1/|o|$)"), ("dr_grpo", r"Dr.\ GRPO ($1/L$)")):
        r = n.loc[key]
        p = N["t3_vs_midpoint"][f"norm_{key}"]
        rows.append(rf"{name} & {f(r.heldout_reward)} & {f(p['reward_diff'])} {ci(p['ci95'])} & {r.heldout_kl_token * 1e4:.2f} & {r.heldout_len_mean:.0f} ({r.heldout_len_std:.0f}) & "
                    rf"{f(r.train_grad_norm_mean, 3)} & {f(r.corr_len_gradnorm)} \\")
    body = r"""\begin{tabular}{@{}lcccccc@{}}
\toprule
Loss & RM & $\Delta$RM [95\% CI] & KL/tok.\ ($10^{-4}$) & tokens (sd) & grad norm & corr(len., grad) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_grpo_norm", body)


def t_safety(R, out):
    t4 = R / "task4_safety"
    s = pd.read_csv(t4 / "safety_summary.csv", index_col=0)
    rc = pd.read_csv(t4 / "refusal_check_summary.csv", index_col=0)
    rows = []
    for p in ["sft", "dpo", "ppo", "grpo"]:
        a, b = s.loc[p], rc.loc[p]
        rows.append(rf"{p.upper()} & {pct(a.safe_answer_rate, 1)} & {pct(a.safe_over_refusal_rate, 1)} & {pct(a.unsafe_compliance_rate, 1)} & {pct(a.justified_refusal_rate, 1)} & "
                    rf"{pct(a.ambiguous_rate, 1)} & {pct(a.class_inconsistent_rate, 1)} & {pct(b.check_over_refusal, 1)} [{pct(b.check_over_refusal_lo)}, {pct(b.check_over_refusal_hi)}] & "
                    rf"{pct(b.check_unsafe_compliance_upper, 1)} & {a.response_tokens_mean:.0f} ({a.response_tokens_std:.0f}) \\")
    body = r"""\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{6}{c}{AI judge (\%)} & \multicolumn{2}{c}{Refusal rule (\%)} & \\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}
 & answer & over-ref. & unsafe & justified & ambig. & incons. & over-ref.\ [95\% CI] & unsafe$^\dagger$ & tokens (sd) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_safety", body)
    conf = pd.read_csv(t4 / "audit_confusion_manual_rows_judge_cols.csv", index_col=0)
    rc_conf = pd.read_csv(t4 / "refusal_check_confusion_manual_rows.csv", index_col=0)
    labs = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL"]
    short = {"SAFE_ANSWER": "SA", "JUSTIFIED_REFUSAL": "JR", "UNSAFE_COMPLIANCE": "UC", "OVER_REFUSAL": "OR"}
    rows = []
    for l in labs:
        rows.append(f"{short[l]} & " + " & ".join(str(int(conf.loc[l, c])) for c in labs) + " & & " + " & ".join(str(int(rc_conf.loc[l, c])) for c in labs) + r" \\")
    body = r"""\begin{tabular}{@{}lcccc c cccc@{}}
\toprule
 & \multicolumn{4}{c}{AI judge} & & \multicolumn{4}{c}{Refusal rule} \\
\cmidrule(lr){2-5}\cmidrule(l){7-10}
Manual & SA & JR & UC & OR & & SA & JR & UC & OR \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_audit", body)


def t_feedback(R, out):
    t5 = R / "task5_feedback"
    gs = json.loads((t5 / "gsm_summary.json").read_text())["policies"]
    sv = json.loads((t5 / "transfer_summary.json").read_text())["policies"]
    ex = json.loads((t5 / "task5_extra.json").read_text())
    rows = []
    for p in ["sft", "rlvr", "rlaif"]:
        a, b = gs[p], sv[p]
        la, lb = ex["GSM8K"]["policies"][p]["lenient_accuracy"], ex["SVAMP"]["policies"][p]["lenient_accuracy"]
        va, vb = a.get("vs_sft", {}), b.get("vs_sft", {})
        wtl = lambda v: f"{v['wins']}/{v['ties']}/{v['losses']}" if "wins" in v else "--"
        wr = lambda v: f(v["win_rate"], 3) if "wins" in v else "--"
        ag = lambda v: pct(v["verifier_judge_agreement_3way"]) if "wins" in v else "--"
        rows.append(rf"{p.upper()} & {pct(a['exact_accuracy'], 1)} & {pct(la, 1)} & {pct(a['format_compliance'])} & {a['response_tokens']['mean']:.0f} ({a['response_tokens']['std']:.0f}) & "
                    rf"{wtl(va)} & {wr(va)} & {ag(va)} & {pct(b['exact_accuracy'])} & {b['response_tokens']['mean']:.0f} ({b['response_tokens']['std']:.0f}) & {wtl(vb)} & {wr(vb)} & "
                    rf"{f(100 * (b['exact_accuracy'] - a['exact_accuracy']), 1)} \\")
    body = r"""\begin{tabular}{@{}lccccccc|ccccc@{}}
\toprule
 & \multicolumn{7}{c|}{GSM8K, in-domain ($n=300$)} & \multicolumn{5}{c}{SVAMP, transfer ($n=100$)} \\
 & exact & lenient & format & tokens (sd) & W/T/L & win & agree & exact & tokens (sd) & W/T/L & win & $\Delta^\ddagger$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_feedback", body)
    d = json.loads((t5 / "diagnostic_summary.json").read_text())
    rows = []
    for m in ["verifier", "judge"]:
        rows.append(rf"{m} & {pct(d[f'S_reason_{m}'])} & {pct(d[f'S_reason_{m}_tie'])} & {pct(d[f'S_reason_{m}_wrong'])} & "
                    rf"{pct(d[f'S_outcome_{m}'], 1)} & {pct(d[f'S_outcome_{m}_tie'])} & {pct(d[f'S_outcome_{m}_wrong'], 1)} & "
                    rf"{pct(d['by_category']['persuasive_filler_correct'][m]['wrong_rate'])} \\")
    body = r"""\begin{tabular}{@{}lccccccc@{}}
\toprule
 & \multicolumn{3}{c}{$S_{\text{reason}}$ (\%)} & \multicolumn{3}{c}{$S_{\text{outcome}}$ (\%)} & filler preferred \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Mechanism & better & tie & wrong & better & tie & wrong & (\%) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write(out, "tab_diag", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/tables")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    N = json.loads((R / "report_numbers.json").read_text())
    t_dpo(R, N, out)
    t_ppo(R, N, out)
    t_grpo(R, N, out)
    t_safety(R, out)
    t_feedback(R, out)


if __name__ == "__main__":
    main()
