#!/usr/bin/env python3
"""Figure A5 (revised manuscript, Appendix D): ablation results across the
contextual-broadening ladder. Every value is computed from saved study files:

  --long  draft_per_case_long.csv   (arm, model, case_id, gold, pred, correct, article_f1)
  --agg   draft_benchmark_table.csv (per-condition aggregate table; ROUGE/BLEU,
          article P/R, empty_pred, trunc_at_ceiling)
  --out   output PNG

Truncation is shown only where token metadata exist (A1-A3); A4 is "n.m."
(not measured), as in Table 14. Bootstrap: cases resampled, B = 2000, seed 42.

Round 2 (4 Nov 2026): panel (b) labels give the case-level Holm p-values of Table 17;
panel (g) shows the matched-pairs rank-biserial correlation on case-mean article F1
(n = 90) instead of Cliff's delta on 450 pooled rows; smaller canvas for larger printed text.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

ARMS = ["Bare LLM", "RAG+LLM", "GraphRAG+LLM", "RL+LLM"]
TICK = ["A1\nBare", "A2\nRAG", "A3\nGraphRAG", "A4\nSelection"]
TICKFS = 11  # round 2: arm tick labels slightly smaller so "GraphRAG" and "Selection" do not touch
LEG = ["A1 Bare LLM", "A2 RAG", "A3 GraphRAG", "A4 Selection"]
ACOL = ["#9aa1ad", "#2f5597", "#c06a80", "#7b68b0"]
MODELS = ["gemma2-9b-q4", "llama31-8b-q4", "llama31-8b-q8", "mistral-7b-v03-q4", "qwen25-7b-q4"]
MCOL = ["#2e7d32", "#1565c0", "#6a1b9a", "#c62828", "#ef6c00"]
CLASSES = ["Procedente", "Improcedente", "Parcialmente Improcedente"]
MAJ = 49 / 90


def cliffs(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation, as in src/revision_reanalysis.py (round 2)."""
    from scipy import stats
    d = np.asarray(d); nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True); ap.add_argument("--agg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.long); d["pred"] = d["pred"].fillna("NONE")
    g = pd.read_csv(a.agg)

    mcc = {(r, m): matthews_corrcoef(x.gold, x.pred) for (r, m), x in d.groupby(["arm", "model"])}
    acc = d.groupby(["arm", "model"]).correct.mean() * 100

    # article-F1: mean over all 450 case-backbone rows, CI by resampling cases
    rng = np.random.default_rng(42)
    cases = d.case_id.unique()
    piv = {r: d[d.arm == r].pivot(index="case_id", columns="model", values="article_f1").loc[cases].mean(axis=1).values for r in ARMS}
    idx = rng.integers(0, len(cases), size=(2000, len(cases)))
    f1m = {r: piv[r].mean() for r in ARMS}
    f1ci = {r: np.percentile(piv[r][idx].mean(axis=1), [2.5, 97.5]) for r in ARMS}

    fig, ax = plt.subplots(3, 3, figsize=(13.5, 14.5))  # round 2: smaller canvas = larger printed text
    plt.subplots_adjust(hspace=0.62, wspace=0.42, top=0.875, bottom=0.055, left=0.075, right=0.99)
    fig.suptitle("Ablation results across the contextual-broadening ladder", fontsize=19, fontweight="bold", color="#1f2f57")
    fig.text(0.5, 0.925, "4 arms × 5 quantized backbones × 90 cases = 1,800 case-level observations\n"
             "C1 retrieval  ·  C2 graph  ·  C3 exemplars + selection (not separable)",
             ha="center", fontsize=13, style="italic", color="#444")

    def style(axx, title, xfs=12):
        axx.set_title(title, fontsize=13.5, fontweight="bold", color="#1f2f57", pad=10)
        axx.spines[["top", "right"]].set_visible(False)
        axx.grid(axis="y", ls=":", alpha=0.6)
        axx.tick_params(axis="y", labelsize=12); axx.tick_params(axis="x", labelsize=xfs)

    # (a) MCC by backbone
    x = np.arange(4); A = ax[0, 0]
    for m, c in zip(MODELS, MCOL):
        A.plot(x, [mcc[(r, m)] for r in ARMS], "-o", color=c, lw=2.2, ms=7, label=m)
    A.plot(x, [np.mean([mcc[(r, m)] for m in MODELS]) for r in ARMS], "--D", color="k", lw=2.5, ms=8, label="mean")
    A.set_xticks(x, TICK, fontsize=TICKFS); A.set_ylabel("MCC", fontsize=12); A.set_ylim(0, 0.98); A.set_yticks(np.arange(0, 0.71, 0.1))
    A.legend(ncol=2, fontsize=10.5, frameon=False, loc="upper center", columnspacing=1.0, handlelength=1.6)
    style(A, "(a) MCC by backbone", TICKFS)

    # (b) article-F1 with case-bootstrap CI
    B = ax[0, 1]
    for i, r in enumerate(ARMS):
        lo, hi = f1ci[r]
        B.bar(i, f1m[r], color=ACOL[i], edgecolor="k", width=0.65)
        B.errorbar(i, f1m[r], yerr=[[f1m[r] - lo], [hi - f1m[r]]], color="k", capsize=6, lw=1.6)
        B.text(i, hi + 0.006, f"{f1m[r]:.3f}", ha="center", fontsize=12, fontweight="bold")
    B.annotate("", xy=(1, 0.175), xytext=(0, 0.175), arrowprops=dict(arrowstyle="<->", color="#1b5e20", lw=2))
    B.text(0.5, 0.181, "C1  $p_{Holm}$ = 1.7×10$^{-7}$", ha="center", va="bottom", color="#1b5e20", fontsize=11, fontweight="bold")
    B.annotate("", xy=(2, 0.235), xytext=(1, 0.235), arrowprops=dict(arrowstyle="<->", color="#b00020", lw=2))
    B.text(1.5, 0.241, "C2 n.s.  $p_{Holm}$ = 0.950", ha="center", va="bottom", color="#b00020", fontsize=11, fontweight="bold")
    B.set_xticks(x, TICK, fontsize=TICKFS); B.set_ylabel("article F1 (95% case-bootstrap CI)", fontsize=12); B.set_ylim(0, 0.36)
    style(B, "(b) Statutory grounding (article F1)", TICKFS)

    # (c) accuracy vs majority baseline
    C = ax[0, 2]; w = 0.16
    for k, (m, c) in enumerate(zip(MODELS, MCOL)):
        C.bar(x + (k - 2) * w, [acc[(r, m)] for r in ARMS], w, color=c, label=m)
    C.axhline(MAJ * 100, color="#b00020", ls="--", lw=2.2)
    C.text(-0.45, 75.5, "- - majority-class baseline 54.4%", color="#b00020", fontsize=11.5, fontweight="bold")
    C.set_xticks(x, TICK, fontsize=TICKFS); C.set_ylabel("accuracy (%)", fontsize=12); C.set_ylim(40, 78)
    style(C, "(c) Decision accuracy vs baseline", TICKFS)

    # (d) delta MCC
    D = ax[1, 0]; steps = ["A1→A2", "A2→A3", "A3→A4"]
    for k, (m, c) in enumerate(zip(MODELS, MCOL)):
        D.bar(np.arange(3) + (k - 2) * w, [mcc[(ARMS[i + 1], m)] - mcc[(ARMS[i], m)] for i in range(3)], w, color=c)
    D.axhline(0, color="k", lw=1); D.set_xticks(np.arange(3), steps); D.set_ylabel("Δ MCC", fontsize=12)
    D.text(1, 0.355, "no step sign-consistent\nacross backbones", ha="center", va="top", color="#b00020", fontsize=11.5, fontweight="bold", linespacing=1.1)
    D.set_ylim(-0.33, 0.38)
    style(D, "(d) Δ MCC per broadening step")

    # (e) article precision vs recall per condition
    E = ax[1, 1]
    for i, r in enumerate(ARMS):
        s = g[g["Artefacto"] == r]
        E.scatter(s["Article-R"], s["Article-P"], s=110, color=ACOL[i], edgecolor="k", label=LEG[i], zorder=3)
        E.scatter(s["Article-R"].mean(), s["Article-P"].mean(), s=380, marker="*", color=ACOL[i], edgecolor="k", zorder=4)
    E.text(0.465, 0.03, "A4: highest recall;\nexemplar access, structured\noutput and larger budget\nact in the same direction",
           fontsize=11, color="#5e4b9c", fontweight="bold", va="bottom", ha="right", linespacing=1.1)
    E.set_xlabel("article recall", fontsize=12); E.set_ylabel("article precision", fontsize=12)
    E.legend(fontsize=10.5, loc="upper left", bbox_to_anchor=(0.43, 1.0), frameon=False, handletextpad=0.2, borderaxespad=0.1); E.set_ylim(0.02, 0.30); E.set_xlim(-0.01, 0.47)
    style(E, "(e) Grounding P vs R (★ = arm mean)")

    # (f) per-class recall, pooled over backbones
    F = ax[1, 2]; w4 = 0.2
    for i, r in enumerate(ARMS):
        s = d[d.arm == r]
        rec = [(s[s.gold == c].pred == c).mean() for c in CLASSES]
        F.bar(np.arange(3) + (i - 1.5) * w4, rec, w4, color=ACOL[i], label=LEG[i])
    F.set_xticks(np.arange(3), ["Procedente", "Impro-\ncedente", "Parcialmente\nImprocedente"])
    F.set_ylabel("recall", fontsize=12); F.set_ylim(0, 1.3); F.set_yticks(np.arange(0, 1.01, 0.2)); F.legend(fontsize=10.5, frameon=False, ncol=2, loc="upper center", columnspacing=1.0, handlelength=1.4)
    style(F, "(f) Per-class recall (all backbones)")  # round 2: "pooled" reserved for the 450-pair tests

    # (g) Round-2 (4 Nov 2026): matched-pairs rank-biserial on case-mean article F1 (n = 90),
    # the effect size of Table 17; case means rounded to 10 decimals as in revision_reanalysis.py
    G = ax[2, 0]
    cm = {r: np.round(piv[r], 10) for r in ARMS}
    cs = [("C1\nA1 → A2", rank_biserial(np.round(cm[ARMS[1]] - cm[ARMS[0]], 10)), "#1b5e20"),
          ("C2\nA2 → A3", rank_biserial(np.round(cm[ARMS[2]] - cm[ARMS[1]], 10)), "#b00020"),
          ("C3\nA2 → A4", rank_biserial(np.round(cm[ARMS[3]] - cm[ARMS[1]], 10)), "#7b68b0")]
    for k, (lab, v, c) in enumerate(cs[::-1]):
        G.barh(k, v, color=c, edgecolor="k", height=0.55)
        G.text(max(v, 0) + 0.02, k, f"$r_{{rb}}$ = {v:.3f}".replace("-", "\u2212"), va="center", fontsize=12, fontweight="bold")
    G.axvline(0, color="k", lw=1)
    G.set_yticks(range(3), [c[0] for c in cs[::-1]], fontsize=12)
    G.set_xlim(-0.1, 1.3); G.set_ylim(-0.5, 2.6); G.set_xlabel("matched-pairs rank-biserial $r_{rb}$\n(case-mean article F1, n = 90)", fontsize=12)
    style(G, "(g) Effect size on grounding"); G.grid(axis="x", ls=":", alpha=0.4)

    # (h) truncation and unparseable decisions (mean count per condition)
    H = ax[2, 1]
    tr = [g[g["Artefacto"] == r]["trunc_at_ceiling"] for r in ARMS]
    em = [g[g["Artefacto"] == r]["empty_pred"].mean() for r in ARMS]
    for i in range(4):
        if tr[i].notna().all() and len(tr[i]):
            H.bar(i - 0.18, tr[i].mean(), 0.34, color="#c9a227", edgecolor="k", label="truncated at 512-token ceiling" if i == 0 else None)
            H.text(i - 0.18, tr[i].mean() + 1.2, f"{tr[i].mean():.1f}", ha="center", fontsize=11, fontweight="bold")
        else:
            H.text(i - 0.2, 3, "n.m.", ha="center", fontsize=11, fontweight="bold", color="#555")
        H.bar(i + 0.18, em[i], 0.34, color="#c06a80", edgecolor="k", label="unparseable decision" if i == 0 else None)
        H.text(i + 0.22, em[i] + 1.2, f"{em[i]:.1f}", ha="center", fontsize=11, fontweight="bold")
    H.set_xticks(x, TICK, fontsize=TICKFS); H.set_ylabel("cases per condition (of 90)", fontsize=12); H.set_ylim(0, 100)
    H.legend(fontsize=10.5, frameon=False, loc="upper right", handlelength=1.4)
    H.text(3, 14, "token usage\nnot stored", ha="center", fontsize=10.5, color="#555", style="italic")
    style(H, "(h) Process metrics (mean per condition)", TICKFS)

    # (i) Tier-3 surface similarity
    I = ax[2, 2]; mets = [("ROUGE-1", 1), ("ROUGE-L", 1), ("BLEU", 4)]
    for i, r in enumerate(ARMS):
        s = g[g["Artefacto"] == r]
        I.bar(np.arange(3) + (i - 1.5) * w4, [s[mm].mean() * f for mm, f in mets], w4, color=ACOL[i], label=LEG[i])
    I.set_xticks(np.arange(3), ["ROUGE-1", "ROUGE-L", "BLEU ×4"]); I.set_ylabel("score", fontsize=12); I.set_ylim(0, 0.66); I.set_yticks(np.arange(0, 0.51, 0.1))
    I.legend(fontsize=10.5, frameon=False, ncol=2, loc="upper center", columnspacing=1.0, handlelength=1.4)
    I.text(2.45, 0.50, "weak evidence: A4 output is\nstructured, not prose", ha="right", va="top", fontsize=11, color="#b00020", fontweight="bold")
    style(I, "(i) Tier 3: surface similarity (weak)")

    fig.savefig(a.out, dpi=300)
    print("saved", a.out)
    print("article-F1 means", {r: round(v, 4) for r, v in f1m.items()})
    print("rank-biserial", [(c[0], round(c[1], 3)) for c in cs])
    print("trunc means", [None if t.isna().any() else round(t.mean(), 1) for t in tr], "empty", [round(e, 1) for e in em])


if __name__ == "__main__":
    main()
