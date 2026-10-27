#!/usr/bin/env python3
"""Figures 1 and 2 of the revised manuscript (overview and methodology diagrams).
All numbers are taken from the revised manuscript text and tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#1f2f57"; GREY = "#444444"


def canvas(w, h):
    fig = plt.figure(figsize=(w / 10, h / 10))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec="#888", lw=1.6, ls="-", r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


FS = 1.25  # global font scale for print legibility


def txt(ax, x, y, s, size=11, color=GREY, weight="normal", style="normal", ha="center", va="center", z=5, ls=1.25):
    ax.text(x, y, s, fontsize=size * FS, color=color, fontweight=weight, fontstyle=style, ha=ha, va=va,
            zorder=z, linespacing=ls)


def arrow(ax, x0, y0, x1, y1, color="#555", lw=1.8, z=3, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z))


# --------------------------------------------------------------------------- Figure 1
def figure1(path):
    W, H = 200, 128
    fig, ax = canvas(W, H)
    txt(ax, W / 2, 124, "Overview of the Mexican labor-law benchmark", 21, NAVY, "bold")
    txt(ax, W / 2, 119.6, "Four architectural arms share one frozen retrieval substrate, one gold benchmark and one fixed judge; "
        "architecture and backbone are the manipulated factors.", 12, GREY, style="italic")

    # ablation ladder
    box(ax, 3, 98, 194, 18, "#e3f1df", "#6aa84f", 1.6)
    txt(ax, 13, 107, "Ablation\nladder", 17, "#1b5e20", "bold")
    ax.plot([23, 23], [99.5, 114.5], color="#6aa84f", lw=1.5)
    arms = [("A1  Bare LLM", "parametric only", "no retrieval"),
            ("A2  RAG", "dense retrieval", "k = 4 chunks"),
            ("A3  GraphRAG", "dense retrieval + graph", "entity/relation subgraph"),
            ("A4  Reward-guided selection", "dense retrieval + 3 CBR exemplars", "best-of-5 + guarded revision")]
    bw, gap, x0 = 39, 4.5, 26.5
    for i, (t, l1, l2) in enumerate(arms):
        x = x0 + i * (bw + gap)
        box(ax, x, 100, bw, 14, "white", "#6aa84f", 1.6)
        txt(ax, x + bw / 2, 110.6, t, 13, NAVY, "bold")
        txt(ax, x + bw / 2, 106.4, l1, 11, "#222")
        txt(ax, x + bw / 2, 102.8, l2, 10.5, "#555", style="italic")
        if i < 3:
            arrow(ax, x + bw + 0.4, 107, x + bw + gap - 0.4, 107, "#6aa84f")
    txt(ax, W / 2, 94.6, "increasing contextual breadth  →  planned contrasts  C1 (retrieval)  ·  C2 (graph)  ·  "
        "C3 (exemplars + selection, not separable)", 13, "#1b5e20", "bold")

    # frozen substrate
    box(ax, 3, 33, 46, 58, "#dde6f2", "#2f5597", 1.8)
    txt(ax, 26, 87.6, "FROZEN SHARED SUBSTRATE", 13.5, "#2f5597", "bold")
    txt(ax, 26, 84.2, "manifest-asserted · identical in all arms", 10.5, GREY, style="italic")
    items = [("Corpus (330 PDFs)", "321 SCJN resolutions\n+ 9 federal laws"),
             ("FAISS dense index", "multilingual-e5-large · 1024-d\n25,461 vectors · k = 4"),
             ("Legal knowledge graph", "63,526 nodes\n192,838 edges"),
             ("Gold benchmark  n = 90", "written by two attorneys\n49 / 26 / 15 · baseline 54.4%")]
    for i, (t, b) in enumerate(items):
        y = 69.5 - i * 12
        box(ax, 6, y, 40, 11, "white", "#2f5597", 1.4)
        txt(ax, 26, y + 8.2, t, 12, NAVY, "bold")
        txt(ax, 26, y + 3.6, b, 10.5, "#222")

    # generation
    box(ax, 53, 33, 92, 58, "#efeef3", "#888", 1.8)
    txt(ax, 99, 87.6, "GENERATION  (5 quantized backbones × 4 arms = 20 conditions)", 13.5, "#333", "bold")
    rows = [("A1", "case summary + facts", "", "#efeef3", "#888", "512 tokens", "strategy + citations", "(prose)"),
            ("A2", "case summary + facts", "+ 4 retrieved chunks", "#dde6f2", "#2f5597", "512 tokens", "strategy + citations", "(prose)"),
            ("A3", "case summary + facts", "+ 4 chunks + graph subgraph", "#f6dbe2", "#c06a80", "512 tokens", "strategy + citations", "(prose)"),
            ("A4", "case summary + facts + 4 chunks", "+ 3 CBR exemplars (own case excluded)", "#fbf2cf", "#c9a227",
             "900–1000 tokens", "strategy + citations", "(structured fields)")]
    for i, (a, l1, l2, fc, ec, bud, o1, o2) in enumerate(rows):
        y = 72.5 - i * 10.8
        h = 9.6
        box(ax, 56, y, 38, h, fc, ec, 1.4)
        txt(ax, 58, y + h - 2.1, a, 12, NAVY, "bold", ha="left")
        txt(ax, 76, y + 5.6 if l2 else y + 4.4, l1, 10.3, "#222")
        if l2:
            txt(ax, 76, y + 2.3, l2, 10.3, "#222")
        box(ax, 98, y, 18, h, "white", ec, 1.4)
        txt(ax, 107, y + 6.3, "LLM generation", 10.8, NAVY, "bold")
        txt(ax, 107, y + 2.7, bud, 10, "#555")
        box(ax, 120, y, 22, h, "white", ec, 1.4)
        txt(ax, 131, y + 6.3, o1, 10.8, NAVY, "bold")
        txt(ax, 131, y + 2.7, o2, 10, "#555")
        arrow(ax, 94.2, y + h / 2, 97.8, y + h / 2)
        arrow(ax, 116.2, y + h / 2, 119.8, y + h / 2)
    txt(ax, 99, 37.4, "A4: 5 candidates → arg max of fixed reward R → one guarded revision (accepted only if R increases)",
        10, "#7a5c00", style="italic")
    txt(ax, 99, 34.8, "backbones: llama31-8b-q4 · llama31-8b-q8 · qwen25-7b-q4 · mistral-7b-v03-q4 · gemma2-9b-q4",
        10.3, "#333", "bold")

    # evaluation
    box(ax, 149, 33, 48, 58, "#f6dbe2", "#c06a80", 1.8)
    txt(ax, 173, 87.6, "EVALUATION", 13.5, "#c06a80", "bold")
    ev = [("Fixed judge (never a candidate)", "Phi-3.5-mini Q4_K_M · greedy\n20-question rubric · scores the rubric only"),
          ("Tier 1 · decision", "accuracy · macro-F1 · MCC\nbootstrap CI  B = 2000, seed 42"),
          ("Tier 2 · statutory grounding", "article P / R / F1\ncanonical normalization ν(·)"),
          ("Tier 3 · surface similarity", "ROUGE · BLEU · BERTScore\n(weak evidence)")]
    for i, (t, b) in enumerate(ev):
        y = 69.5 - i * 12
        box(ax, 152, y, 42, 11, "white", "#c06a80", 1.4)
        txt(ax, 173, y + 8.2, t, 11.5, NAVY, "bold")
        txt(ax, 173, y + 3.6, b, 10, "#222")
    arrow(ax, 49.3, 62, 52.7, 62); arrow(ax, 145.3, 62, 148.7, 62)

    # inference
    box(ax, 3, 2, 194, 28, "white", "#aab", 1.6)
    txt(ax, W / 2, 27, "INFERENCE  (1,800 case-level observations · paired throughout)", 13.5, NAVY, "bold")
    inf = [("McNemar exact", "450 case–backbone pairs\nHolm, 6 contrasts"),
           ("Case-level contrasts", "n = 90 cases · Wilcoxon\nbootstrap · MDE · TOST ±5 pp"),
           ("Friedman + Wilcoxon", "per-case article F1\nHolm · Cliff's δ"),
           ("Mixed logistic LR + AIC", "arm × backbone interaction\ncase random intercept"),
           ("Exact binomial", "each condition vs\n54.44% majority baseline")]
    iw, ig = 36.4, 2.0
    for i, (t, b) in enumerate(inf):
        x = 5 + i * (iw + ig)
        box(ax, x, 15.2, iw, 9.6, "#efeef3", "#888", 1.3)
        txt(ax, x + iw / 2, 22.4, t, 11.0, NAVY, "bold")
        txt(ax, x + iw / 2, 18.3, b, 9.0, "#333", ls=1.15)
    res = [("C1  retrieval", "#1b5e20", "article F1 0.026 → 0.101 ($p_{Holm}$ = 1.7×10$^{-11}$, δ = 0.173)\ndecision correctness unchanged ($p_{Holm}$ = 1.00)"),
           ("C2  graph", "#b00020", "ΔF1 = 0.0002 (δ = 0.007, $p_{Holm}$ = 0.670)\nnon-empty graph context in 84% of cases"),
           ("C3  exemplars + selection", "#7b68b0", "highest article F1 (0.247), not attributable to\nselection (budget, exemplars, structured output)"),
           ("Arm × backbone", "#c9a227", "LR = 28.22, df = 12, p = 0.005 (case effect)\nbackbone spread 0.373 vs arm spread 0.048")]
    rw, rg = 46.1, 2.0
    for i, (t, c, b) in enumerate(res):
        x = 5 + i * (rw + rg)
        box(ax, x, 3.6, rw, 10.2, "white", c, 1.8)
        txt(ax, x + rw / 2, 11.4, t, 11.0, c, "bold")
        txt(ax, x + rw / 2, 7.1, b, 8.8, "#222", ls=1.15)
    fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- Figure 2
def figure2(path):
    W, H = 200, 122
    fig, ax = canvas(W, H)
    txt(ax, W / 2, 118, "Methodology of the Mexican labor-law benchmark", 21, NAVY, "bold")
    txt(ax, W / 2, 113.6, "A six-phase sequence under frozen experimental controls, with an explicit audit–correction–re-run cycle; "
        "corrections detected mid-campaign were applied and all affected conditions re-executed.", 11.5, GREY, style="italic")
    txt(ax, W / 2, 108.6, "PHASE SEQUENCE", 13, "#333", "bold")
    ph = [("Corpus\npreparation", "#e3f1df", "#6aa84f", "#6aa84f",
           ["330 PDFs: 321 SCJN", "resolutions + 9 federal laws", "chunks 2000/200 · 2400/300", "e5-large → FAISS (25,461)"]),
          ("Benchmark\nconstruction", "#dde6f2", "#2f5597", "#2f5597",
           ["two attorneys wrote", "all 90 cases", "PRED-001–060: 49/10/1", "PRED-061–090: 0/16/14"]),
          ("Experimental\ndesign", "#e7e2f3", "#7b68b0", "#7b68b0",
           ["4 arms × 5 backbones", "frozen substrate + manifest", "manipulated factors:", "arm and backbone"]),
          ("Execution", "#efeef3", "#9aa1ad", "#9aa1ad",
           ["20 conditions × 90 cases", "per-case checkpoint/resume", "schema validation", "on output"]),
          ("Evaluation", "#f6dbe2", "#c06a80", "#c06a80",
           ["fixed out-of-family judge", "canonical citation ν(·)", "Tier 1 / Tier 2 / Tier 3", ""]),
          ("Statistical\ninference", "#e3f1df", "#6aa84f", "#6aa84f",
           ["paired McNemar · Friedman", "case-level contrasts,", "MDE, TOST ±5 pp", "mixed-model LR · Holm"])]
    pw, pg = 30.0, 2.4
    for i, (t, fc, ec, cc, lines) in enumerate(ph):
        x = 4 + i * (pw + pg)
        box(ax, x, 78, pw, 27, fc, ec, 1.6)
        ax.add_patch(plt.Circle((x + 4.2, 100.6), 2.6, color=cc, zorder=4))
        txt(ax, x + 4.2, 100.6, str(i + 1), 12, "white", "bold", z=6)
        txt(ax, x + pw / 2 + 2.6, 100.6, t, 12, NAVY, "bold")
        for k, l in enumerate(lines):
            if l:
                txt(ax, x + pw / 2, 93.6 - k * 3.6, l, 10.2, "#222")
        if i < 5:
            arrow(ax, x + pw + 0.3, 91.5, x + pw + pg - 0.3, 91.5)
    box(ax, 70, 69.5, 126, 6.2, "white", "#2f5597", 1.5, ls="--")
    txt(ax, 79, 72.6, "FROZEN", 12, "#2f5597", "bold")
    txt(ax, 141, 72.6, "gold benchmark  ·  index + manifest  ·  prompt scaffold  ·  judge + rubric  ·  metric definitions  ·  seed",
        10.5, "#333")
    txt(ax, W / 2, 65.2, "corrections applied  →  affected conditions re-executed under the corrected pipeline", 11.5, "#7a5c00", "bold")

    box(ax, 4, 31, 192, 31, "#fffdf5", "#c9a227", 1.6, ls="--")
    txt(ax, W / 2, 58.6, "METHODOLOGICAL AUDIT  →  CORRECTION  →  RE-RUN   (13 defects; 5 arising from cross-arm configuration drift)",
        12, "#7a5c00", "bold")
    au = [("Instrument", "#c06a80", ["candidate graded its own output", "judge keyed on echoed identifier", "judge device / offload policy"]),
          ("Contamination", "#b00020", ["exemplar bank included the case", "gold label reached prompt + reward", ""]),
          ("Measurement", "#2f5597", ["citations matched as raw strings", "generation budget unequal across arms", "extraction path differs by arm"]),
          ("Substrate", "#7b68b0", ["arms retrieved from different indexes", "obsolete cache guard → destructive rebuild", "environment reset drifted silently"])]
    aw, ag = 44.5, 3
    for i, (t, c, lines) in enumerate(au):
        x = 8 + i * (aw + ag)
        box(ax, x, 33.5, aw, 21.5, "white", c, 1.5)
        txt(ax, x + aw / 2, 51.5, t, 12, c, "bold")
        for k, l in enumerate(lines):
            if l:
                txt(ax, x + aw / 2, 46 - k * 4.6, "– " + l, 10.3, "#222")
    # feedback arrows from audit to phases 1 and 6
    ax.plot([4, 2, 2], [46.5, 46.5, 88], color="#c9a227", lw=2); arrow(ax, 2, 88, 4, 88, "#c9a227")
    ax.plot([196, 198, 198], [46.5, 46.5, 88], color="#c9a227", lw=2); arrow(ax, 198, 88, 196, 88, "#c9a227")

    box(ax, 4, 2, 192, 26, "white", "#aab", 1.6)
    txt(ax, W / 2, 25, "OUTPUTS", 13, NAVY, "bold")
    out = [("90-case gold benchmark", "#1b5e20", "49 / 26 / 15 · baseline 54.4%\ntwo batches · no double annotation"),
           ("20-condition results table", "#2f5597", "1,800 case-level observations\n3 metric tiers + process metrics"),
           ("Correction register", "#b8860b", "13 documented failure modes\nwith corrective procedures"),
           ("Reproducibility package", "#7b68b0", "run-provenance map · checksums\nper-case outputs · analysis scripts")]
    for i, (t, c, b) in enumerate(out):
        x = 8 + i * (aw + ag)
        box(ax, x, 4.5, aw, 16, "#fafafd", c, 1.6)
        txt(ax, x + aw / 2, 16.6, t, 12, c, "bold")
        txt(ax, x + aw / 2, 10, b, 10.5, "#222")
    fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    figure1(f"{d}/fig_benchmark_overview_r1.png")
    figure2(f"{d}/methodology_framework_r1.png")
    print("saved")
