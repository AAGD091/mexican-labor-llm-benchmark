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
# Figure 1, round-2 layout (4 Nov 2026): stacked blocks, larger text for print legibility (K1 = 2.0,
# as Figure 2); Inference block: case level primary, pooled tests in Appendix E.
K1 = 2.0


def t1(ax, x, y, s, size=11, color=GREY, weight="normal", style="normal", ha="center", va="center", z=5, ls=1.25):
    ax.text(x, y, s, fontsize=size * K1, color=color, fontweight=weight, fontstyle=style, ha=ha, va=va,
            zorder=z, linespacing=ls)


def figure1(path):
    W, H = 200, 240
    fig, ax = canvas(W, H)
    t1(ax, W / 2, 235, "Overview of the Mexican labor-law benchmark", 21, NAVY, "bold")
    t1(ax, W / 2, 226.5, "Four architectural arms share one frozen retrieval substrate, one gold benchmark and one fixed judge;\n"
       "architecture and backbone are the manipulated factors.", 12, GREY, style="italic")

    # ablation ladder
    top = 219
    box(ax, 3, top - 33, 194, 33, "#e3f1df", "#6aa84f", 1.8)
    t1(ax, W / 2, top - 3.6, "ABLATION LADDER", 13.5, "#1b5e20", "bold")
    arms = [("A1  Bare LLM", "parametric only", "no retrieval"),
            ("A2  RAG", "dense retrieval", "k = 4 chunks"),
            ("A3  GraphRAG", "dense retrieval + graph", "entity/relation subgraph"),
            ("A4  Reward-guided\nselection", "dense retrieval + 3 CBR exemplars", "best-of-5 + guarded revision")]
    bw, gap, x0 = 44.5, 4.0, 6
    for i, (t, l1, l2) in enumerate(arms):
        x = x0 + i * (bw + gap); y = top - 30
        box(ax, x, y, bw, 23, "white", "#6aa84f", 1.6)
        if "\n" in t:
            t1(ax, x + bw / 2, y + 18.0, t, 12, NAVY, "bold", ls=1.1)
            t1(ax, x + bw / 2, y + 9.4, l1.replace(" + 3", " +\n3"), 10, "#222", ls=1.15)
            t1(ax, x + bw / 2, y + 3.4, l2, 9.5, "#555", style="italic")
        else:
            t1(ax, x + bw / 2, y + 17.6, t, 12, NAVY, "bold")
            t1(ax, x + bw / 2, y + 10.6, l1, 10, "#222")
            t1(ax, x + bw / 2, y + 5.4, l2, 9.5, "#555", style="italic")
        if i < 3:
            arrow(ax, x + bw + 0.3, y + 11, x + bw + gap - 0.3, y + 11, "#6aa84f")
    t1(ax, W / 2, top - 37.5, "increasing contextual breadth  →  planned contrasts  C1 (retrieval)  ·  C2 (graph)  ·\n"
       "C3 (exemplars + selection, not separable)", 11.5, "#1b5e20", "bold", ls=1.15)

    # frozen substrate
    top = 172
    box(ax, 3, top - 28, 194, 28, "#dde6f2", "#2f5597", 1.8)
    t1(ax, W / 2, top - 3.6, "FROZEN SHARED SUBSTRATE  ·  manifest-asserted · identical in all arms", 13.5, "#2f5597", "bold")
    items = [("Corpus (330 PDFs)", "321 SCJN resolutions\n+ 9 federal laws"),
             ("FAISS dense index", "multilingual-e5-large · 1024-d\n25,461 vectors · k = 4"),
             ("Legal knowledge graph", "63,526 nodes\n192,838 edges"),
             ("Gold benchmark  n = 90", "written by two attorneys\n49 / 26 / 15 · baseline 54.4%")]
    for i, (t, b) in enumerate(items):
        x = x0 + i * (bw + gap); y = top - 25.5
        box(ax, x, y, bw, 17.5, "white", "#2f5597", 1.4)
        t1(ax, x + bw / 2, y + 13.6, t, 11, NAVY, "bold")
        t1(ax, x + bw / 2, y + 5.6, b, 9.5, "#222", ls=1.2)

    # generation
    top = 140
    box(ax, 3, top - 56, 194, 56, "#efeef3", "#888", 1.8)
    t1(ax, W / 2, top - 3.6, "GENERATION  (5 quantized backbones × 4 arms = 20 conditions)", 13.5, "#333", "bold")
    rows = [("A1", "case summary + facts", "", "#efeef3", "#888", "512 tokens", "strategy + citations", "(prose)"),
            ("A2", "case summary + facts", "+ 4 retrieved chunks", "#dde6f2", "#2f5597", "512 tokens", "strategy + citations", "(prose)"),
            ("A3", "case summary + facts", "+ 4 chunks + graph subgraph", "#f6dbe2", "#c06a80", "512 tokens", "strategy + citations", "(prose)"),
            ("A4", "case summary + facts + 4 chunks", "+ 3 CBR exemplars (own case excluded)", "#fbf2cf", "#c9a227",
             "900–1000 tokens", "strategy + citations", "(structured fields)")]
    for i, (a, l1, l2, fc, ec, bud, o1, o2) in enumerate(rows):
        h = 8.8; y = top - 16.5 - i * (h + 1.4)
        box(ax, 6, y, 96, h, fc, ec, 1.4)
        t1(ax, 8.5, y + h / 2, a, 11, NAVY, "bold", ha="left")
        t1(ax, 57, y + h / 2, l1 + ("\n" + l2 if l2 else ""), 9.5, "#222", ls=1.15)
        box(ax, 108, y, 38, h, "white", ec, 1.4)
        t1(ax, 127, y + h / 2, "LLM generation\n" + bud, 9.5, NAVY, ls=1.15)
        box(ax, 152, y, 42, h, "white", ec, 1.4)
        t1(ax, 173, y + h / 2, o1 + "\n" + o2, 9.5, NAVY, ls=1.15)
        arrow(ax, 102.2, y + h / 2, 107.8, y + h / 2)
        arrow(ax, 146.2, y + h / 2, 151.8, y + h / 2)
    t1(ax, W / 2, top - 51.0, "A4: 5 candidates → arg max of fixed reward R → one guarded revision (accepted only if R increases)\n"
       "backbones: llama31-8b-q4 · llama31-8b-q8 · qwen25-7b-q4 · mistral-7b-v03-q4 · gemma2-9b-q4",
       9.5, "#7a5c00", style="italic", ls=1.25)

    # evaluation
    top = 81
    box(ax, 3, top - 26, 194, 26, "#f6dbe2", "#c06a80", 1.8)
    t1(ax, W / 2, top - 3.6, "EVALUATION", 13.5, "#c06a80", "bold")
    ev = [("Fixed judge (never a candidate)", "Phi-3.5-mini Q4_K_M · greedy\n20-question rubric\nscores the rubric only"),
          ("Tier 1 · decision", "accuracy · macro-F1 · MCC\nbootstrap CI  B = 2000, seed 42"),
          ("Tier 2 · statutory grounding", "article P / R / F1\ncanonical normalization ν(·)"),
          ("Tier 3 · surface similarity", "ROUGE · BLEU · BERTScore\n(weak evidence)")]
    for i, (t, b) in enumerate(ev):
        x = x0 + i * (bw + gap); y = top - 24
        box(ax, x, y, bw, 16.5, "white", "#c06a80", 1.4)
        t1(ax, x + bw / 2, y + 13.2, t, 9.8, NAVY, "bold")
        t1(ax, x + bw / 2, y + 5.6, b, 9.2, "#222", ls=1.15)

    # vertical flow arrows in the gaps between blocks
    for yb, yt in [(144.0, 140.0), (84.0, 81.0), (55.0, 52.0)]:
        arrow(ax, W / 2, yb - 0.2, W / 2, yt + 0.2, "#555", 2.2, ms=22)

    # inference (round 2: case level primary; pooled tests in Appendix E)
    top = 52
    box(ax, 3, 1, 194, top - 1, "white", "#aab", 1.8)
    t1(ax, W / 2, top - 3.6, "INFERENCE  (case as unit of analysis, n = 90 · 1,800 observations · paired throughout)",
       12.5, NAVY, "bold")
    inf = [("Case-level contrasts", "n = 90 cases · Wilcoxon,\nHolm · bootstrap\nMDE · TOST ±5 pp"),
           ("Friedman + Wilcoxon", "case-mean article F1\nHolm · rank-biserial"),
           ("Mixed logistic LR + AIC", "arm × backbone interaction\ncase random intercept"),
           ("Exact binomial", "each condition vs\n54.44% majority baseline"),
           ("Pooled (Appendix E)", "McNemar · Friedman ·\nWilcoxon on 450\ncase–backbone pairs")]
    iw, ig = 36.4, 2.0
    for i, (t, b) in enumerate(inf):
        x = 5 + i * (iw + ig); y = top - 21
        box(ax, x, y, iw, 15, "#efeef3", "#888", 1.3)
        t1(ax, x + iw / 2, y + 12.0, t, 9.3, NAVY, "bold")
        t1(ax, x + iw / 2, y + 5.2, b, 8.6, "#333", ls=1.15)
    res = [("C1  retrieval", "#1b5e20", "article F1 0.026 → 0.101 ($p_{Holm}$ = 1.7×10$^{-7}$, $r_{rb}$ = 0.873)\ndecision correctness unchanged ($p_{Holm}$ = 1.00)"),
           ("C2  graph", "#b00020", "ΔF1 = −0.0002 ($r_{rb}$ = −0.012, $p_{Holm}$ = 0.950)\nnon-empty graph context in 84% of cases"),
           ("C3  exemplars + selection", "#7b68b0", "highest article F1 (0.247), not attributable to\nselection (budget, exemplars, structured output)"),
           ("Arm × backbone", "#c9a227", "LR = 28.22, df = 12, p = 0.005 (case effect)\nbackbone spread 0.373 vs arm spread 0.048")]
    rw, rg = 93, 2.0
    for i, (t, c, b) in enumerate(res):
        r, cc = divmod(i, 2); x = 5 + cc * (rw + rg); y = 15.2 - r * 13.2
        box(ax, x, y, rw, 12, "white", c, 1.8)
        t1(ax, x + rw / 2, y + 9.2, t, 10, c, "bold")
        t1(ax, x + rw / 2, y + 4.0, b, 8.8, "#222", ls=1.15)
    fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- Figure 2
# Figure 2, round-2 layout (4 Nov 2026): same content, larger text for print legibility
K2 = 2.0  # font scale for Figure 2 (Figure 1 keeps FS)
FIG2_BOX6 = ["case level (n = 90): Wilcoxon,", "bootstrap, MDE, TOST ±5 pp", "mixed-model LR (case effect)", "pooled tests: Appendix E"]  # round 2: Appendix E (5 Nov 2026)


def txt2(ax, x, y, s, size=11, color=GREY, weight="normal", style="normal", ha="center", va="center", z=5, ls=1.25):
    ax.text(x, y, s, fontsize=size * K2, color=color, fontweight=weight, fontstyle=style, ha=ha, va=va,
            zorder=z, linespacing=ls)


def figure2(path, box6=FIG2_BOX6):
    W, H = 200, 214
    fig, ax = canvas(W, H)
    txt2(ax, W/2, 208, "Methodology of the Mexican labor-law benchmark", 21, NAVY, "bold")
    txt2(ax, W/2, 199.5, "A six-phase sequence under frozen experimental controls, with an explicit audit–correction–re-run cycle;\n"
        "ten of the thirteen defects were corrected and the affected conditions re-executed, and three are disclosed.", 11.5, GREY, style="italic")
    txt2(ax, W/2, 191, "PHASE SEQUENCE", 13, "#333", "bold")
    ph = [("Corpus preparation", "#e3f1df", "#6aa84f", ["330 PDFs: 321 SCJN", "resolutions + 9 federal laws", "chunks 2000/200 · 2400/300", "e5-large → FAISS (25,461)"]),
          ("Benchmark construction", "#dde6f2", "#2f5597", ["two attorneys wrote", "all 90 cases", "PRED-001–060: 49/10/1", "PRED-061–090: 0/16/14"]),
          ("Experimental design", "#e7e2f3", "#7b68b0", ["4 arms × 5 backbones", "frozen substrate + manifest", "manipulated factors:", "arm and backbone"]),
          ("Execution", "#efeef3", "#9aa1ad", ["20 conditions × 90 cases", "per-case checkpoint/resume", "schema validation", "on output"]),
          ("Evaluation", "#f6dbe2", "#c06a80", ["fixed out-of-family judge", "canonical citation ν(·)", "Tier 1 / Tier 2 / Tier 3", ""]),
          ("Statistical inference", "#e3f1df", "#6aa84f", box6)]
    pw, pg, ph_h = 61.33, 4.0, 33
    rows_y = [153, 116]
    for i, (t, fc, ec, lines) in enumerate(ph):
        r, c = divmod(i, 3); x = 4 + c*(pw+pg); y = rows_y[r]
        box(ax, x, y, pw, ph_h, fc, ec, 1.8)
        ax.add_patch(plt.Circle((x+5, y+ph_h-5), 3.0, color=ec, zorder=4))
        txt2(ax, x+5, y+ph_h-5, str(i+1), 12, "white", "bold", z=6)
        txt2(ax, x+pw/2+3, y+ph_h-5, t, 12.5, NAVY, "bold")
        for k, l in enumerate(lines):
            if l: txt2(ax, x+pw/2, y+ph_h-12.5-k*5.0, l, 10.5, "#222")
        if c < 2: arrow(ax, x+pw+0.3, y+ph_h/2, x+pw+pg-0.3, y+ph_h/2)
    # row 1 -> row 2 connector (phase 3 to phase 4)
    ax.plot([4+2*(pw+pg)+pw/2, 4+2*(pw+pg)+pw/2], [153, 151.5], color="#555", lw=1.8)
    ax.plot([4+2*(pw+pg)+pw/2, 4+pw/2], [151.5, 151.5], color="#555", lw=1.8)
    arrow(ax, 4+pw/2, 151.5, 4+pw/2, 149.3)
    box(ax, 4, 104, 192, 8.5, "white", "#2f5597", 1.6, ls="--")
    txt2(ax, 15, 108.2, "FROZEN", 12, "#2f5597", "bold")
    txt2(ax, 112, 108.2, "gold benchmark · index + manifest · prompt scaffold · judge + rubric · metric definitions · seed", 10.5, "#333")
    txt2(ax, W/2, 98.5, "corrections applied  →  affected conditions re-executed under the corrected pipeline", 11.5, "#7a5c00", "bold")
    box(ax, 4, 34.5, 192, 59.5, "#fffdf5", "#c9a227", 1.8, ls="--")
    txt2(ax, W/2, 89.5, "METHODOLOGICAL AUDIT  →  CORRECTION  →  RE-RUN", 12, "#7a5c00", "bold")
    txt2(ax, W/2, 84.3, "(13 defects; 5 arising from cross-arm configuration drift)", 11, "#7a5c00", "bold")
    au = [("Instrument", "#c06a80", ["candidate graded its own output", "judge keyed on echoed identifier", "judge device / offload policy"]),
          ("Contamination", "#b00020", ["exemplar bank included the case", "gold label reached prompt + reward", ""]),
          ("Measurement", "#2f5597", ["citations matched as raw strings", "generation budget unequal across arms", "extraction path differs by arm"]),
          ("Substrate", "#7b68b0", ["arms retrieved from different indexes", "obsolete cache guard → destructive rebuild", "environment reset drifted silently"])]
    aw, ag, ah = 91, 2, 21
    for i, (t, c, lines) in enumerate(au):
        r, cc = divmod(i, 2); x = 8 + cc*(aw+ag); y = 58.5 - r*(ah+1)
        box(ax, x, y, aw, ah, "white", c, 1.6)
        txt2(ax, x+aw/2, y+ah-3.6, t, 12, c, "bold")
        for k, l in enumerate(lines):
            if l: txt2(ax, x+aw/2, y+ah-8.6-k*4.2, "– "+l, 10.3, "#222")
    ax.plot([4, 2, 2], [67, 67, 169.5], color="#c9a227", lw=2); arrow(ax, 2, 169.5, 4, 169.5, "#c9a227")
    ax.plot([196, 198, 198], [67, 67, 132.5], color="#c9a227", lw=2); arrow(ax, 198, 132.5, 196, 132.5, "#c9a227")
    box(ax, 4, 1, 192, 31.5, "white", "#aab", 1.8)
    txt2(ax, W/2, 29.3, "OUTPUTS", 13, NAVY, "bold")
    out = [("90-case gold benchmark", "#1b5e20", "49 / 26 / 15 · baseline 54.4% · two batches · no double annotation"),
           ("20-condition results table", "#2f5597", "1,800 case-level observations · 3 metric tiers + process metrics"),
           ("Correction register", "#b8860b", "13 documented failure modes with corrective procedures"),
           ("Reproducibility package", "#7b68b0", "run-provenance map · checksums · per-case outputs · analysis scripts")]
    for i, (t, c, b) in enumerate(out):
        r, cc = divmod(i, 2); x = 8 + cc*(aw+ag); y = 14.3 - r*12
        box(ax, x, y, aw, 11.3, "#fafafd", c, 1.6)
        txt2(ax, x+aw/2, y+8.0, t, 11.5, c, "bold")
        txt2(ax, x+aw/2, y+3.4, b, 9.2, "#222")
    fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    figure1(f"{d}/fig_benchmark_overview_r1.png")
    figure2(f"{d}/methodology_framework_r1.png")
    print("saved")
