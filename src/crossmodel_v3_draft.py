#!/usr/bin/env python3
"""
CrossModelCompar v2 (DRAFT builder)
===================================
Builds the arm x model benchmark table from checkpoint artifacts.

Design notes (see session audit):
  * Everything is RECOMPUTED from checkpoints. The per-run CSVs are not trusted
    for Tier-2 because the notebooks' compute_article_hit_at_k() compares raw
    strings with only .strip().lower(), which scores ~0 for RAG/GraphRAG purely
    because their surface form differs from gold. A canonical normalizer is
    applied identically to every arm here.
  * Handles partial data: prints a coverage matrix and builds whatever exists.
  * Arm differences live ONLY in the ARMS registry (adapter pattern).

Usage:  python3 crossmodel_v2_draft.py [artifact_dir] [out_dir]
"""
import json, os, re, sys, glob
import numpy as np, pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm import normalize_articles

ART_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
# Colab:  ART_DIR = "/content/drive/MyDrive/SCJN_RAG/results"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs"
GOLD_FILE = "gold_prediction_90_compatible.json"

LABELS = ["Procedente", "Improcedente", "Parcialmente Improcedente"]
MODELS = ["llama31-8b-q4", "qwen25-7b-q4", "mistral-7b-v03-q4",
          "gemma2-9b-q4", "llama31-8b-q8"]
# Ablation ladder order (increasing contextual breadth), used for display.
ARM_ORDER = ["Bare LLM", "RAG+LLM", "GraphRAG+LLM", "RL+LLM"]
# Generation ceiling of the single-pass arms. Generations landing exactly on it
# are truncated mid-text, which systematically suppresses Tier-2 recall because
# Spanish legal analyses place operative citations in concluding passages.
GEN_CEILING = 512
NBOOT, SEED = 2000, 42

# ── Arm registry: the ONLY place arm-specific knowledge lives ────────────────
ARMS = {
    # Ablation floor: no retrieval of any kind. Record shape is identical to the
    # RAG arm (the notebook was forked from it), so the same adapters apply.
    "Bare LLM": dict(
        pattern="checkpoint_bare_{model}.jsonl",
        pred=lambda r: r.get("pred_dec"),
        arts=lambda r: r.get("pred_arts", []),
        text=lambda r: r.get("analysis", ""),
        ref=lambda r, GT: r.get("ground_truth", ""),
        tier3_native=True,
    ),
    "RAG+LLM": dict(
        pattern="checkpoint_ragonly_{model}.jsonl",
        pred=lambda r: r.get("pred_dec"),
        arts=lambda r: r.get("pred_arts", []),
        text=lambda r: r.get("analysis", ""),
        ref=lambda r, GT: r.get("ground_truth", ""),
        tier3_native=True,
    ),
    "GraphRAG+LLM": dict(
        pattern="checkpoint_graphrag_{model}.jsonl",
        pred=lambda r: r.get("pred_dec"),
        arts=lambda r: r.get("pred_arts", []),
        text=lambda r: r.get("analysis", ""),
        ref=lambda r, GT: r.get("ground_truth", ""),
        tier3_native=True,
    ),
    "RL+LLM": dict(
        pattern="checkpoint_{model}__grounded-guard.jsonl",
        pred=lambda r: r.get("pred_decision"),
        arts=lambda r: (r.get("final_strategy_json") or {}).get("articulos_clave", [])
                       if isinstance(r.get("final_strategy_json"), dict) else [],
        text=lambda r: " ".join(
            str((r.get("final_strategy_json") or {}).get(k, ""))
            for k in ("estrategia", "razonamiento")
        ) if isinstance(r.get("final_strategy_json"), dict) else "",
        # RL stores no ground_truth -> join gold conclusions by case_id
        ref=lambda r, GT: GT.get(r["case_id"], ""),
        tier3_native=False,   # genre mismatch: strategy JSON fields vs analysis prose
    ),
}


def load_gold():
    p = os.path.join(ART_DIR, GOLD_FILE)
    cases = json.load(open(p, encoding="utf-8"))["cases"]
    GD = {g["case_id"]: g["gold_standard"]["decision"] for g in cases}
    GA = {g["case_id"]: g["expected_cel_output"]["articulos_citados"] for g in cases}
    GT = {g["case_id"]: " ".join(g["expected_cel_output"]["conclusiones_legales"])
          for g in cases}
    return GD, GA, GT


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def find_artifact(basename):
    """Recursively locate `basename` under ART_DIR.

    Artifacts live in per-run subfolders, so a flat listing misses them.
    Two real hazards handled here:
      * PILOT files (e.g. *_pilot__grounded-guard.jsonl) are 3-5 case smoke
        tests, NOT full runs — including one would silently corrupt a cell.
      * DUPLICATE basenames exist across folders from successive notebook
        versions (e.g. RL v5 vs v10 runs of the same model). We take the most
        recently modified and report every collision so the choice is visible
        rather than silent.
    """
    hits = [p for p in glob.glob(os.path.join(ART_DIR, "**", basename),
                                 recursive=True)
            if "_pilot" not in os.path.basename(p)]
    if not hits:
        return None
    if len(hits) > 1:
        hits.sort(key=os.path.getmtime, reverse=True)
        import datetime as _dt
        print(f"  ⚠️  {len(hits)} copies of {basename} — using most recent:")
        for i, h in enumerate(hits):
            ts = _dt.datetime.fromtimestamp(os.path.getmtime(h)).strftime("%Y-%m-%d %H:%M")
            print(f"        {'USING ' if i == 0 else '  skip'} {ts}  "
                  f"{os.path.getsize(h):>8,}B  {os.path.relpath(h, ART_DIR)}")
    return hits[0]


def prf(pred, gold):
    if not pred and not gold: return (1., 1., 1.)
    if not pred or not gold: return (0., 0., 0.)
    tp = len(pred & gold)
    p = tp / len(pred); r = tp / len(gold)
    return (p, r, 2 * p * r / (p + r) if p + r else 0.)


def boot_ci(vals, rng, nboot=NBOOT):
    v = np.asarray(vals, dtype=float); n = len(v)
    if n == 0: return (0., 0.)
    stats = [v[rng.integers(0, n, n)].mean() for _ in range(nboot)]
    return tuple(np.percentile(stats, [2.5, 97.5]))


def main():
    GD, GA, GT = load_gold()
    rng = np.random.default_rng(SEED)

    # ── Coverage matrix ─────────────────────────────────────────────────
    cov, paths = {}, {}
    print("Scanning artifacts recursively under:", ART_DIR, "\n")
    for arm, cfg in ARMS.items():
        for m in MODELS:
            p = find_artifact(cfg["pattern"].format(model=m))
            paths[(arm, m)] = p
            cov[(arm, m)] = len(jl(p)) if p else 0
    covdf = pd.DataFrame(
        [[cov[(a, m)] for m in MODELS] for a in ARMS],
        index=list(ARMS), columns=MODELS)
    print("=" * 78)
    print("COVERAGE MATRIX (cases found per condition)")
    print("=" * 78)
    print(covdf.to_string())
    have = sum(1 for v in cov.values() if v > 0)
    print(f"\nconditions with data: {have}/{len(cov)}   "
          f"complete (60 cases): {sum(1 for v in cov.values() if v == 60)}")

    mc = Counter(GD.values()).most_common(1)[0]
    print(f"majority-class baseline: '{mc[0]}' = {mc[1]}/{len(GD)} "
          f"({mc[1]/len(GD)*100:.1f}%)\n")

    # ── Metrics (lazy imports so the script still runs without them) ────
    try:
        from sklearn.metrics import f1_score, matthews_corrcoef
        from rouge_score import rouge_scorer
        import sacrebleu
        rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        METRICS_OK = True
    except ImportError as e:
        print(f"⚠️  metric libs missing ({e}) — install scikit-learn rouge_score sacrebleu")
        return

    try:
        from bert_score import score as bert_score
        BERT_OK = True
    except Exception:
        BERT_OK = False
        print("ℹ️  bert_score unavailable — BERTScore-F1 will be blank "
              "(needs a HuggingFace model download).\n")

    rows, percase = [], []
    for arm, cfg in ARMS.items():
        for m in MODELS:
            p = paths.get((arm, m))
            if not p:
                continue
            recs = sorted(jl(p), key=lambda r: r["case_id"])
            cids = [r["case_id"] for r in recs]
            y_true = [GD[c] for c in cids]
            y_pred = [(cfg["pred"](r) or "") for r in recs]
            corr = np.array([1. if a == b else 0. for a, b in zip(y_true, y_pred)])

            lo, hi = boot_ci(corr, rng)
            mf1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0

            P, R, F = [], [], []
            for r in recs:
                ps = normalize_articles(cfg["arts"](r))
                gs = normalize_articles(GA[r["case_id"]])
                a, b, c = prf(ps, gs); P.append(a); R.append(b); F.append(c)

            # Process metric: generations terminating exactly at the token
            # ceiling are truncated. Reported per condition (see paper §8.4).
            _ct = [ (r.get("row") or {}).get("completion_tokens") for r in recs ]
            _trunc = int(sum(1 for v in _ct if v == GEN_CEILING))

            # Provenance: the fixed judge must be identical in every condition.
            # A condition whose rubric artifact names a different judge is not
            # comparable and must be quarantined, not averaged in.
            _judge = ""
            _rj = os.path.join(os.path.dirname(p),
                               os.path.basename(p)
                                 .replace("checkpoint_", "rubric_evaluation_results_")
                                 .replace(".jsonl", ".json"))
            if os.path.exists(_rj):
                try:
                    _jd = json.load(open(_rj, encoding="utf-8"))
                    _judge = _jd.get("judge_model") or _jd.get("judge") or ""
                except Exception:
                    _judge = "unreadable"

            preds = [cfg["text"](r) or "" for r in recs]
            refs = [cfg["ref"](r, GT) or "" for r in recs]
            pairs = [(x, y) for x, y in zip(preds, refs) if x.strip() and y.strip()]
            if pairs:
                sc = [rs.score(y, x) for x, y in pairs]
                r1 = np.mean([s["rouge1"].fmeasure for s in sc])
                r2 = np.mean([s["rouge2"].fmeasure for s in sc])
                rl = np.mean([s["rougeL"].fmeasure for s in sc])
                bleu = sacrebleu.corpus_bleu([x for x, _ in pairs],
                                             [[y for _, y in pairs]]).score / 100
                bsc = ""
                if BERT_OK:
                    try:
                        _, _, f1t = bert_score([x for x, _ in pairs], [y for _, y in pairs],
                                               lang="es", verbose=False)
                        bsc = round(float(f1t.mean()), 4)
                    except Exception:
                        bsc = ""
            else:
                r1 = r2 = rl = bleu = 0.; bsc = ""

            rows.append({
                "Artefacto": arm, "Modelo base": m, "N": len(recs),
                "ROUGE-1": round(r1, 4), "ROUGE-2": round(r2, 4), "ROUGE-L": round(rl, 4),
                "BLEU": round(bleu, 4), "BERTScore-F1": bsc,
                "Decision accuracy": f"{int(corr.sum())}/{len(recs)} ({corr.mean()*100:.1f}%)",
                "Acc 95% CI": f"[{lo*100:.1f}, {hi*100:.1f}]",
                "Macro-F1": round(mf1, 4), "MCC": round(mcc, 4),
                "Article-P": round(np.mean(P), 4), "Article-R": round(np.mean(R), 4),
                "Article-F1": round(np.mean(F), 4),
                "empty_pred": int(sum(1 for v in y_pred if not v)),
                "trunc_at_ceiling": _trunc,
                "judge": _judge,
                "tier3_comparable": cfg["tier3_native"],
                "source_file": os.path.relpath(p, ART_DIR),
            })
            for i, r in enumerate(recs):
                percase.append({"arm": arm, "model": m, "case_id": r["case_id"],
                                "gold": y_true[i], "pred": y_pred[i],
                                "correct": int(corr[i]), "article_f1": F[i]})

    if not rows:
        print("No artifacts found — nothing to build.")
        return

    df = pd.DataFrame(rows)
    df["_ord"] = df["Artefacto"].apply(lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
    df = df.sort_values(["_ord", "Modelo base"]).drop(columns=["_ord"])
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "draft_benchmark_table.csv"), index=False)
    pd.DataFrame(percase).to_csv(os.path.join(OUT_DIR, "draft_per_case_long.csv"), index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
    print("=" * 78)
    print("DRAFT BENCHMARK TABLE  (arm x model)")
    print("=" * 78)
    print(df.to_string(index=False))

    # ── Arm-level means (only over models present in ALL arms) ──────────
    common = set(df["Modelo base"])
    for a in df["Artefacto"].unique():
        common &= set(df[df["Artefacto"] == a]["Modelo base"])
    if common:
        print(f"\nARM MEANS over models common to all arms ({sorted(common)}):")
        sub = df[df["Modelo base"].isin(common)]
        num = ["Macro-F1", "MCC", "Article-P", "Article-R", "Article-F1",
               "ROUGE-1", "ROUGE-L", "BLEU"]
        _means = sub.groupby("Artefacto")[num].mean().round(4)
        _means = _means.reindex([a for a in ARM_ORDER if a in _means.index])
        print(_means.to_string())

    # ── Provenance audit ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PROVENANCE AUDIT")
    print("=" * 78)
    judges = sorted(set(x for x in df["judge"] if x))
    if len(judges) == 1:
        print(f"  \u2705 judge identical in every condition: {judges[0]}")
    elif not judges:
        print("  \u26a0\ufe0f  no rubric artifacts found \u2014 judge provenance unverified")
    else:
        print(f"  \u274c judge DIFFERS across conditions: {judges}")
        print("     Conditions with a non-matching judge are NOT comparable.")
    tot_trunc = int(df["trunc_at_ceiling"].sum()); tot_n = int(df["N"].sum())
    print(f"  generations at the {GEN_CEILING}-token ceiling: {tot_trunc}/{tot_n} "
          f"({100*tot_trunc/max(tot_n,1):.1f}%)")
    print(f"  extraction failures (unparseable decision): {int(df['empty_pred'].sum())}/{tot_n}")

    print(f"\nSaved → {OUT_DIR}/draft_benchmark_table.csv")
    print(f"Saved → {OUT_DIR}/draft_per_case_long.csv  (for McNemar / GLMM)")


if __name__ == "__main__":
    main()
