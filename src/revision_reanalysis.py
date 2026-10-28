#!/usr/bin/env python3
"""
revision_reanalysis.py — inferential and diagnostic analyses for the MDPI
Applied Sciences revision (reviewer comments R1-3, R1-4, R2-2, R2-3, R2-4).

Inputs (all saved study artifacts; no model is run):
  --long      draft_per_case_long.csv  (arm, model, case_id, gold, pred, correct, article_f1)
  --results   results/ directory with the 20 checkpoint JSONL files
  --gold      gold_prediction_90_compatible.json
  --out       output directory

Sections
  1. Re-implementation of the submitted inferential analyses
     (revised manuscript Table 16 [submitted Table 15]: pooled exact McNemar +
     Holm; Table 18 [submitted Table 16]: Friedman + Wilcoxon + Cliff's delta +
     Holm; mixed logistic LR test; one-sided binomial tests).

Output files are named after the table numbers of the REVISED manuscript:
  table16_pooled_mcnemar.csv, table17_case_clustered_mde_tost.csv,
  table18_pooled_wilcoxon_articleF1.csv, table18_case_level_articleF1.csv,
  table23_accuracy_by_batch_class.csv,
  truncation_by_condition.csv, revision_reanalysis.json.
  2. Case-clustered reanalysis (unit = case; mean over 5 backbones).
  3. Minimum detectable effect and TOST equivalence (margin +/-5 pp).
  4. Truncation recount from token metadata (A1-A3); A4 token metadata check.
  5. Batch analyses (PRED-001-060 vs PRED-061-090).
  6. A4 diagnostics (revision acceptance, discarded rollouts, reward vs
     own-gold article-F1).

Seeds: bootstrap 42 (B = 5000 for new analyses). Python >= 3.10; numpy,
pandas, scipy, scikit-learn.
"""
import argparse, glob, json, math, os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import expit, logsumexp

ARMS = ["Bare LLM", "RAG+LLM", "GraphRAG+LLM", "RL+LLM"]
SHORT = {"Bare LLM": "A1", "RAG+LLM": "A2", "GraphRAG+LLM": "A3", "RL+LLM": "A4"}
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
CKPT = {"Bare LLM": "checkpoint_bare_{m}.jsonl", "RAG+LLM": "checkpoint_ragonly_{m}.jsonl",
        "GraphRAG+LLM": "checkpoint_graphrag_{m}.jsonl", "RL+LLM": "checkpoint_{m}__grounded-guard.jsonl"}
GEN_CEILING = 512
MAJORITY = 49 / 90
MARGIN = 0.05
B = 5000
SEED = 42


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def find(results, name):
    hits = glob.glob(os.path.join(results, "**", name), recursive=True)
    hits += glob.glob(os.path.join(results, "*" + name))
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits)[0]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- GLMM (LR test)
def glmm_negloglik(theta, X, y, groups, nodes, weights):
    beta, log_s = theta[:-1], theta[-1]
    s = math.exp(log_s); eta = X @ beta
    ll = 0.0
    for idx in groups:
        e = eta[idx][:, None] + math.sqrt(2) * s * nodes[None, :]
        yy = y[idx][:, None]
        lp = (yy * -np.logaddexp(0, -e) + (1 - yy) * -np.logaddexp(0, e)).sum(0)
        ll += logsumexp(lp + np.log(weights)) - 0.5 * math.log(math.pi)
    return -ll


def fit_glmm(X, y, groups, q=100):
    nodes, weights = np.polynomial.hermite.hermgauss(q)
    th0 = np.zeros(X.shape[1] + 1)
    r = optimize.minimize(glmm_negloglik, th0, args=(X, y, groups, nodes, weights),
                          method="L-BFGS-B", options={"maxiter": 2000})
    return -r.fun, r


def design(df, interaction):
    A = pd.get_dummies(pd.Categorical(df["arm"], categories=ARMS), drop_first=True).astype(float)
    models = sorted(df["model"].unique())
    M = pd.get_dummies(pd.Categorical(df["model"], categories=models), drop_first=True).astype(float)
    cols = [np.ones(len(df))] + [A[c].values for c in A] + [M[c].values for c in M]
    if interaction:
        cols += [A[a].values * M[m].values for a in A for m in M]
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--gold", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--skip_glmm", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    R = {}

    df = pd.read_csv(a.long)
    models = sorted(df["model"].unique()); cases = sorted(df["case_id"].unique())
    assert len(df) == 1800 and df.groupby(["arm", "model"]).size().eq(90).all()
    C = {(r.arm, r.model, r.case_id): int(r.correct) for r in df.itertuples()}
    F = {(r.arm, r.model, r.case_id): float(r.article_f1) for r in df.itertuples()}

    # ---------------------------------------------------------------- 1. submitted analyses
    t15 = []
    for i, j in PAIRS:
        x, z = ARMS[i], ARMS[j]
        a_ok = sum(C[(x, m, c)] for m in models for c in cases); b_ok = sum(C[(z, m, c)] for m in models for c in cases)
        ab = sum(1 for m in models for c in cases if C[(x, m, c)] == 1 and C[(z, m, c)] == 0)
        ba = sum(1 for m in models for c in cases if C[(x, m, c)] == 0 and C[(z, m, c)] == 1)
        p = stats.binomtest(ab, ab + ba, 0.5).pvalue if ab + ba else 1.0
        t15.append(dict(contrast=f"{SHORT[x]} vs {SHORT[z]}", A_correct=a_ok, B_correct=b_ok,
                        A_gt_B=ab, B_gt_A=ba, OR=ab / ba if ba else float("inf"), p=p))
    for r, ph in zip(t15, holm([r["p"] for r in t15])): r["p_holm"] = ph
    R["table16_pooled_mcnemar"] = t15

    f1mat = np.array([[F[(arm, m, c)] for arm in ARMS] for m in models for c in cases])  # 450 x 4
    fr = stats.friedmanchisquare(*f1mat.T)
    t16 = []
    for i, j in PAIRS:
        d = f1mat[:, i] - f1mat[:, j]
        p = stats.wilcoxon(f1mat[:, i], f1mat[:, j]).pvalue if np.any(d != 0) else 1.0
        t16.append(dict(contrast=f"{SHORT[ARMS[i]]} vs {SHORT[ARMS[j]]}", mean_A=f1mat[:, i].mean(),
                        mean_B=f1mat[:, j].mean(), delta=d.mean(),
                        cliffs_delta=cliffs_delta(f1mat[:, i], f1mat[:, j]), p=p))
    for r, ph in zip(t16, holm([r["p"] for r in t16])): r["p_holm"] = ph
    R["table18_pooled_wilcoxon_articleF1"] = t16
    R["friedman_article_f1"] = dict(chi2=fr.statistic, p=fr.pvalue, n=450)

    R["binomial_vs_majority"] = [dict(arm=SHORT[arm], model=m, correct=int(k), p_one_sided=stats.binomtest(int(k), 90, MAJORITY, alternative="greater").pvalue)
                                 for arm in ARMS for m in models for k in [sum(C[(arm, m, c)] for c in cases)]]

    if not a.skip_glmm:
        y = df["correct"].astype(float).values
        cid = df["case_id"].values; groups = [np.where(cid == c)[0] for c in cases]
        ll0, _ = fit_glmm(design(df, False), y, groups)
        ll1, _ = fit_glmm(design(df, True), y, groups)
        lr = 2 * (ll1 - ll0)
        # Same LR test WITHOUT the case random intercept (ordinary logistic regression):
        # reproduces the value reported in the submitted manuscript (LR = 14.24, p = 0.286).
        import statsmodels.api as sm
        g0 = sm.GLM(y, design(df, False), family=sm.families.Binomial()).fit()
        g1 = sm.GLM(y, design(df, True), family=sm.families.Binomial()).fit()
        lr_nore = 2 * (g1.llf - g0.llf)
        R["logistic_lr_test_without_case_effect"] = dict(LR=lr_nore, df=12, p=stats.chi2.sf(lr_nore, 12),
                                                         aic_main=g0.aic, aic_full=g1.aic)
        k0 = design(df, False).shape[1] + 1; k1 = design(df, True).shape[1] + 1  # + case SD
        R["glmm_lr_test"] = dict(LR=lr, df=12, p=stats.chi2.sf(lr, 12), ll_main=ll0, ll_full=ll1,
                                 aic_main=2 * k0 - 2 * ll0, aic_full=2 * k1 - 2 * ll1,
                                 method="logistic GLMM, case random intercept, non-adaptive Gauss-Hermite (100 nodes), ML; stable from 50 to 150 nodes")

    # ---------------------------------------------------------------- 2./3. case-clustered, MDE, TOST
    def case_means(arm, ids=cases):
        return np.array([np.mean([C[(arm, m, c)] for m in models]) for c in ids])
    cc = []
    for i, j in PAIRS:
        d = case_means(ARMS[j]) - case_means(ARMS[i]); n = len(d); sd = d.std(ddof=1); se = sd / math.sqrt(n)
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        bs = d[rng.integers(0, n, (B, n))].mean(1)
        tcrit = stats.t.ppf(0.975, n - 1); mde = (tcrit + stats.t.ppf(0.80, n - 1)) * se
        t_lo = (d.mean() + MARGIN) / se; t_hi = (d.mean() - MARGIN) / se
        p_tost = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
        ci90 = (d.mean() - stats.t.ppf(0.95, n - 1) * se, d.mean() + stats.t.ppf(0.95, n - 1) * se)
        cc.append(dict(contrast=f"{SHORT[ARMS[i]]}->{SHORT[ARMS[j]]}", n_cases=n, delta_pp=100 * d.mean(),
                       boot95_lo=100 * np.percentile(bs, 2.5), boot95_hi=100 * np.percentile(bs, 97.5),
                       wilcoxon_p=w, sd_pp=100 * sd, mde80_pp=100 * mde,
                       tost_p=p_tost, ci90_lo_pp=100 * ci90[0], ci90_hi_pp=100 * ci90[1],
                       equivalent_within_5pp=bool(p_tost < 0.05)))
    for r, ph in zip(cc, holm([r["wilcoxon_p"] for r in cc])): r["wilcoxon_p_holm"] = ph
    for r, ph in zip(cc, holm([r["tost_p"] for r in cc])): r["tost_p_holm"] = ph
    R["case_clustered"] = cc

    # 2b. Article-F1 contrasts with the case as unit (added 27 Oct 2026 for R2-4):
    # per-case mean article-F1 over the 5 backbones (n = 90), Wilcoxon signed-rank + Holm,
    # case bootstrap. Separate generator so that all other outputs are unchanged.
    rng_f1 = np.random.default_rng(SEED)
    def case_f1(arm):
        return np.array([np.mean([F[(arm, m, c)] for m in models]) for c in cases])
    cf = []
    for i, j in PAIRS:
        d = case_f1(ARMS[j]) - case_f1(ARMS[i]); n = len(d)
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        bs = d[rng_f1.integers(0, n, (B, n))].mean(1)
        cf.append(dict(contrast=f"{SHORT[ARMS[i]]}->{SHORT[ARMS[j]]}", n_cases=n,
                       mean_A=case_f1(ARMS[i]).mean(), mean_B=case_f1(ARMS[j]).mean(), delta=d.mean(),
                       boot95_lo=np.percentile(bs, 2.5), boot95_hi=np.percentile(bs, 97.5), wilcoxon_p=w))
    for r, ph in zip(cf, holm([r["wilcoxon_p"] for r in cf])): r["wilcoxon_p_holm"] = ph
    R["case_level_article_f1"] = cf

    # ---------------------------------------------------------------- 4. truncation and token metadata
    trunc = []; acc_tr = Counter(); n_tr = Counter()
    for arm in ARMS:
        for m in models:
            rows = read_jsonl(find(a.results, CKPT[arm].format(m=m)))
            assert len(rows) == 90 and len({(r.get("row") or r).get("case_id") or r.get("case_id") for r in rows}) == 90
            rr = [(r.get("row") or r) for r in rows]
            ct = [x.get("completion_tokens") for x in rr]
            has = sum(v is not None for v in ct)
            k = sum(1 for v in ct if v == GEN_CEILING)
            pt_zero = sum(1 for x in rr if not x.get("prompt_tokens"))
            trunc.append(dict(arm=SHORT[arm], model=m, rows=len(rr), completion_tokens_present=has,
                              at_ceiling=k if has else None, prompt_tokens_missing_or_zero=pt_zero))
            if has:
                for x in rr:
                    t = x["completion_tokens"] == GEN_CEILING
                    n_tr[t] += 1; acc_tr[t] += int(x["correct_decision"])
    tdf = pd.DataFrame(trunc); tdf.to_csv(os.path.join(a.out, "truncation_by_condition.csv"), index=False)
    prose = tdf[tdf.arm != "A4"]
    R["truncation"] = dict(total_at_ceiling=int(prose.at_ceiling.sum()), prose_records=int(prose.rows.sum()),
                           share_of_prose=float(prose.at_ceiling.sum() / prose.rows.sum()),
                           share_of_all_1800=float(prose.at_ceiling.sum() / 1800),
                           by_arm={k: int(v) for k, v in prose.groupby("arm").at_ceiling.sum().items()},
                           a4_completion_tokens_present=int(tdf[tdf.arm == "A4"].completion_tokens_present.sum()),
                           accuracy_truncated=acc_tr[True] / n_tr[True], n_truncated=n_tr[True],
                           accuracy_not_truncated=acc_tr[False] / n_tr[False], n_not_truncated=n_tr[False])

    # ---------------------------------------------------------------- 5. batches
    base = [c for c in cases if int(c.split("-")[1]) <= 60]; ext = [c for c in cases if int(c.split("-")[1]) > 60]
    gold = json.load(open(a.gold, encoding="utf-8"))
    gcases = {c["case_id"]: c for c in gold["cases"]}
    R["batch_composition"] = {name: dict(Counter(gcases[c]["gold_standard"]["decision"] for c in ids))
                              for name, ids in [("PRED-001-060", base), ("PRED-061-090", ext)]}
    bt = []
    for name, ids in [("PRED-001-060", base), ("PRED-061-090", ext)]:
        for arm in ARMS:
            bt.append(dict(batch=name, arm=SHORT[arm], accuracy=float(case_means(arm, ids).mean())))
            for cls in ["Procedente", "Improcedente", "Parcialmente Improcedente"]:
                sub = [c for c in ids if gcases[c]["gold_standard"]["decision"] == cls]
                if sub:
                    bt.append(dict(batch=name, arm=SHORT[arm], gold_class=cls, n_cases=len(sub),
                                   accuracy=float(case_means(arm, sub).mean())))
    pd.DataFrame(bt).to_csv(os.path.join(a.out, "table23_accuracy_by_batch_class.csv"), index=False)
    wb = []
    for name, ids in [("PRED-001-060", base), ("PRED-061-090", ext)]:
        for i, j in PAIRS:
            d = case_means(ARMS[j], ids) - case_means(ARMS[i], ids)
            bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            wb.append(dict(batch=name, contrast=f"{SHORT[ARMS[i]]}->{SHORT[ARMS[j]]}", delta_pp=100 * d.mean(),
                           boot95_lo=100 * np.percentile(bs, 2.5), boot95_hi=100 * np.percentile(bs, 97.5)))
    R["within_batch_contrasts_unadjusted"] = wb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    yb = np.array([int(c["case_id"].split("-")[1]) > 60 for c in gold["cases"]]).astype(int)
    auc = {}
    for fld in ["input_summary", "facts"]:
        X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([c[fld] for c in gold["cases"]])
        p = cross_val_predict(LogisticRegression(max_iter=2000), X, yb,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        auc[fld] = roc_auc_score(yb, p)
    R["batch_text_separability_auc"] = auc
    R["summary_words_mean"] = {name: float(np.mean([len(gcases[c]["input_summary"].split()) for c in ids]))
                               for name, ids in [("PRED-001-060", base), ("PRED-061-090", ext)]}

    # ---------------------------------------------------------------- 6. A4 diagnostics
    acc, disc, rew, f1 = 0, 0, [], []
    per = {}
    for m in models:
        rows = read_jsonl(find(a.results, CKPT["RL+LLM"].format(m=m)))
        ap_ = sum(r["refinement_applied"] for r in rows); dc = sum(5 - r["n_rollouts"] for r in rows)
        per[m] = dict(revision_accepted=ap_, discarded_rollouts=dc)
        acc += ap_; disc += dc
        for r in rows:
            rew.append(round(r["reward_accepted"], 10)); f1.append(round(F[("RL+LLM", m, r["case_id"])], 10))  # round: float noise (0.5 vs 0.5000000000000001) must not break ties
    rho = stats.spearmanr(rew, f1)
    R["a4_diagnostics"] = dict(revision_accepted=acc, of=450, discarded_rollouts=disc, of_rollouts=2250,
                               per_model=per, spearman_reward_vs_own_gold_f1=dict(rho=rho.statistic, p=rho.pvalue, n=len(rew)))

    with open(os.path.join(a.out, "revision_reanalysis.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=float, ensure_ascii=False)
    pd.DataFrame(t15).to_csv(os.path.join(a.out, "table16_pooled_mcnemar.csv"), index=False)
    pd.DataFrame(t16).to_csv(os.path.join(a.out, "table18_pooled_wilcoxon_articleF1.csv"), index=False)
    pd.DataFrame(cc).to_csv(os.path.join(a.out, "table17_case_clustered_mde_tost.csv"), index=False)
    pd.DataFrame(R["case_level_article_f1"]).to_csv(os.path.join(a.out, "table18_case_level_articleF1.csv"), index=False)
    print(json.dumps(R, indent=1, default=float)[:200], "...\nwritten to", a.out)


if __name__ == "__main__":
    main()
