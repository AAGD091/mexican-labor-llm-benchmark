# Reproducibility guide

## 1. Inspect the saved study

Read the benchmark JSON, evaluation rubric, source manifest, aggregate CSV, and case-level CSV first. Confirm the 90 cases, 20 conditions, and 1800 observations. The verified source manifest contains 321 court PDFs and nine statute PDFs. Precedente30834.pdf is empty and was not indexed; September 24 additions are outside the study collection.

## 2. Recompute aggregate tables from checkpoints

The main notebook calls `crossmodel_v3_draft.py` with the results directory and a separate output directory. The script imports the sibling `norm.py` module and expects `gold_prediction_90_compatible.json` at the results-directory root. Preserve all 20 arm/model directories and their checkpoint filenames.

For the planned layout, arrange:

```text
src/crossmodel_v3_draft.py
src/norm.py
results/gold_prediction_90_compatible.json
results/BARE_LLM__<model>/checkpoint_bare_<model>.jsonl
results/RAG_LLM_NO_KG__<model>/checkpoint_ragonly_<model>.jsonl
results/GRAPH_RAG__<model>/checkpoint_graphrag_<model>.jsonl
results/RL_LLM__<model>/checkpoint_<model>__grounded-guard.jsonl
```

Dependencies found in the aggregation source and driver notebook include NumPy, pandas, scikit-learn, rouge_score, sacrebleu, and bert_score. An exact historical environment lockfile has not yet been verified. BERTScore can require additional model downloads and computational resources. Missing optional metric libraries can leave scores blank.

From the repository root, the intended command is:

```bash
python src/crossmodel_v3_draft.py results reproduced_tables
```

On 22 October 2026 this command was executed on the 20 saved checkpoints in a fresh Python virtual environment (numpy, pandas, scikit-learn, scipy, rouge_score, sacrebleu; BERTScore not installed). The decision accuracy, MCC, article-F1, empty-prediction and truncation columns matched Table 14 of the revised manuscript for all 20 conditions. Since version 1.1, a condition whose checkpoints lack `completion_tokens` (the five A4 conditions) reports truncation as not measured (empty value / n.m.) instead of 0; the printed total is 972/1,350 (72.0%) over the 15 measured conditions. The script's internal header still refers to v2, although the driver invokes the v3 filename. Retain this fact for traceability until code cleanup is reviewed.

The script uses 2000 bootstrap resamples and seed 42. It can process partial inputs, so successful execution alone does not demonstrate complete coverage. Inspect the reported coverage and verify 20 conditions of 90 cases before comparing outputs with saved tables.

## 3. Recompute the inferential analyses

`src/revision_reanalysis.py` reproduces the pairwise contrasts on decision correctness (pooled exact McNemar, Holm), the Friedman and Wilcoxon tests on article-F1 (Holm, Cliff's δ), and the one-sided binomial tests against the 54.44% majority baseline, and adds the case-clustered analyses, minimum detectable effects, TOST equivalence tests (±5 percentage points), truncation recount, batch analyses and A4 diagnostics.

```bash
python src/revision_reanalysis.py --long results/_benchmark_final/draft_per_case_long.csv \
  --results results --gold results/gold_prediction_90_compatible.json --out revision/outputs
```

Dependencies: numpy, pandas, scipy, scikit-learn, statsmodels. Bootstrap seed 42 (B = 5000). The mixed logistic model is fitted by maximum likelihood with Gauss–Hermite quadrature (100 nodes); the likelihood-ratio statistic is stable from 50 to 150 nodes. Outputs used in the revised manuscript are in `revision/outputs/`. File names follow the table numbers of the revised manuscript:

| File | Revised manuscript | Submitted manuscript |
|---|---|---|
| `table16_pooled_mcnemar.csv` | Table 16 | Table 15 |
| `table17_case_clustered_mde_tost.csv` | Table 17 (new) | — |
| `table18_pooled_wilcoxon_articleF1.csv` | Table 18 | Table 16 |
| `table18_case_level_articleF1.csv` | Section 6.3.2 (case as unit; added 27 Oct 2026) | — |
| `table23_accuracy_by_batch_class.csv` | Table 23 (new) | — |
| `truncation_by_condition.csv` | Section 4.4, Table 14 (Trunc.) | Section 4.4 |
| `revision_reanalysis.json` | all of the above, plus the Friedman test, binomial tests, mixed-model LR test and AIC (Section 6.3.3), batch separability (Section 6.7) and A4 diagnostics (Section 6.4) | — |

### Sensitivity analysis (cases whose text names the gold label)

```bash
python src/sensitivity_exclude_cases.py --exclude PRED-020,PRED-051 --long results/_benchmark_final/draft_per_case_long.csv \
  --results results --gold results/gold_prediction_90_compatible.json --out revision/outputs/sensitivity_excl_PRED020_051
```

`--exclude ""` reproduces `revision_reanalysis.json` (sections 1–3) exactly. Case count and majority baseline are computed from the remaining cases.

## 4. Rerun generation

Generation requires a separate environment setup. The reported system used Google Colab with an NVIDIA T4 16 GB accelerator, CUDA-enabled llama.cpp Python bindings, local GGUF candidate models, and a separate Phi judge. Model acquisition sources are described in the manuscript artifact manifest. Obtain models from their original publishers or distributors under their applicable terms; this repository does not redistribute weights.

Review and adapt notebook paths before execution. Restore `data/eval_json`, `data/leyes_mexicanas`, `data/scjn_resoluciones`, `lib`, `faiss_index`, `faiss_index_cbr`, and `legal_kg.graphml` under the notebook's expected SCJN_RAG directory. Verify frozen-index assertions rather than silently rebuilding the substrate.

The frozen dense index records multilingual-e5-large embeddings, 1024 dimensions, 25461 vectors, normalized inner-product retrieval, precedent chunk size/overlap 2000/200, statute chunk size/overlap 2400/300, and retrieval depth four.

Retained settings differ by arm: the single-pass generation ceiling is 512 tokens, while A4 uses 900–1000. Judge execution device and context windows also differ. Consult the manuscript and actual selected notebook configuration rather than applying one global setting to every arm.

The arm notebooks were matched to the reported runs on 21 October 2026 (see `manifests/run_provenance.csv`). Repository preparation has not rerun generation, validated historical environment versions, or established byte-for-byte source identity with the original index build.
