# Changelog

## [1.2] — 2026-11-05 (second revision, MDPI *Applied Sciences*, round 2 of review)

Reviewer 2 asked that the analyses with the case as the unit of analysis (n = 90) be the primary evidence and that the tests treating the 450 case–backbone pairs as independent be kept as supplementary comparisons. The saved study outputs (checkpoints, per-case table, Hugging Face data) are unchanged.

### Correction
- `src/revision_reanalysis.py`, `src/sensitivity_exclude_cases.py`: per-case means over the five backbones are rounded to 10 decimals before the Wilcoxon signed-rank tests (`ROUND = 10`). Without rounding, floating-point residues of about 1e-17 remained where two arms are equal for a case, and the test, which drops only exact zeros, counted them as differences. Values that changed: decision-correctness case-level contrasts (`table17_case_clustered_mde_tost.csv`): all six unadjusted Wilcoxon p-values, and A2→A3 pHolm 0.612 → 0.729; case-level article-F1 (`table18_case_level_articleF1.csv`): A1→A3 pHolm 5.66e-7 → 5.85e-7 and A2→A3 pHolm 0.972 → 0.950; sensitivity analysis (n = 88): unadjusted p-values and A2→A3 pHolm 0.436 → 0.428. All other changed values (means, differences, bootstrap limits, TOST p-values) differ by less than 1e-11. No conclusion changes.

### Additions
- `src/revision_reanalysis.py`: matched-pairs rank-biserial correlation for the case-level article-F1 contrasts (`rank_biserial`); Friedman test on the per-case means (`friedman_article_f1_case_level`: χ² = 129.24, p = 7.9e-28); case-level Spearman correlation between the A4 reward and own-gold article-F1 (`a4_diagnostics.spearman_reward_vs_own_gold_f1_case_level`: ρ = −0.008, p = 0.94, n = 90).
- `src/sensitivity_exclude_cases.py`: the same case-level article-F1 contrasts and Friedman test without PRED-020 and PRED-051 (n = 88; χ² = 124.83, p = 7.0e-27; A2→A3 pHolm = 0.950); new output `sensitivity_excl_PRED020_051/sens_case_level_articleF1.csv`.
- `src/figures/make_diagrams.py`: Figure 1 (stacked layout, larger text; Inference block lists the case-level analyses first and the pooled tests in Appendix E; C1/C2 annotated with case-level Holm p-values and rank-biserial correlations) and Figure 2 (larger text; box 6 lists the case-level analyses and "pooled tests: Appendix E").
- `src/figures/make_figure_a5.py`: panel (b) case-level Holm p-values; panel (g) matched-pairs rank-biserial correlation on case-mean article-F1 (n = 90) instead of Cliff's δ on the 450 pooled rows; panel (f) titled "all backbones"; smaller canvas for larger printed text.
- `revision/outputs/`: all outputs regenerated with the scripts of this version. Unchanged files are byte-identical; `docs/REPRODUCIBILITY.md` maps the file names to the round-2 table numbers.

## [1.1] — 2026-10-21 (revision in response to peer review, MDPI *Applied Sciences*)

### Amendment of 27 October 2026
- `src/revision_reanalysis.py` (reviewer R2-4): the article-F1 contrasts are also repeated with the case as the unit of analysis (per-case mean over the five backbones, n = 90; Wilcoxon signed-rank, Holm; case bootstrap B = 5000, seed 42, separate generator). New output `revision/outputs/table18_case_level_articleF1.csv` and JSON key `case_level_article_f1`. Re-executed: all previously released outputs are byte-identical.

### Amendment of 22 October 2026 (before release)
- `src/crossmodel_v3_draft.py` (reviewer R2-3): missing `completion_tokens` is no longer counted as a complete output. Conditions without token metadata (the five A4 conditions) now report truncation as not measured instead of 0. Re-executed on the 20 saved checkpoints: all other aggregate columns are unchanged and match Table 14 of the revised manuscript.
- `src/revision_reanalysis.py`: output files and JSON keys renamed to the table numbers of the revised manuscript (`table16_pooled_mcnemar`, `table17_case_clustered_mde_tost`, `table18_pooled_wilcoxon_articleF1`, `table23_accuracy_by_batch_class`). Re-executed: every value is identical to the 21 October outputs.
- Hugging Face aggregate table `draft_benchmark_table.csv`: `trunc_at_ceiling` of the five A4 (`RL+LLM`) rows changed from 0 to empty (not measured). No other cell changed.
- Added `src/figures/make_figure_a5.py` (Figure A5, computed from `draft_per_case_long.csv` and `draft_benchmark_table.csv`; A4 truncation shown as n.m.) and `src/figures/make_diagrams.py` (Figures 1 and 2 of the revised manuscript).
- Added `src/sensitivity_exclude_cases.py` and `revision/outputs/sensitivity_excl_PRED020_051/` (26 Oct): the case text of PRED-020 and PRED-051 names their own gold label, so the pooled McNemar, Friedman/Wilcoxon, binomial, mixed-model LR and case-clustered/MDE/TOST analyses were re-run without them (n = 88). With no case excluded the script reproduces `revision_reanalysis.json` exactly. Conclusions unchanged: no decision-correctness contrast significant; C2 article-F1 null (pHolm = 0.670); interaction LR = 28.21, p = 0.005.
- `manifests/run_provenance.csv`: `manuscript_table_row` now cites Table 14 of the revised manuscript (Table 13 in the submitted version).
- Added `aic_main` and `aic_full` to the likelihood-ratio results (AIC 1642.7 main effects vs 1638.5 interaction).

### Benchmark data (Hugging Face dataset)
- `benchmark/gold_prediction_90_compatible.json`: removed an internal planning note from `metadata.transform_notes`. The note said the 30 added cases (PRED-061–090) required expert validation and double annotation (κ ≥ 0.70) before publication. That step was not carried out. Added `metadata.annotation_provenance` describing who wrote the cases and how, and `metadata.revision_history`. **No case record was changed** (the 90 records are identical to version 1.0).

### Code and documentation (GitHub)
- Added `src/revision_reanalysis.py`: re-implements the submitted inferential analyses (submitted Tables 15–16 = revised Tables 16 and 18, Friedman test, one-sided binomial tests, likelihood-ratio test) and adds the analyses requested in review: case-clustered contrasts, minimum detectable effect, TOST equivalence (±5 percentage points), truncation recount, batch analyses, and A4 diagnostics. Outputs in `revision/outputs/`.
- Added `manifests/run_provenance.csv`: mapping of the 20 conditions to notebook, results folder, checkpoint (with SHA-256), generation budget, context window, judge and manuscript table row.
- `manifests/notebook_preparation.json`, `README.md`, `docs/REPRODUCIBILITY.md`, `docs/PROVENANCE.md`, `docs/AUTHOR_RELEASE_CHECKLIST.md`: updated provenance status of the four arm notebooks (now verified), annotation provenance, and how to run the reanalysis.

### Corrections documented (no change to saved results)
- Truncation at the 512-token ceiling: 972 of 1,350 prose-arm generations (72.0%); 972 of all 1,800 (54.0%). A4 token usage was not stored, so A4 truncation is not measured.
- GraphRAG context window was 4,096 tokens for both gemma2-9b-q4 and llama31-8b-q8 (other arms: 8,192; 6,144 for Q8_0).

## [1.0] — 2026-10-03
- Initial release accompanying the submitted manuscript (GitHub commit 9e46af8).
