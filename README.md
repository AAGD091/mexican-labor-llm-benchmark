# Mexican Labor-Law LLM Benchmark

Controlled comparison of four language-model architectures across five quantized model configurations for statutory grounding and legal outcome prediction in Spanish-language Mexican labor law.

## Study and assets

- 90 benchmark cases: 49 Procedente, 26 Improcedente, and 15 Parcialmente Improcedente. The scenarios and outcome labels were written by two practising attorneys; each label is the attorneys' professional determination for a written scenario, not a recorded court ruling. Initial batch PRED-001–060; added batch PRED-061–090. No written annotation protocol or double annotation was used, and no agreement statistic exists.
- Four arms: parametric-only generation, RAG, GraphRAG, and reward-guided best-of-5 selection with guarded revision.
- Five candidate configurations: Llama 3.1 8B Q4_K_M and Q8_0; Qwen2.5 7B, Mistral 7B Instruct v0.3, and Gemma 2 9B at Q4_K_M.
- Phi 3.5 Mini Instruct Q4_K_M is the independent automated judge.
- 20 conditions and 1800 case-level observations.
- Retrieval corpus: 321 SCJN resolutions (2013–2014) and nine Mexican statutory documents, totaling 330 PDFs. The indexed court documents total 8747 pages.

This documentation was prepared from retained notebooks, checkpoints, source metadata, and saved outputs. It does not represent a fresh execution of the experiments.

## Repository layout

```text
README.md
CHANGELOG.md
LICENSE, LICENSE-DATA.md, NOTICE
requirements-aggregation.txt
docs/
  REPRODUCIBILITY.md
  PROVENANCE.md
  AUTHOR_RELEASE_CHECKLIST.md
src/
  crossmodel_v3_draft.py           aggregate tables (Table 14)
  norm.py                          citation normalizer
  index_manifest.py                frozen-index manifest checks
  revision_reanalysis.py           inferential analyses (Tables 16-18, 23; mixed-model LR test)
  sensitivity_exclude_cases.py     sensitivity analysis without PRED-020 and PRED-051 (Section 8.1)
  figures/
    make_diagrams.py               Figures 1 and 2
    make_figure_a5.py              Figure A5
revision/
  outputs/                         outputs of the two analysis scripts
notebooks/
  BARE_LLM_Legal_recomm_v1_gold90.ipynb
  RAG_LLM_Legal_recomm_90Golden_v2.ipynb
  Graph_RAG_LLM_Legal_90Golden_v3.ipynb
  RL_LLM_Legal_recomm_90Golden_v5.ipynb
  Full Models Benchmark_90Gold_v1.ipynb
  Freeze_Canonical_Index.ipynb
manifests/
  Verified_Retrieval_Corpus_Manifest.csv
  faiss_index_manifest.json
  notebook_preparation.json
  run_provenance.csv
  upload_file_list.json
```
The four arm notebooks are the notebooks that produced the reported runs (verified 21 October 2026; only the per-run `MODEL_KEY` setting changes between runs). `manifests/run_provenance.csv` maps each of the 20 conditions to its notebook, checkpoint file (with SHA-256), generation budget, context window and table row. Earlier versions are retained in the working archive rather than presented as the primary workflow.

The companion [Hugging Face dataset repository](https://huggingface.co/datasets/AAGD091/mexican-labor-llm-benchmark) publicly provides benchmark data and saved results. Both repositories were verified as accessible without signing in on 3 October 2026.

## Getting started

Begin with `docs/REPRODUCIBILITY.md`. There are three workflows: recomputing aggregate tables from saved checkpoints, recomputing the inferential analyses (`src/revision_reanalysis.py`, with the sensitivity variant `src/sensitivity_exclude_cases.py`), and rerunning model generation. Table and figure numbers refer to the revised manuscript (see `CHANGELOG.md`). The former does not require the candidate GGUF weights. Full generation requires the frozen assets, locally acquired models, and a compatible GPU environment.

Original scripts assume a Google Colab Drive directory at `/content/drive/MyDrive/SCJN_RAG`. These paths need adjustment for another environment. Do not run an entire notebook without reviewing its installation, mounting, cache, and file-writing cells.

## Interpretation

Statutory grounding and prediction are evaluated separately. A4 uses reward-based selection, not parameter-updating reinforcement learning. Its larger generation budget, exemplar context, and structured outputs limit causal attribution of cross-arm differences. Automated judge agreement with human experts has not been established in the reported study.

## Licensing and citation

The authors have approved Apache-2.0 for original code and CC BY 4.0 for author-owned benchmark annotations and documentation. See LICENSE and LICENSE-DATA.md. These licenses do not license third-party court texts, statutes, model weights, or dependencies. See `docs/PROVENANCE.md`. License notices are included; the complete citation record remains to be finalized. No article DOI is currently established by this package.

## Public release scope

The release excludes raw court and statutory PDFs, model weights, retrieval indexes, and the knowledge-graph artifact. Full model generation requires those additional assets. The aggregation and analysis scripts were re-executed on the 20 saved checkpoints in a fresh Python environment on 22 and 26 October 2026 and reproduce the reported tables; model generation was not re-executed.
