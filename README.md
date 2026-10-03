# Mexican Labor-Law LLM Benchmark

Controlled comparison of four language-model architectures across five quantized model configurations for statutory grounding and legal outcome prediction in Spanish-language Mexican labor law.

## Study and assets

- 90 benchmark cases: 49 Procedente, 26 Improcedente, and 15 Parcialmente Improcedente.
- Four arms: parametric-only generation, RAG, GraphRAG, and reward-guided best-of-5 selection with guarded revision.
- Five candidate configurations: Llama 3.1 8B Q4_K_M and Q8_0; Qwen2.5 7B, Mistral 7B Instruct v0.3, and Gemma 2 9B at Q4_K_M.
- Phi 3.5 Mini Instruct Q4_K_M is the independent automated judge.
- 20 conditions and 1800 case-level observations.
- Retrieval corpus: 321 SCJN resolutions and nine Mexican statutory documents, totaling 330 PDFs. The indexed court documents total 8747 pages.

This documentation was prepared from retained notebooks, checkpoints, source metadata, and saved outputs. It does not represent a fresh execution of the experiments.

## Planned repository layout

```text
README.md
docs/
  REPRODUCIBILITY.md
  PROVENANCE.md
src/
  crossmodel_v3_draft.py
  norm.py
  index_manifest.py
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
```

The listed arm notebooks are release candidates. Their exact correspondence to the final saved runs must be confirmed before publication. Earlier versions are retained in the working archive rather than presented as the primary workflow.

The companion Hugging Face dataset repository will hold benchmark data and saved results. Its permanent URL must be added here after creation. The repositories are not yet published.

## Getting started

Begin with `docs/REPRODUCIBILITY.md`. There are two distinct workflows: recomputing tables from saved checkpoints and rerunning model generation. The former does not require the candidate GGUF weights. Full generation requires the frozen assets, locally acquired models, and a compatible GPU environment.

Original scripts assume a Google Colab Drive directory at `/content/drive/MyDrive/SCJN_RAG`. These paths need adjustment for another environment. Do not run an entire notebook without reviewing its installation, mounting, cache, and file-writing cells.

## Interpretation

Statutory grounding and prediction are evaluated separately. A4 uses reward-based selection, not parameter-updating reinforcement learning. Its larger generation budget, exemplar context, and structured outputs limit causal attribution of cross-arm differences. Automated judge agreement with human experts has not been established in the reported study.

## Licensing and citation

The authors have approved Apache-2.0 for original code and CC BY 4.0 for author-owned benchmark annotations and documentation. See LICENSE and LICENSE-DATA.md. These proposals do not license third-party court texts, statutes, model weights, or dependencies. See `docs/PROVENANCE.md`. License notices are included; a complete citation record must be finalized before publication. No article DOI is currently established by this package.
