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

This command is documented from the retained driver; it has not been executed as a clean-environment reproduction during repository preparation. The script's internal header still refers to v2, although the driver invokes the v3 filename. Retain this fact for traceability until code cleanup is reviewed.

The script uses 2000 bootstrap resamples and seed 42. It can process partial inputs, so successful execution alone does not demonstrate complete coverage. Inspect the reported coverage and verify 20 conditions of 90 cases before comparing outputs with saved tables.

## 3. Rerun generation

Generation requires a separate environment setup. The reported system used Google Colab with an NVIDIA T4 16 GB accelerator, CUDA-enabled llama.cpp Python bindings, local GGUF candidate models, and a separate Phi judge. Model acquisition sources are described in the manuscript artifact manifest. Obtain models from their original publishers or distributors under their applicable terms; this repository does not redistribute weights.

Review and adapt notebook paths before execution. Restore `data/eval_json`, `data/leyes_mexicanas`, `data/scjn_resoluciones`, `lib`, `faiss_index`, `faiss_index_cbr`, and `legal_kg.graphml` under the notebook's expected SCJN_RAG directory. Verify frozen-index assertions rather than silently rebuilding the substrate.

The frozen dense index records multilingual-e5-large embeddings, 1024 dimensions, 25461 vectors, normalized inner-product retrieval, precedent chunk size/overlap 2000/200, statute chunk size/overlap 2400/300, and retrieval depth four.

Retained settings differ by arm: the single-pass generation ceiling is 512 tokens, while A4 uses 900–1000. Judge execution device and context windows also differ. Consult the manuscript and actual selected notebook configuration rather than applying one global setting to every arm.

Final arm notebooks must be matched to checkpoint provenance before this workflow can be called fully validated. Repository preparation has not rerun generation, validated historical environment versions, or established byte-for-byte source identity with the original index build.
