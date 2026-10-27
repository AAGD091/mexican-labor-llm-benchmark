# Author release checklist

## Completed

- Retrieved scripts, benchmark, rubric, checkpoints, tables, indexes, graph, and source PDFs.
- Excluded September 24 additions from the active study collection.
- Matched 321 court resolutions and nine statutes to the frozen index's 330 PDF filenames.
- Matched the court-page total of 8747 to saved notebook logs.
- Corrected the six corpus descriptions in the open manuscript without changing numerical results.
- Prepared GitHub README, Hugging Face dataset card, reproducibility guide, and provenance notes.

## Before publication

1. DONE 2026-10-21: arm notebooks confirmed as the reported runs (`manifests/run_provenance.csv`). Inferential analyses re-implemented in `src/revision_reanalysis.py`.
2. DONE 2026-10-21: documented — two attorneys wrote all cases; no written protocol, double annotation or agreement statistic (dataset `annotation_provenance`).
3. Review original code and annotation ownership, expert-contribution permissions, third-party source terms, and model terms. Then approve separate code/data license scopes.
4. Supply verified official source links and acquisition dates; check the manuscript's 2013–2014 court-date claim against the documents.
5. Review release candidates for sensitive data and remove credentials and irrelevant notebook output from copies prepared for publication.
6. PARTLY DONE 2026-10-22: the aggregation command was run on the 20 saved checkpoints in a fresh Python virtual environment without BERTScore; all other Table 14 columns matched (see `docs/REPRODUCIBILITY.md`). An exact historical environment lockfile is still not available.
7. Assemble the GitHub code package and Hugging Face data/results package from the reviewed file list. Add actual links to both READMEs, citation information, and optionally an archival DOI.
8. Replace the manuscript's Data Availability AUTHOR ACTION with a factual statement only after repository access and release versions are verified.

The revised manuscript compiles with pdfLaTeX and BibTeX (MDPI template) without errors (verified 2026-10-22).
