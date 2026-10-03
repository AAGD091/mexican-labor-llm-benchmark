# Author release checklist

## Completed

- Retrieved scripts, benchmark, rubric, checkpoints, tables, indexes, graph, and source PDFs.
- Excluded September 24 additions from the active study collection.
- Matched 321 court resolutions and nine statutes to the frozen index's 330 PDF filenames.
- Matched the court-page total of 8747 to saved notebook logs.
- Corrected the six corpus descriptions in the open manuscript without changing numerical results.
- Prepared GitHub README, Hugging Face dataset card, reproducibility guide, and provenance notes.

## Before publication

1. Confirm the candidate final arm notebooks match the saved runs. Identify any separate statistical-analysis or figure-generation scripts needed to reproduce the paper beyond its aggregate tables.
2. Confirm the completed expert-review procedure for all 90 benchmark cases, including what was actually done for double annotation and agreement assessment.
3. Review original code and annotation ownership, expert-contribution permissions, third-party source terms, and model terms. Then approve separate code/data license scopes.
4. Supply verified official source links and acquisition dates; check the manuscript's 2013–2014 court-date claim against the documents.
5. Review release candidates for sensitive data and remove credentials and irrelevant notebook output from copies prepared for publication.
6. Verify an environment specification and test the aggregation command in a clean environment. Do not describe the draft running guide as tested before that succeeds.
7. Assemble the GitHub code package and Hugging Face data/results package from the reviewed file list. Add actual links to both READMEs, citation information, and optionally an archival DOI.
8. Replace the manuscript's Data Availability AUTHOR ACTION with a factual statement only after repository access and release versions are verified.

The built-in LaTeX compiler failed with “Unable to find standard directories for platform” after the corpus edit. PDF compilation remains unverified; the source editor remains the working document.
