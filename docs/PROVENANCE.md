# Source provenance and proposed release rights

## Verified corpus

The frozen dense index contains source paths for 321 SCJN resolutions and nine Mexican statutory PDFs. All 330 filenames are available locally. The court-page total is 8747, matching saved GraphRAG ingestion logs.

`Verified_Retrieval_Corpus_Manifest.csv` records source type, filename, size, and checksum of the current copy. It does not contain verified official download URLs or historical build-time source hashes. Add official source links and documented acquisition information before releasing source materials.

Thirty-five original PDFs dated September 24, 2026, and a duplicate copy were excluded at the author's direction. The empty Precedente30834.pdf is not in the indexed corpus. Preserve the exclusion record for audit, but do not include these files in the study's public corpus.

## Rights by material

| Material | Proposed treatment |
|---|---|
| Original research code | Apache-2.0 approved by authors |
| Original benchmark scenarios and annotations | CC BY 4.0 approved by authors; validation procedures still require documentation |
| Original documentation | CC BY 4.0 approved by authors |
| Generated outputs and evaluation results | Review release rights, embedded quotations, and model terms; document scope separately |
| SCJN resolutions and statutory texts | Third-party materials; verify their applicable terms and provide attribution and official sources |
| Retrieval indexes and graph | Derived assets; review embedded third-party text and applicable redistribution terms |
| Model weights and dependency packages | Excluded; refer users to original distributions and their licenses |

The authors approved the original-code and original-annotation/documentation licenses on October 3, 2026. Third-party rights have not been verified by this preparation process. A general repository license must explicitly state its scope. Public accessibility of a document alone does not establish an unrestricted redistribution license.

Before publication, review benchmark records, outputs, notebook saved cells, and document exports for credentials, private identifiers, personal information, and material that is outside the approved study. No comprehensive release-sensitivity review has yet been completed.

Do not upload the original bulk-download ZIP archives: they contain excluded files, notebook history, and compiled caches. Build release packages from an explicit reviewed file list.
