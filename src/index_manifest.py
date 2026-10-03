"""
Canonical FAISS index manifest — shared by RAG, GraphRAG and RL.

WHY THIS EXISTS
---------------
The three retrieval arms read and write ONE index path on Drive. Historically
each arm could rebuild it with its own parameters, and last-writer-wins applied
silently: RL built the index with multilingual-e5-small (384-dim, chunks
1200/800), GraphRAG later rebuilt it with e5-large (1024-dim, chunks
2000/2400). Runs completed normally in both cases, so the arms were compared
against DIFFERENT corpora with no error anywhere in the logs.

A dimension check alone is insufficient: two indexes can share 1024 dims and
still be built from different chunkings — same shape, different corpus.

DESIGN
------
The index is a FROZEN artifact described by a manifest written beside it. Every
arm asserts the full configuration at load time and HARD-FAILS on any mismatch.
Automatic rebuild is deliberately NOT offered: silent auto-rebuild is the
mechanism that produced the drift in the first place. A mismatch means a human
must decide which artifact is authoritative.
"""
import json, os, datetime

MANIFEST_NAME = "faiss_index_manifest.json"


def manifest_path(index_dir):
    return os.path.join(os.path.dirname(index_dir.rstrip("/")), MANIFEST_NAME)


def write_manifest(index_dir, *, embedder, dim, ntotal, chunk_prec, overlap_prec,
                   chunk_law, overlap_law, built_by, normalize=True):
    m = {
        "embedder": embedder,
        "dim": int(dim),
        "ntotal": int(ntotal),
        "chunk_size_prec": int(chunk_prec), "chunk_overlap_prec": int(overlap_prec),
        "chunk_size_law": int(chunk_law),   "chunk_overlap_law": int(overlap_law),
        "normalize_embeddings": bool(normalize),
        "built_by": built_by,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "frozen": True,
    }
    p = manifest_path(index_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return p, m


def assert_manifest(index_dir, vector_db, *, embedder, chunk_prec, overlap_prec,
                    chunk_law, overlap_law, arm):
    """Hard-fail unless the loaded index matches the frozen manifest exactly."""
    p = manifest_path(index_dir)
    if not os.path.exists(p):
        raise RuntimeError(
            f"[{arm}] No index manifest at {p}.\n"
            f"  The FAISS index is a frozen artifact shared by all retrieval "
            f"arms and must be described by a manifest before use.\n"
            f"  Run make_index_manifest.py once against the canonical index.")
    m = json.load(open(p, encoding="utf-8"))

    actual = {
        "embedder": embedder,
        "dim": int(vector_db.index.d),
        "ntotal": int(vector_db.index.ntotal),
        "chunk_size_prec": int(chunk_prec), "chunk_overlap_prec": int(overlap_prec),
        "chunk_size_law": int(chunk_law),   "chunk_overlap_law": int(overlap_law),
    }
    diffs = [(k, m.get(k), v) for k, v in actual.items() if m.get(k) != v]
    if diffs:
        lines = "\n".join(f"    {k:22s} manifest={a!r:42s} this run={b!r}"
                          for k, a, b in diffs)
        raise RuntimeError(
            f"[{arm}] FAISS index does NOT match the frozen manifest — HARD STOP.\n"
            f"{lines}\n"
            f"  Manifest: {p}\n"
            f"  The retrieval substrate must be IDENTICAL across arms or the "
            f"ablation compares architecture and corpus simultaneously.\n"
            f"  No automatic rebuild is performed: silent rebuilds are how the "
            f"arms diverged before. Decide which artifact is authoritative, "
            f"then re-run.")
    print(f"  ✅ [{arm}] index matches frozen manifest "
          f"(embedder={m['embedder']}, dim={m['dim']}, ntotal={m['ntotal']:,}, "
          f"chunks {m['chunk_size_prec']}/{m['chunk_overlap_prec']} + "
          f"{m['chunk_size_law']}/{m['chunk_overlap_law']})")
    return m

def ensure_manifest(index_dir, vector_db, *, embedder, chunk_prec, overlap_prec,
                    chunk_law, overlap_law, arm, embed_fn=None):
    """Assert the index against the frozen manifest, REGISTERING it on first use.

    Distinction that matters:
      * manifest ABSENT  -> first-time registration. The index is described,
        not modified. Permitted, and printed loudly.
      * manifest PRESENT and MISMATCHED -> hard failure, always. This is the
        drift case (an arm rebuilt the index with different parameters) and a
        human must decide which artifact is authoritative.

    Registration is gated on a live dimensionality probe: if `embed_fn` is
    supplied, the embedder must actually produce vectors of the index's width
    before anything is written. That prevents certifying an index which was
    built with a DIFFERENT embedder than the one declared here.

    KNOWN LIMITATION (state this in the paper): dimensionality and vector count
    are read from the index itself and are therefore verified. The CHUNK
    parameters cannot be recovered from a built index — they are recorded as
    DECLARED by the registering arm. All retrieval arms declare identical chunk
    values, so registration order does not change what is written, but a
    manifest registered against an index built under different chunking would
    record values that are internally consistent yet historically wrong.
    """
    p = manifest_path(index_dir)
    if os.path.exists(p):
        return assert_manifest(index_dir, vector_db, embedder=embedder,
                               chunk_prec=chunk_prec, overlap_prec=overlap_prec,
                               chunk_law=chunk_law, overlap_law=overlap_law, arm=arm)

    print(f"  ℹ️  [{arm}] no index manifest found at {p} — first use, registering.")
    if embed_fn is not None:
        probe = len(embed_fn("dim probe"))
        if probe != int(vector_db.index.d):
            raise RuntimeError(
                f"[{arm}] REFUSING to register manifest: index dim="
                f"{vector_db.index.d} but declared embedder '{embedder}' "
                f"produces dim={probe}.\n"
                f"  The index on disk was NOT built with this embedder. "
                f"Registering would freeze a false description of the "
                f"retrieval substrate.")
    _p, m = write_manifest(
        index_dir, embedder=embedder, dim=vector_db.index.d,
        ntotal=vector_db.index.ntotal, chunk_prec=chunk_prec,
        overlap_prec=overlap_prec, chunk_law=chunk_law,
        overlap_law=overlap_law, built_by=arm)
    print(f"  ✅ [{arm}] index manifest registered → {_p}")
    for k, v in m.items():
        print(f"        {k:22s} {v}")
    print(f"  Every subsequent run of ANY arm now asserts against this manifest "
          f"and hard-fails on divergence.")
    return m
