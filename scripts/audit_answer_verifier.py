"""End-to-end audit of the answer / citation / verifier layer.

Usage:
    python scripts/audit_answer_verifier.py [--suite answer_e2e] [--output PATH] [--dry-run]

Dry-run mode replaces real LLM calls with a synthetic response so the script
can run in CI without an LLM server.  The verification pipeline still executes
in full; the LLM output is just substituted.

Exit codes:
    0 — all entries PASS or WARN
    1 — at least one entry FAIL
    2 — unrecoverable runtime error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is importable when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Query suites
# ---------------------------------------------------------------------------

QUERY_SUITES: dict[str, list[dict]] = {
    "answer_e2e": [
        # --- Easy / direct answer ----------------------------------------
        {
            "id": "e01",
            "text": "What does SDMX stand for?",
            "category": "easy_direct",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        {
            "id": "e02",
            "text": "What is a DataFlow in SDMX?",
            "category": "easy_direct",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        {
            "id": "e03",
            "text": "What is a Key Family in SDMX?",
            "category": "easy_direct",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        # --- Technical SDMX ----------------------------------------------
        {
            "id": "t01",
            "text": "What is the difference between a Data Structure Definition and a DataFlow in SDMX 2.1?",
            "category": "technical_sdmx",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        {
            "id": "t02",
            "text": "How does SDMX represent time series data, and what role does the TIME_PERIOD dimension play?",
            "category": "technical_sdmx",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        {
            "id": "t03",
            "text": "What are the mandatory structural components of a Concept Scheme in SDMX?",
            "category": "technical_sdmx",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        # --- Spanish queries ---------------------------------------------
        {
            "id": "s01",
            "text": "¿Qué es SDMX y para qué sirve en el intercambio de datos estadísticos?",
            "category": "spanish",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        {
            "id": "s02",
            "text": "¿Cómo se define una estructura de datos en SDMX y qué elementos la componen?",
            "category": "spanish",
            "expect_citations": True,
            "expect_in_corpus": True,
        },
        # --- Ambiguous ---------------------------------------------------
        {
            "id": "a01",
            "text": "What is a key?",
            "category": "ambiguous",
            "expect_citations": True,
            "expect_in_corpus": True,
            # In SDMX context a "key" is a specific concept; the system should handle
            # the ambiguity and still produce cited SDMX-related content.
        },
        # --- Out of corpus -----------------------------------------------
        {
            "id": "o01",
            "text": "What is the capital of France?",
            "category": "out_of_corpus",
            "expect_citations": False,
            "expect_in_corpus": False,
            # The system should explicitly say the information is not in the docs.
        },
    ],
}

# Synthetic response used in --dry-run mode.  It contains one valid citation so
# the verifier has something to check.
_DRY_RUN_RESPONSE = (
    "SDMX stands for Statistical Data and Metadata eXchange. "
    "[[1] Fuente: SDMX-Training-introduction-2015 | Sección: What is SDMX | Líneas: 18-21]"
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    id: str
    category: str
    query: str
    expect_citations: bool
    expect_in_corpus: bool
    response: str
    citation_results: list[dict]
    warnings: list[str]
    verification_block: str
    evidence_map: dict
    confidence_score: float
    confidence_level: str
    consistency_parse_success: bool
    elapsed_s: float
    verdict: str           # PASS | WARN | FAIL
    verdict_reason: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Assessment logic
# ---------------------------------------------------------------------------

def assess(entry_def: dict, response: str, vr: Any) -> tuple[str, str]:
    """Return (verdict, reason) given query definition and VerificationResult."""
    category = entry_def["category"]
    expect_citations = entry_def["expect_citations"]
    expect_in_corpus = entry_def["expect_in_corpus"]

    sr = vr.score_result
    cr_list = vr.citation_results
    n_total = len(cr_list)
    n_valid = sum(1 for c in cr_list if c.status.value == "VALID")

    # Out-of-corpus query: system should say it can't find the info
    if not expect_in_corpus:
        no_info_markers = [
            "no encuentro",
            "no find",
            "not find",
            "not in the documents",
            "no está en los documentos",
            "información no está",
        ]
        lower = response.lower()
        if any(m in lower for m in no_info_markers):
            return "PASS", "System correctly declared information not in corpus."
        if n_total == 0:
            return "WARN", "No citations — system may have speculated without corpus basis."
        return "WARN", f"Out-of-corpus query but system produced {n_total} citations."

    # Hallucination detected by consistency check
    if (vr.consistency_result.parse_success
            and vr.consistency_result.has_hallucinations):
        return "FAIL", "Consistency check detected hallucinations."

    # Zero citations when citations are expected
    if expect_citations and n_total == 0:
        return "FAIL", "Response has no citations — factual claims are untraced."

    # All citations are INVALID
    if expect_citations and n_total > 0 and n_valid == 0:
        return "FAIL", f"All {n_total} citation(s) are INVALID — response is unverifiable."

    # Confidence LOW
    if sr.confidence_level.value == "LOW":
        return "WARN", f"Confidence LOW (score={sr.final_score:.2f})."

    # Ambiguous queries — relax: WARN is fine, PASS if citations present
    if category == "ambiguous":
        if n_valid >= 1:
            return "PASS", f"Ambiguous query resolved with {n_valid}/{n_total} valid citations."
        return "WARN", "Ambiguous query — citations present but none fully VALID."

    # Good path
    if n_valid == n_total and n_total > 0:
        return "PASS", f"All {n_total} citation(s) VALID, confidence {sr.confidence_level.value}."

    partial_or_invalid = n_total - n_valid
    return "WARN", (
        f"{n_valid}/{n_total} citations VALID, "
        f"{partial_or_invalid} PARTIAL/INVALID, "
        f"confidence {sr.confidence_level.value}."
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_single(query_def: dict, dry_run: bool, top_k: int = 5) -> AuditEntry:
    """Run the full retrieval→generation→verification pipeline for one query."""
    from rag_lab.config import (
        RERANK_TOP_K,
        EMBEDDING_DEVICE,
        ENABLE_CONSISTENCY_CHECK,
    )
    from rag_lab.storage.vector_store import VectorStore
    from rag_lab.storage.fts_store import FTSStore
    from rag_lab.storage.docstore import DocStore
    from rag_lab.embedding.encoder import encode_chunks
    from rag_lab.retrieval.hybrid_search import hybrid_search
    from rag_lab.retrieval.reranker import rerank
    from rag_lab.generation.prompt_builder import build_prompt
    from rag_lab.generation.llm_client import generate_response
    from rag_lab.verification.pipeline import verify_and_score
    from rag_lab.exceptions import LLMConnectionError

    query_text = query_def["text"]
    error: Optional[str] = None
    t0 = time.monotonic()

    try:
        # Stores (shared initialization is cheap — each call opens its own conn)
        vs = VectorStore()
        fts = FTSStore()
        ds = DocStore()
        vs.initialize()
        fts.initialize()
        ds.initialize()

        # Embed query
        device = os.getenv("EMBEDDING_DEVICE", EMBEDDING_DEVICE)
        dense_emb, sparse_dict = encode_chunks(
            [{"text": query_text}], batch_size=1, device=device
        )
        query_dense = dense_emb[0]
        query_sparse = next(iter(sparse_dict.values()), {}) if sparse_dict else {}

        # Retrieve
        results = hybrid_search(
            query_text, vs, ds, fts,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k * 2,
        )
        if not results:
            raise RuntimeError("No retrieval results — corpus may be empty.")

        # Rerank
        results = rerank(query_text, results[:20], top_k=top_k * 2, device=device)

        top_chunks = results[:RERANK_TOP_K]
        retrieval_scores = [
            r.get("rerank_score", r.get("score", 0.5)) for r in top_chunks
        ]

        # Generate (or use dry-run stub)
        if dry_run:
            response = _DRY_RUN_RESPONSE
        else:
            system_prompt, user_prompt = build_prompt(query_text, top_chunks)
            response = generate_response(system_prompt, user_prompt)
            if not response:
                raise RuntimeError("LLM returned empty response.")

        # Verify
        vr = verify_and_score(
            response,
            top_chunks,
            retrieval_scores,
            enable_consistency_check=(not dry_run) and ENABLE_CONSISTENCY_CHECK,
        )

        elapsed = time.monotonic() - t0
        verdict, reason = assess(query_def, response, vr)

        return AuditEntry(
            id=query_def["id"],
            category=query_def["category"],
            query=query_text,
            expect_citations=query_def["expect_citations"],
            expect_in_corpus=query_def["expect_in_corpus"],
            response=response,
            citation_results=[
                {
                    "citation_text": c.citation_text,
                    "status": c.status.value,
                    "chunk_id": c.chunk_id,
                }
                for c in vr.citation_results
            ],
            warnings=vr.get_warnings(),
            verification_block=vr.format_verification_block(verbose=True),
            evidence_map={str(k): v for k, v in vr.evidence_map.items()},
            confidence_score=vr.score_result.final_score,
            confidence_level=vr.score_result.confidence_level.value,
            consistency_parse_success=vr.consistency_result.parse_success,
            elapsed_s=round(elapsed, 2),
            verdict=verdict,
            verdict_reason=reason,
        )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        error_msg = f"{type(exc).__name__}: {exc}"
        logging.getLogger("rag_lab").error("Audit query %s failed: %s", query_def["id"], error_msg)
        return AuditEntry(
            id=query_def["id"],
            category=query_def["category"],
            query=query_text,
            expect_citations=query_def["expect_citations"],
            expect_in_corpus=query_def["expect_in_corpus"],
            response="",
            citation_results=[],
            warnings=[],
            verification_block="",
            evidence_map={},
            confidence_score=0.0,
            confidence_level="LOW",
            consistency_parse_success=False,
            elapsed_s=round(elapsed, 2),
            verdict="FAIL",
            verdict_reason=f"Pipeline error: {error_msg}",
            error=error_msg,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end audit of the answer/citation/verifier layer."
    )
    parser.add_argument(
        "--suite",
        default="answer_e2e",
        choices=list(QUERY_SUITES.keys()),
        help="Query suite to run (default: answer_e2e).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON report. Default: data/audits/v1.18.1_answer_verifier_e2e.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Substitute a synthetic LLM response — runs verification without a live LLM.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5).",
    )
    args = parser.parse_args(argv)

    suite = QUERY_SUITES[args.suite]
    output_path = Path(args.output) if args.output else (
        _ROOT / "data" / "audits" / "v1.18.1_answer_verifier_e2e.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RAG-Lab Answer Verifier E2E Audit")
    print(f"  Suite : {args.suite}  ({len(suite)} queries)")
    print(f"  Mode  : {'DRY-RUN (no LLM)' if args.dry_run else 'LIVE (LLM required)'}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    entries: list[AuditEntry] = []
    for q in suite:
        print(f"[{q['id']}] {q['category']:<20} | {q['text'][:60]}")
        entry = run_single(q, dry_run=args.dry_run, top_k=args.top_k)
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(entry.verdict, "?")
        print(f"      {icon} {entry.verdict:<5} | {entry.verdict_reason}")
        if entry.warnings:
            for w in entry.warnings:
                print(f"      · {w}")
        entries.append(entry)

    # Summary
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for e in entries:
        counts[e.verdict] = counts.get(e.verdict, 0) + 1

    print(f"\n{'─'*60}")
    print(f"  Results: {counts['PASS']} PASS  {counts['WARN']} WARN  {counts['FAIL']} FAIL")
    print(f"{'─'*60}\n")

    # Verification checks specific to v1.18 bug fixes
    _print_v1_18_checks(entries)

    # Save JSON
    report = {
        "suite": args.suite,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "counts": counts,
        "entries": [asdict(e) for e in entries],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved → {output_path}\n")

    return 1 if counts["FAIL"] > 0 else 0


def _print_v1_18_checks(entries: list[AuditEntry]) -> None:
    """Print v1.18-specific invariant checks as a mini-audit."""
    print("v1.18 invariant checks:")
    all_ok = True

    # BUG-1: 0/0 ✓ false positive must not appear
    for e in entries:
        if "0/0 ✓" in e.verification_block:
            print(f"  FAIL [BUG-1] {e.id}: 0/0 ✓ false positive still present!")
            all_ok = False
    if all_ok:
        print("  ✓ BUG-1: No 0/0 ✓ false positives found.")

    # BUG-2: responses with zero citations must show warning
    bug2_ok = True
    for e in entries:
        if not e.citation_results and not any("sin citas" in w.lower() or "sin cita" in w.lower() for w in e.warnings):
            # Only flag if the response is non-empty (not a pipeline error)
            if e.response:
                print(f"  FAIL [BUG-2] {e.id}: zero citations but no warning emitted.")
                bug2_ok = False
    if bug2_ok:
        print("  ✓ BUG-2: Zero-citation warnings emitted correctly.")

    # BUG-3: N/A should not appear as consistency status (DEGRADED ⚠ expected instead)
    bug3_ok = True
    for e in entries:
        for line in e.verification_block.splitlines():
            if "Consistencia" in line and "N/A" in line:
                print(f"  FAIL [BUG-3] {e.id}: consistency line still shows N/A.")
                bug3_ok = False
    if bug3_ok:
        print("  ✓ BUG-3: No N/A consistency status found.")

    # FEATURE: evidence_map should be populated for entries with valid citations
    feat_ok = True
    for e in entries:
        valid_cits = [c for c in e.citation_results if c["status"] == "VALID"]
        if valid_cits and not e.evidence_map:
            print(f"  FAIL [FEAT] {e.id}: has VALID citations but evidence_map is empty.")
            feat_ok = False
    if feat_ok:
        print("  ✓ FEAT: evidence_map populated for all entries with valid citations.")

    print()


if __name__ == "__main__":
    from rag_lab.logging_config import setup_logging
    setup_logging("WARNING")
    sys.exit(main())
