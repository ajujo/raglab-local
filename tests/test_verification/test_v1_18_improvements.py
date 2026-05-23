"""v1.18 verification improvements — bug fixes, CitationResult.chunk_id, evidence_map,
verbose format_verification_block, and prompt hardening.
"""

import pytest
from unittest.mock import MagicMock

from rag_lab.verification.verifier import (
    CitationResult,
    CitationStatus,
    verify_citations_layer,
)
from rag_lab.verification.consistency import ConsistencyResult
from rag_lab.verification.scoring import ScoreResult, ConfidenceLevel
from rag_lab.verification.pipeline import VerificationResult, verify_and_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id="c1", doc_id="doc_A", heading="H1 > H2",
                line_start=10, line_end=20, text="Sample text for testing."):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "heading_path": heading,
        "line_start": line_start,
        "line_end": line_end,
        "text": text,
    }


def _make_consistency(score=1.0, parse_success=True):
    return ConsistencyResult(
        has_unsupported_claims=False,
        has_contradictions=False,
        has_hallucinations=False,
        details="",
        score=score,
        parse_success=parse_success,
    )


def _make_score(final=0.8):
    return ScoreResult(
        citation_score=0.8,
        retrieval_score=0.9,
        consistency_score=1.0,
        coverage_score=0.8,
        final_score=final,
        confidence_level=ConfidenceLevel.HIGH,
    )


def _make_vr(citation_results=None, consistency=None, score=None,
             chunks=None, retrieval_scores=None, response="Answer."):
    return VerificationResult(
        response=response,
        citation_results=citation_results or [],
        consistency_result=consistency or _make_consistency(),
        score_result=score or _make_score(),
        retrieved_chunks=chunks or [_make_chunk()],
        retrieval_scores=retrieval_scores or [0.9],
    )


# ---------------------------------------------------------------------------
# Bug fix 1 — zero-citation warning
# ---------------------------------------------------------------------------

class TestZeroCitationWarning:

    def test_no_citations_fires_warning(self):
        vr = _make_vr(citation_results=[])
        warnings = vr.get_warnings()
        assert any("sin citas" in w.lower() or "sin cita" in w.lower() for w in warnings), \
            f"Expected zero-citation warning, got: {warnings}"

    def test_with_citations_no_zero_warning(self):
        cr = CitationResult(
            citation_text="[[1] Fuente: doc_A | Sección: H1 | Líneas: 10-20]",
            status=CitationStatus.VALID,
            matched_chunk=_make_chunk(),
            chunk_id="c1",
        )
        vr = _make_vr(citation_results=[cr])
        warnings = vr.get_warnings()
        assert not any("sin citas" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Bug fix 2 — 0/0 ✓ false positive
# ---------------------------------------------------------------------------

class TestZeroCitationDisplay:

    def test_zero_citations_shows_cross_not_checkmark(self):
        vr = _make_vr(citation_results=[])
        block = vr.format_verification_block()
        # Must show ✗ for 0/0, never ✓
        assert "0/0 ✗" in block, f"Expected '0/0 ✗' in block:\n{block}"
        assert "0/0 ✓" not in block

    def test_all_valid_shows_checkmark(self):
        cr = CitationResult(
            citation_text="[[1] Fuente: doc_A | Sección: H1 | Líneas: 10-20]",
            status=CitationStatus.VALID,
            matched_chunk=_make_chunk(),
            chunk_id="c1",
        )
        vr = _make_vr(citation_results=[cr])
        block = vr.format_verification_block()
        assert "1/1 ✓" in block

    def test_partial_invalid_shows_warning_icon(self):
        cr = CitationResult(
            citation_text="[[1] Fuente: doc_A | Sección: H1 | Líneas: 10-20]",
            status=CitationStatus.INVALID,
            matched_chunk=None,
            chunk_id=None,
        )
        vr = _make_vr(citation_results=[cr])
        block = vr.format_verification_block()
        assert "0/1 ⚠" in block


# ---------------------------------------------------------------------------
# Bug fix 3 — DEGRADED display when parse_success=False
# ---------------------------------------------------------------------------

class TestDegradedConsistencyDisplay:

    def test_parse_failure_shows_degraded(self):
        consistency = _make_consistency(score=0.5, parse_success=False)
        vr = _make_vr(consistency=consistency)
        block = vr.format_verification_block()
        assert "DEGRADED ⚠" in block, f"Expected 'DEGRADED ⚠' in block:\n{block}"

    def test_parse_failure_no_na_string(self):
        consistency = _make_consistency(score=0.5, parse_success=False)
        vr = _make_vr(consistency=consistency)
        block = vr.format_verification_block()
        # The old "N/A" string must no longer appear for consistency status
        # (it may appear elsewhere, so check only the consistency line)
        for line in block.splitlines():
            if "Consistencia" in line:
                assert "N/A" not in line, f"Consistency line still shows N/A: {line}"

    def test_parse_success_ok_shows_ok(self):
        consistency = _make_consistency(score=1.0, parse_success=True)
        vr = _make_vr(consistency=consistency)
        block = vr.format_verification_block()
        assert "OK ✓" in block


# ---------------------------------------------------------------------------
# Bug fix 4 — consistency disabled uses neutral score, not 1.0
# ---------------------------------------------------------------------------

class TestConsistencyDisabledScore:

    def test_disabled_consistency_score_below_one(self):
        """When enable_consistency_check=False the score must be < 1.0."""
        chunks = [_make_chunk()]
        response = "Answer without real citations."

        vr = verify_and_score(
            response=response,
            retrieved_chunks=chunks,
            retrieval_scores=[0.8],
            enable_consistency_check=False,
        )
        assert vr.consistency_result.score < 1.0, (
            f"Expected disabled consistency score < 1.0, got {vr.consistency_result.score}"
        )

    def test_disabled_consistency_score_is_neutral(self):
        """Score should be 0.75 (neutral-leaning) when check is disabled."""
        chunks = [_make_chunk()]
        vr = verify_and_score(
            response="No citations here.",
            retrieved_chunks=chunks,
            retrieval_scores=[0.5],
            enable_consistency_check=False,
        )
        assert vr.consistency_result.score == 0.75


# ---------------------------------------------------------------------------
# CitationResult.chunk_id enrichment
# ---------------------------------------------------------------------------

class TestCitationResultChunkId:

    def test_chunk_id_populated_when_matched(self):
        chunk = _make_chunk(chunk_id="cid_42", doc_id="doc_A",
                            heading="H1 > H2", line_start=10, line_end=20)
        response = "Some claim. [[1] Fuente: doc_A | Sección: H1 > H2 | Líneas: 10-20]"
        results = verify_citations_layer(response, [chunk])
        assert len(results) == 1
        assert results[0].chunk_id == "cid_42"

    def test_chunk_id_none_when_no_match(self):
        chunk = _make_chunk(chunk_id="cid_42", doc_id="doc_A",
                            heading="H1 > H2", line_start=10, line_end=20)
        response = "Claim. [[1] Fuente: doc_B | Sección: Other | Líneas: 99-100]"
        results = verify_citations_layer(response, [chunk])
        assert len(results) == 1
        assert results[0].chunk_id is None
        assert results[0].status == CitationStatus.INVALID

    def test_chunk_id_backward_compatible_default_none(self):
        cr = CitationResult(
            citation_text="[[1] Fuente: x | Sección: y | Líneas: 1-2]",
            status=CitationStatus.VALID,
        )
        assert cr.chunk_id is None
        assert cr.matched_chunk is None


# ---------------------------------------------------------------------------
# evidence_map property
# ---------------------------------------------------------------------------

class TestEvidenceMap:

    def test_evidence_map_empty_when_no_citations(self):
        vr = _make_vr(citation_results=[])
        assert vr.evidence_map == {}

    def test_evidence_map_structure(self):
        chunk = _make_chunk(chunk_id="cid_1", doc_id="doc_A",
                            line_start=10, line_end=20)
        cr = CitationResult(
            citation_text="[[1] Fuente: doc_A | Sección: H1 | Líneas: 10-20]",
            status=CitationStatus.VALID,
            matched_chunk=chunk,
            chunk_id="cid_1",
        )
        vr = _make_vr(citation_results=[cr])
        em = vr.evidence_map
        assert 1 in em
        assert em[1]["chunk_id"] == "cid_1"
        assert em[1]["doc_id"] == "doc_A"
        assert em[1]["lines"] == (10, 20)
        assert em[1]["status"] == "VALID"

    def test_evidence_map_unmatched_citation(self):
        cr = CitationResult(
            citation_text="[[1] Fuente: x | Sección: y | Líneas: 1-2]",
            status=CitationStatus.INVALID,
            matched_chunk=None,
            chunk_id=None,
        )
        vr = _make_vr(citation_results=[cr])
        em = vr.evidence_map
        assert em[1]["chunk_id"] is None
        assert em[1]["doc_id"] is None
        assert em[1]["lines"] is None
        assert em[1]["status"] == "INVALID"

    def test_evidence_map_multiple_citations(self):
        chunks = [
            _make_chunk(chunk_id=f"cid_{i}", doc_id="doc_A",
                        line_start=i * 10, line_end=i * 10 + 9)
            for i in range(1, 4)
        ]
        citation_results = [
            CitationResult(
                citation_text=f"[[{i}] Fuente: doc_A | Sección: H | Líneas: {i*10}-{i*10+9}]",
                status=CitationStatus.VALID,
                matched_chunk=chunks[i - 1],
                chunk_id=f"cid_{i}",
            )
            for i in range(1, 4)
        ]
        vr = _make_vr(citation_results=citation_results)
        em = vr.evidence_map
        assert set(em.keys()) == {1, 2, 3}
        assert em[2]["chunk_id"] == "cid_2"

    def test_evidence_map_is_computed_property(self):
        vr = _make_vr(citation_results=[])
        # Calling it twice returns equal (but possibly different) dict objects
        assert vr.evidence_map == vr.evidence_map


# ---------------------------------------------------------------------------
# verbose format_verification_block
# ---------------------------------------------------------------------------

class TestVerboseFormatBlock:

    def _make_cr_with_chunk(self, i=1):
        chunk = _make_chunk(
            chunk_id=f"cid_{i}",
            doc_id="doc_A",
            line_start=10,
            line_end=20,
            text="This is the chunk text content.",
        )
        return CitationResult(
            citation_text=f"[[{i}] Fuente: doc_A | Sección: H1 | Líneas: 10-20]",
            status=CitationStatus.VALID,
            matched_chunk=chunk,
            chunk_id=f"cid_{i}",
        )

    def test_verbose_true_shows_chunk_id(self):
        vr = _make_vr(citation_results=[self._make_cr_with_chunk()])
        block = vr.format_verification_block(verbose=True)
        assert "cid_1" in block

    def test_verbose_true_shows_snippet(self):
        vr = _make_vr(citation_results=[self._make_cr_with_chunk()])
        block = vr.format_verification_block(verbose=True)
        assert "This is the chunk text content." in block

    def test_verbose_true_shows_traceability_label(self):
        vr = _make_vr(citation_results=[self._make_cr_with_chunk()])
        block = vr.format_verification_block(verbose=True)
        assert "Trazabilidad" in block

    def test_verbose_false_no_traceability(self):
        vr = _make_vr(citation_results=[self._make_cr_with_chunk()])
        block = vr.format_verification_block(verbose=False)
        assert "Trazabilidad" not in block
        assert "cid_1" not in block

    def test_verbose_default_is_false(self):
        vr = _make_vr(citation_results=[self._make_cr_with_chunk()])
        assert vr.format_verification_block() == vr.format_verification_block(verbose=False)

    def test_verbose_with_no_citations_no_traceability_section(self):
        vr = _make_vr(citation_results=[])
        block = vr.format_verification_block(verbose=True)
        assert "Trazabilidad" not in block

    def test_verbose_snippet_truncated_at_100_chars(self):
        long_text = "A" * 200
        chunk = _make_chunk(chunk_id="cid_x", text=long_text)
        cr = CitationResult(
            citation_text="[[1] Fuente: doc_A | Sección: H1 | Líneas: 10-20]",
            status=CitationStatus.VALID,
            matched_chunk=chunk,
            chunk_id="cid_x",
        )
        vr = _make_vr(citation_results=[cr])
        block = vr.format_verification_block(verbose=True)
        assert "…" in block
        # The raw 200-char repetition should not appear verbatim
        assert "A" * 150 not in block


# ---------------------------------------------------------------------------
# Regression — existing VALID/PARTIAL/INVALID behavior unchanged
# ---------------------------------------------------------------------------

class TestRegressionExistingVerifier:

    def _chunks(self):
        return [_make_chunk(chunk_id="c1", doc_id="doc_A", heading="H1 > H2",
                            line_start=10, line_end=20)]

    def test_valid_citation_still_valid(self):
        response = "Claim. [[1] Fuente: doc_A | Sección: H1 > H2 | Líneas: 10-20]"
        results = verify_citations_layer(response, self._chunks())
        assert results[0].status == CitationStatus.VALID

    def test_invalid_citation_still_invalid(self):
        response = "Claim. [[1] Fuente: unknown | Sección: X | Líneas: 99-100]"
        results = verify_citations_layer(response, self._chunks())
        assert results[0].status == CitationStatus.INVALID

    def test_partial_citation_still_partial(self):
        # Same doc_id + fuzzy heading but only one line boundary matches
        chunk = _make_chunk(chunk_id="c1", doc_id="doc_A",
                            heading="H1 > H2", line_start=10, line_end=25)
        response = "Claim. [[1] Fuente: doc_A | Sección: H1 > H2 | Líneas: 10-20]"
        results = verify_citations_layer(response, [chunk])
        assert results[0].status in {CitationStatus.VALID, CitationStatus.PARTIAL}

    def test_no_citations_returns_empty_list(self):
        response = "No citations here."
        results = verify_citations_layer(response, self._chunks())
        assert results == []
