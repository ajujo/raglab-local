"""Componente 1: Verificación de citas.

Extrae las citas de la respuesta del LLM y verifica que existan
en los chunks recuperados. Clasifica cada cita como VALID, PARTIAL o INVALID.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("rag_lab")


class CitationStatus(str, Enum):
    """Estado de una cita verificada."""
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"


@dataclass
class CitationResult:
    """Resultado de la verificación de una cita individual."""
    citation_text: str
    status: CitationStatus
    matched_chunk: Optional[dict] = None


def verify_citations_layer(
    response: str,
    retrieved_chunks: List[dict],
) -> List[CitationResult]:
    """Verificar todas las citas de la respuesta del LLM.

    Args:
        response: La respuesta generada por el LLM.
        retrieved_chunks: Lista de chunks recuperados que se enviaron al LLM.

    Returns:
        Lista de CitationResult con el estado de cada cita.
    """
    # Extraer citas con el formato unificado: [[N] Fuente: ... | Sección: ... | Líneas: ...]
    citation_pattern = r'\[\[(\d+)\]\s*Fuente:\s*([^|]+)\|\s*Sección:\s*([^|]+)\|\s*Líneas:\s*(\d+)-(\d+)\]'
    citations = re.findall(citation_pattern, response)

    if not citations:
        logger.info("No se encontraron citas en la respuesta")
        return []

    results = []
    for index, doc_id, heading_path, line_start, line_end in citations:
        citation_text = f"[[{index}] Fuente: {doc_id} | Sección: {heading_path} | Líneas: {line_start}-{line_end}"
        matched = _find_matching_chunk(doc_id, heading_path, line_start, line_end, retrieved_chunks)
        status = _classify_citation(doc_id, heading_path, line_start, line_end, retrieved_chunks)
        results.append(CitationResult(
            citation_text=citation_text,
            status=status,
            matched_chunk=matched,
        ))

    return results


def _find_matching_chunk(
    doc_id: str,
    heading_path: str,
    line_start: str,
    line_end: str,
    chunks: List[dict],
) -> Optional[dict]:
    """Buscar un chunk que coincida con los metadatos de la cita."""
    doc_id = doc_id.strip()
    heading_path = heading_path.strip()
    line_start = line_start.strip()
    line_end = line_end.strip()

    for chunk in chunks:
        chunk_doc_id = chunk.get("doc_id", "")
        chunk_heading = chunk.get("heading_path", "")
        chunk_line_start = str(chunk.get("line_start", ""))
        chunk_line_end = str(chunk.get("line_end", ""))

        # Verificar doc_id
        if chunk_doc_id != doc_id:
            continue

        # Verificar heading_path con fuzzy matching
        if not _fuzzy_match_path(heading_path, chunk_heading):
            continue

        # Verificar rangos de líneas
        if chunk_line_start == line_start and chunk_line_end == line_end:
            return chunk
        elif chunk_line_start == line_start or chunk_line_end == line_end:
            # Coincidencia parcial en líneas
            return chunk

    return None


def _classify_citation(
    doc_id: str,
    heading_path: str,
    line_start: str,
    line_end: str,
    chunks: List[dict],
) -> CitationStatus:
    """Clasificar una cita como VALID, PARTIAL o INVALID."""
    doc_id = doc_id.strip()
    heading_path = heading_path.strip()
    line_start = line_start.strip()
    line_end = line_end.strip()

    for chunk in chunks:
        chunk_doc_id = chunk.get("doc_id", "")
        chunk_heading = chunk.get("heading_path", "")
        chunk_line_start = str(chunk.get("line_start", ""))
        chunk_line_end = str(chunk.get("line_end", ""))

        # Verificar doc_id
        if chunk_doc_id != doc_id:
            continue

        # Verificar heading_path con fuzzy matching
        if not _fuzzy_match_path(heading_path, chunk_heading):
            continue

        # Verificar rangos de líneas
        if chunk_line_start == line_start and chunk_line_end == line_end:
            return CitationStatus.VALID
        elif chunk_line_start == line_start or chunk_line_end == line_end:
            return CitationStatus.PARTIAL

    return CitationStatus.INVALID


def _fuzzy_match_path(cited_path: str, chunk_path: str) -> bool:
    """Verificar si una ruta de encabezado coincide con la de un chunk."""
    cited_lower = cited_path.strip().lower()
    chunk_lower = chunk_path.strip().lower()

    # Coincidencia exacta
    if cited_lower == chunk_lower:
        return True

    # La ruta citada es un prefijo de la ruta del chunk
    if chunk_lower.startswith(cited_lower):
        return True

    # La ruta citada aparece en la ruta del chunk
    if cited_lower in chunk_lower:
        return True

    return False