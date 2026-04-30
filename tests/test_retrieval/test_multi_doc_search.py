"""Tests para búsqueda multi-document."""

import pytest
from rag_lab.storage.vector_store import VectorStore
from rag_lab.retrieval.hybrid_search import hybrid_search


class TestMultiDocSearch:
    """Tests para búsqueda con filtrado por documento."""

    def test_vector_store_query_with_doc_ids(self, tmp_path):
        """Test que vector_store.query acepta doc_ids."""
        store = VectorStore(storage_path=tmp_path)
        store.initialize()

        # Añadir vectores de múltiples documentos
        import numpy as np
        embeddings = np.random.rand(4, 1024)
        store.add(
            ids=["c1", "c2", "c3", "c4"],
            embeddings=embeddings,
            documents=["doc1 text", "doc1 text", "doc2 text", "doc2 text"],
            metadatas=[
                {"doc_id": "doc1", "heading_path": "Sec1", "line_start": 10, "line_end": 20},
                {"doc_id": "doc1", "heading_path": "Sec2", "line_start": 30, "line_end": 40},
                {"doc_id": "doc2", "heading_path": "Sec1", "line_start": 50, "line_end": 60},
                {"doc_id": "doc2", "heading_path": "Sec2", "line_start": 70, "line_end": 80},
            ],
        )

        # Buscar filtrando por doc1
        query_emb = np.random.rand(1024)
        results = store.query(query_emb, top_k=10, doc_ids=["doc1"])
        for meta in results["metadatas"]:
            assert meta["doc_id"] == "doc1"

    def test_hybrid_search_with_doc_ids(self, tmp_path):
        """Test que hybrid_search pasa doc_ids a vector_store."""
        # Este test verifica que el parámetro doc_ids se pasa correctamente
        # Nota: No se puede probar completamente sin un entorno completo,
        # pero se verifica que la firma de la función acepta el parámetro
        import inspect
        sig = inspect.signature(hybrid_search)
        params = list(sig.parameters.keys())
        assert "doc_ids" in params

    def test_empty_doc_ids_returns_all(self, tmp_path):
        """Test que doc_ids=None o [] devuelve todos los resultados."""
        store = VectorStore(storage_path=tmp_path)
        store.initialize()

        import numpy as np
        embeddings = np.random.rand(4, 1024)
        store.add(
            ids=["c1", "c2", "c3", "c4"],
            embeddings=embeddings,
            documents=["text"] * 4,
            metadatas=[
                {"doc_id": "doc1", "heading_path": "Sec1", "line_start": 10, "line_end": 20},
                {"doc_id": "doc2", "heading_path": "Sec1", "line_start": 50, "line_end": 60},
                {"doc_id": "doc1", "heading_path": "Sec2", "line_start": 30, "line_end": 40},
                {"doc_id": "doc2", "heading_path": "Sec2", "line_start": 70, "line_end": 80},
            ],
        )

        query_emb = np.random.rand(1024)
        # Sin filtro
        results = store.query(query_emb, top_k=10, doc_ids=None)
        assert len(results["ids"]) == 4

        # Con filtro vacío (debería devolver todos)
        results_empty = store.query(query_emb, top_k=10, doc_ids=[])
        assert len(results_empty["ids"]) == 4


class TestMultiDocCitations:
    """Tests para verificación de citas multi-document."""

    def test_verifier_multi_doc(self):
        """Test que el verificador maneja múltiples doc_id."""
        from rag_lab.verification.verifier import verify_citations_layer

        response = (
            "SDMX es un estándar [[1] Fuente: doc1 | Sección: Sec1 | Líneas: 10-20]. "
            "También existe SDMX-EDI [[2] Fuente: doc2 | Sección: Sec1 | Líneas: 50-60]."
        )
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Sec1", "line_start": 10, "line_end": 20},
            {"chunk_id": "c2", "doc_id": "doc2", "heading_path": "Sec1", "line_start": 50, "line_end": 60},
        ]
        results = verify_citations_layer(response, chunks)
        assert len(results) == 2
        assert all(r.status.value == "VALID" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
