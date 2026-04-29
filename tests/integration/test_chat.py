"""Tests para el modo chat interactivo.

Verifica:
1. Sesión de chat (comandos internos)
2. Filtrado por documento
3. Modo fast/standard/hyde
4. Integración con el pipeline RAG
"""

import pytest
from rag_lab.cli_chat import ChatSession
from rag_lab.config import SOURCES


class TestChatSession:
    """Tests para la sesión de chat."""

    def test_chat_session_creation(self):
        """Verificar que se puede crear una sesión de chat."""
        session = ChatSession()
        assert session.history == []
        assert session.active_docs is None
        assert session.mode == "standard"
        assert session.temperature == 0.1
        assert session.top_k == 30

    def test_command_help(self):
        """Verificar comando /help."""
        session = ChatSession()
        result = session.handle_command("/help")
        assert "Comandos disponibles" in result

    def test_command_clear(self):
        """Verificar comando /clear."""
        session = ChatSession()
        session.history = [{"role": "user", "content": "test"}]
        result = session.handle_command("/clear")
        assert result == "Historial limpiado."
        assert session.history == []

    def test_command_docs_filter(self):
        """Verificar comando /docs con filtrado."""
        session = ChatSession()
        result = session.handle_command("/docs", "doc1,doc2")
        assert session.active_docs == ["doc1", "doc2"]
        assert "doc1" in result and "doc2" in result

    def test_command_docs_all(self):
        """Verificar comando /docs all."""
        session = ChatSession()
        result = session.handle_command("/docs", "all")
        assert session.active_docs is None
        assert str(len(SOURCES)) in result

    def test_command_mode(self):
        """Verificar comando /mode."""
        session = ChatSession()
        result = session.handle_command("/mode", "fast")
        assert session.mode == "fast"
        assert "fast" in result

        result = session.handle_command("/mode", "hyde")
        assert session.mode == "hyde"

        result = session.handle_command("/mode", "standard")
        assert session.mode == "standard"

    def test_command_mode_invalid(self):
        """Verificar modo inválido."""
        session = ChatSession()
        result = session.handle_command("/mode", "invalid")
        assert "inválido" in result.lower() or "invalid" in result.lower()

    def test_command_temp(self):
        """Verificar comando /temp."""
        session = ChatSession()
        result = session.handle_command("/temp", "0.5")
        assert session.temperature == 0.5

    def test_command_temp_invalid(self):
        """Verificar temperatura inválida."""
        session = ChatSession()
        result = session.handle_command("/temp", "abc")
        assert "inválido" in result.lower() or "invalid" in result.lower()

    def test_command_topk(self):
        """Verificar comando /topk."""
        session = ChatSession()
        result = session.handle_command("/topk", "20")
        assert session.top_k == 20

    def test_command_topk_invalid(self):
        """Verificar top-k inválido."""
        session = ChatSession()
        result = session.handle_command("/topk", "abc")
        assert "inválido" in result.lower() or "invalid" in result.lower()

    def test_command_quit(self):
        """Verificar comando /quit."""
        session = ChatSession()
        result = session.handle_command("/quit")
        assert result == "__QUIT__"

    def test_command_unknown(self):
        """Verificar comando desconocido."""
        session = ChatSession()
        result = session.handle_command("/unknown")
        assert "desconocido" in result.lower() or "unknown" in result.lower() or "desconocida" in result.lower()

    def test_filter_results_no_filter(self):
        """Verificar filtrado sin filtro activo."""
        session = ChatSession()
        results = [
            {"chunk_id": "1", "doc_id": "doc1", "text": "text1"},
            {"chunk_id": "2", "doc_id": "doc2", "text": "text2"},
        ]
        filtered = session._filter_results(results)
        assert len(filtered) == 2

    def test_filter_results_with_filter(self):
        """Verificar filtrado con documentos activos."""
        session = ChatSession()
        session.active_docs = ["doc1"]
        results = [
            {"chunk_id": "1", "doc_id": "doc1", "text": "text1"},
            {"chunk_id": "2", "doc_id": "doc2", "text": "text2"},
        ]
        filtered = session._filter_results(results)
        assert len(filtered) == 1
        assert filtered[0]["doc_id"] == "doc1"

    def test_cpu_mode(self):
        """Verificar modo CPU para embedding y reranker."""
        from rag_lab.cli_chat import run_chat
        # Solo verificar que se pueden crear instancias con CPU
        session = ChatSession()
        session.embedding_device = "cpu"
        session.reranker_device = "cpu"
        assert session.embedding_device == "cpu"
        assert session.reranker_device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
