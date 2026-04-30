"""Tests para los comandos del modo chat."""

import pytest
from rag_lab.cli_chat import ChatSession


class TestChatCommands:
    """Tests para los comandos /hyde, /rewrite, /feedback."""

    @pytest.fixture
    def session(self):
        """Crear una sesión de chat sin conectar a bases de datos reales."""
        # No inicializar almacenes reales para los tests
        session = object.__new__(ChatSession)
        session.history = []
        session.active_docs = None
        session.mode = "standard"
        session.temperature = 0.1
        session.top_k = 20
        session.rerank_top_k = 8
        session.embedding_device = "cpu"
        session.reranker_device = "cpu"
        session.hyde_enabled = False
        session.rewrite_enabled = False
        session.feedback_enabled = True
        return session

    def test_hyde_on(self, session):
        result = session.handle_command("/hyde", "on")
        assert session.hyde_enabled == True
        assert "activado" in result

    def test_hyde_off(self, session):
        session.hyde_enabled = True
        result = session.handle_command("/hyde", "off")
        assert session.hyde_enabled == False
        assert "desactivado" in result

    def test_hyde_status(self, session):
        session.hyde_enabled = True
        result = session.handle_command("/hyde")
        assert "activado" in result

    def test_rewrite_on(self, session):
        result = session.handle_command("/rewrite", "on")
        assert session.rewrite_enabled == True
        assert "activado" in result

    def test_rewrite_off(self, session):
        session.rewrite_enabled = True
        result = session.handle_command("/rewrite", "off")
        assert session.rewrite_enabled == False
        assert "desactivado" in result

    def test_feedback_on(self, session):
        session.feedback_enabled = False
        result = session.handle_command("/feedback", "on")
        assert session.feedback_enabled == True
        assert "activado" in result

    def test_feedback_off(self, session):
        result = session.handle_command("/feedback", "off")
        assert session.feedback_enabled == False
        assert "desactivado" in result

    def test_mode_hyde_sets_hyde_flag(self, session):
        result = session.handle_command("/mode", "hyde")
        assert session.mode == "hyde"
        assert session.hyde_enabled == True

    def test_mode_standard_clears_hyde_flag(self, session):
        session.hyde_enabled = True
        session.mode = "hyde"
        result = session.handle_command("/mode", "standard")
        assert session.mode == "standard"
        assert session.hyde_enabled == False

    def test_unknown_command(self, session):
        result = session.handle_command("/unknown")
        assert "desconocido" in result

    def test_quit_command(self, session):
        result = session.handle_command("/quit")
        assert result == "__QUIT__"

    def test_help_command(self, session):
        result = session.handle_command("/help")
        assert "/hyde" in result
        assert "/rewrite" in result
        assert "/feedback" in result


class TestChatStatePersistence:
    """Tests para la persistencia del estado entre turnos."""

    @pytest.fixture
    def session(self):
        session = object.__new__(ChatSession)
        session.history = []
        session.active_docs = None
        session.mode = "standard"
        session.temperature = 0.1
        session.top_k = 20
        session.rerank_top_k = 8
        session.embedding_device = "cpu"
        session.reranker_device = "cpu"
        session.hyde_enabled = False
        session.rewrite_enabled = False
        session.feedback_enabled = True
        return session

    def test_state_persists_across_turns(self, session):
        # Activar HyDE
        session.handle_command("/hyde", "on")
        assert session.hyde_enabled == True

        # Simular otro turno — el estado se mantiene
        assert session.hyde_enabled == True
        assert session.rewrite_enabled == False
        assert session.feedback_enabled == True

    def test_multiple_flags_independent(self, session):
        session.handle_command("/hyde", "on")
        session.handle_command("/rewrite", "on")
        session.handle_command("/feedback", "off")

        assert session.hyde_enabled == True
        assert session.rewrite_enabled == True
        assert session.feedback_enabled == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
