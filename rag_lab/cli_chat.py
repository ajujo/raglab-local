"""Modo chat interactivo para RAG-Lab.

Permite conversaciones continuas con:
- Filtrado por documento (/docs)
- Historial de conversación
- Comandos internos (/help, /clear, /mode, /temp, /topk, /quit)
- Indicadores de fuentes
- HyDE, query rewriting y feedback loop integrados
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.prompt import Prompt

from rag_lab.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP,
    DATA_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    SOURCES,
    STORAGE_DIR,
)
from rag_lab.embedding.encoder import encode_chunks
from rag_lab.feedback.feedback_store import FeedbackEntry, save_feedback, init_db
from rag_lab.generation.llm_client import generate_response
from rag_lab.generation.prompt_builder import build_prompt
from rag_lab.retrieval.query_rewriter import rewrite_query
from rag_lab.verification.pipeline import verify_and_score
from rag_lab.config import ENABLE_CONSISTENCY_CHECK
from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.retrieval.query_processor import process_query
from rag_lab.retrieval.reranker import rerank
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.sparse_store import SparseStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")
console = Console()


class ChatSession:
    """Sesión de chat con historial y configuración dinámica."""

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.active_docs: Optional[List[str]] = None  # None = todos
        self.mode: str = "standard"  # fast, standard, hyde
        self.temperature: float = LLM_TEMPERATURE
        self.top_k: int = RETRIEVAL_TOP_K
        self.rerank_top_k: int = RERANK_TOP_K
        self.embedding_device: str = EMBEDDING_DEVICE
        self.reranker_device: str = EMBEDDING_DEVICE

        # Flags de mejora del chat
        self.hyde_enabled: bool = False
        self.rewrite_enabled: bool = False
        self.feedback_enabled: bool = True

        # Inicializar almacenes
        self.vector_store = VectorStore()
        self.sparse_store = SparseStore()
        self.doc_store = DocStore()
        self.vector_store.initialize()
        self.sparse_store.load()

    def _get_doc_ids(self) -> Optional[List[str]]:
        """Obtener IDs de documentos activos."""
        if self.active_docs is None:
            return None
        return self.active_docs

    def _filter_results(self, results: List[dict]) -> List[dict]:
        """Filtrar resultados por documentos activos."""
        if self.active_docs is None:
            return results
        return [r for r in results if r.get("doc_id") in self.active_docs]

    def _run_query(self, question: str) -> Tuple[str, List[str], object, List[dict]]:
        """Ejecutar una consulta RAG y devolver (respuesta, fuentes, verification, chunks)."""
        # Procesar consulta
        queries = process_query(
            question,
            use_hyde=self.hyde_enabled,
            use_rewriting=self.rewrite_enabled,
        )

        # Obtener embeddings de consulta
        all_query_data = []
        for q in queries:
            dense_emb, sparse_dict = encode_chunks(
                [{"text": q["text"]}, {"text": q.get("hyde_text", q["text"])}, {"text": q["text"]}],
                batch_size=1,
                device=self.embedding_device,
            )
            query_dense = dense_emb[0]
            query_sparse = next(iter(sparse_dict.values()), {}) if sparse_dict else {}
            all_query_data.append((query_dense, query_sparse))

        # Búsqueda híbrida
        all_results = []
        doc_ids = self._get_doc_ids()
        for query_dense, query_sparse in all_query_data:
            results = hybrid_search(
                question,
                self.vector_store,
                self.sparse_store,
                self.doc_store,
                query_dense=query_dense,
                query_sparse=query_sparse,
                top_k=self.top_k,
                doc_ids=doc_ids,
            )
            all_results.extend(results)

        # Filtrar por documentos activos
        all_results = self._filter_results(all_results)

        # Deduplicar
        seen = set()
        unique_results = []
        for r in all_results:
            cid = r.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                unique_results.append(r)

        # Reranking (salto en modo fast)
        if self.mode != "fast" and unique_results:
            unique_results = rerank(
                question,
                unique_results[:min(20, len(unique_results))],
                top_k=min(self.rerank_top_k, len(unique_results)),
                device=self.reranker_device,
            )

        if not unique_results:
            return "No se encontraron resultados relevantes.", [], None, []

        # Construir prompt con historial
        system_prompt, user_prompt = build_prompt(question, unique_results[:self.rerank_top_k])

        # Generar respuesta
        response = generate_response(system_prompt, user_prompt, temperature=self.temperature)

        # Extraer fuentes
        sources = list({r.get("doc_id", "desconocido") for r in unique_results[:self.rerank_top_k]})

        # Ejecutar pipeline de verificación
        retrieval_scores = [r.get("score", 0.5) for r in unique_results[:self.rerank_top_k]]
        verification = verify_and_score(
            response or "No se pudo generar una respuesta.",
            unique_results[:self.rerank_top_k],
            retrieval_scores,
            enable_consistency_check=ENABLE_CONSISTENCY_CHECK,
        )

        return verification.response, sources, verification, unique_results[:self.rerank_top_k]

    def handle_command(self, command: str, *args) -> Optional[str]:
        """Manejar comandos internos."""
        cmd = command.lower().lstrip("/")

        if cmd == "help":
            return self._help_text()
        elif cmd == "clear":
            self.history.clear()
            return "Historial limpiado."
        elif cmd == "docs":
            if not args:
                active = self.active_docs or ["todos"]
                return f"Documentos activos: {', '.join(active)}"
            doc_list = args[0].split(",")
            if args[0].lower() == "all":
                self.active_docs = None
                return f"Mostrando todos los documentos ({len(SOURCES)} fuentes)"
            self.active_docs = [d.strip() for d in doc_list]
            return f"Documentos activos: {', '.join(self.active_docs)}"
        elif cmd == "mode":
            if args:
                mode = args[0].lower()
                if mode in ("fast", "standard", "hyde"):
                    self.mode = mode
                    # Backward compatibility: /mode hyde also enables hyde_enabled
                    if mode == "hyde":
                        self.hyde_enabled = True
                    elif mode == "standard":
                        self.hyde_enabled = False
                    return f"Modo cambiado a: {mode}"
                return "Modo inválido. Opciones: fast, standard, hyde"
            return f"Modo actual: {self.mode}"
        elif cmd == "temp":
            if args:
                try:
                    self.temperature = float(args[0])
                    return f"Temperatura: {self.temperature}"
                except ValueError:
                    return "Valor inválido. Debe ser un número (ej. 0.1)"
            return f"Temperatura actual: {self.temperature}"
        elif cmd == "topk":
            if args:
                try:
                    self.top_k = int(args[0])
                    return f"Top-k: {self.top_k}"
                except ValueError:
                    return "Valor inválido. Debe ser un número entero."
            return f"Top-k actual: {self.top_k}"
        elif cmd == "hyde":
            if not args:
                return f"HyDE: {'activado' if self.hyde_enabled else 'desactivado'}"
            val = args[0].lower()
            if val == "on":
                self.hyde_enabled = True
                return "HyDE activado: las consultas usarán hipótesis generadas por el LLM."
            elif val == "off":
                self.hyde_enabled = False
                return "HyDE desactivado."
            return "Uso: /hyde [on|off]"
        elif cmd == "rewrite":
            if not args:
                return f"Query rewriting: {'activado' if self.rewrite_enabled else 'desactivado'}"
            val = args[0].lower()
            if val == "on":
                self.rewrite_enabled = True
                return "Query rewriting activado: las preguntas se reformularán para mejorar la recuperación."
            elif val == "off":
                self.rewrite_enabled = False
                return "Query rewriting desactivado."
            return "Uso: /rewrite [on|off]"
        elif cmd == "feedback":
            if not args:
                return f"Feedback: {'activado' if self.feedback_enabled else 'desactivado'}"
            val = args[0].lower()
            if val == "on":
                self.feedback_enabled = True
                return "Feedback activado."
            elif val == "off":
                self.feedback_enabled = False
                return "Feedback desactivado."
            return "Uso: /feedback [on|off]"
        elif cmd in ("quit", "exit", "bye"):
            return "__QUIT__"
        else:
            return f"Comando desconocido: /{cmd}. Escribe /help para ver los comandos disponibles."

    def _help_text(self) -> str:
        return """
Comandos disponibles:
  /help          - Muestra esta ayuda
  /clear         - Limpia el historial de conversación
  /docs <docs>   - Filtra documentos (ej. /docs doc1,doc2 o /docs all)
  /mode <modo>   - Cambia modo: fast, standard, hyde
  /temp <valor>  - Cambia temperatura del LLM (ej. /temp 0.1)
  /topk <n>      - Cambia número de chunks (ej. /topk 20)
  /hyde [on|off] - Activa/desactiva HyDE (hipótesis con LLM)
  /rewrite [on|off] - Activa/desactiva query rewriting
  /feedback [on|off] - Activa/desactiva prompt de feedback
  /quit           - Sale del chat
"""

    def _collect_feedback(
        self,
        question: str,
        rewritten_query: str | None,
        hyde_used: bool,
        chunks: list[dict],
        final_score: float,
        score_level: str,
    ) -> None:
        """Prompt al usuario por feedback y guardar en SQLite."""
        # Construir metadatos de chunks (sin texto completo)
        chunk_metas = []
        for c in chunks:
            chunk_metas.append({
                "doc_id": c.get("doc_id", ""),
                "heading_path": c.get("heading_path", ""),
                "line_start": c.get("line_start", 0),
                "line_end": c.get("line_end", 0),
                "retrieval_score": c.get("score", 0.5),
            })

        console.print("\n¿Esta respuesta fue útil? [s/n] (Enter para omitir): ", end="")
        try:
            answer = sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if answer in ("s", "sí", "si", "yes", "y"):
            entry = FeedbackEntry(
                question=question,
                rewritten_query=rewritten_query,
                hyde_used=hyde_used,
                chunks_retrieved=json.dumps(chunk_metas, ensure_ascii=False),
                final_score=final_score,
                score_level=score_level,
                useful=True,
                timestamp=datetime.now().isoformat(),
            )
            save_feedback(entry)
            console.print("[bold green]✅ Feedback guardado: Útil[/bold green]")
        elif answer in ("n", "no"):
            entry = FeedbackEntry(
                question=question,
                rewritten_query=rewritten_query,
                hyde_used=hyde_used,
                chunks_retrieved=json.dumps(chunk_metas, ensure_ascii=False),
                final_score=final_score,
                score_level=score_level,
                useful=False,
                timestamp=datetime.now().isoformat(),
            )
            save_feedback(entry)
            console.print("[bold yellow]⚠️ Feedback guardado: No útil[/bold yellow]")
        # Enter o input inválido → se omite

    def chat_loop(self):
        """Bucle principal del chat."""
        console.print("[bold cyan]🔒 Chat RAG-Lab iniciado[/bold cyan]")
        console.print(f"[dim]Documentos disponibles: {len(SOURCES)} fuentes[/dim]")
        console.print(
            f"[dim]Estado: HyDE={'ON' if self.hyde_enabled else 'OFF'}, "
            f"Rewriting={'ON' if self.rewrite_enabled else 'OFF'}, "
            f"Feedback={'ON' if self.feedback_enabled else 'OFF'}[/dim]"
        )
        console.print("[dim]Escribe /help para ver los comandos disponibles.[/dim]")
        console.print("[dim]Escribe tu pregunta o un comando con /[/dim]\n")

        # Inicializar DB de feedback
        init_db()

        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]👋 Chat finalizado.[/bold yellow]")
                return

            user_input = user_input.strip()
            if not user_input:
                continue

            # Comandos internos
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0]
                args = parts[1].split() if len(parts) > 1 else []
                result = self.handle_command(cmd, *args)

                if result == "__QUIT__":
                    console.print("[bold yellow]👋 Chat finalizado.[/bold yellow]")
                    return
                elif result:
                    console.print(f"[dim]{result}[/dim]")
                continue

            # Consulta normal
            console.print("[dim]Buscando...[/dim]")
            response, sources, verification, chunks = self._run_query(user_input)

            # Si no hay resultados, saltar feedback
            if verification is None:
                console.print(f"\n[bold magenta]🤖 RAG-Lab:[/bold magenta] {response}")
                continue

            # Mostrar respuesta y fuentes
            console.print(f"\n[bold magenta]🤖 RAG-Lab:[/bold magenta] {response}")

            # Mostrar advertencias si las hay
            warnings = verification.get_warnings()
            for warning in warnings:
                console.print(f"[bold yellow]⚠️ {warning}[/bold yellow]")

            # Mostrar bloque de verificación
            console.print(f"\n{verification.format_verification_block()}")

            if sources:
                console.print(f"[dim]Fuentes: {', '.join(sources)}[/dim]")

            # Feedback prompt
            if self.feedback_enabled:
                self._collect_feedback(
                    question=user_input,
                    rewritten_query=None,
                    hyde_used=self.hyde_enabled,
                    chunks=chunks,
                    final_score=verification.score_result.final_score,
                    score_level=verification.score_result.confidence_level.value,
                )

            console.print()


def run_chat(
    cpu_embedding: bool = False,
    cpu_reranker: bool = False,
) -> None:
    """Ejecutar el modo chat interactivo."""
    session = ChatSession()

    if cpu_embedding:
        session.embedding_device = "cpu"
    if cpu_reranker:
        session.reranker_device = "cpu"

    session.chat_loop()
