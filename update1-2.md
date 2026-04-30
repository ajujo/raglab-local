# RAG-Lab Update 1.2

## Descripción
Nueva fase de desarrollo para el sistema RAG-Lab. Este documento registra las mejoras, correcciones y nuevas funcionalidades que se implementaron en esta versión.

## Tareas Completadas
- [x] Bloque 1: Estructura Base y Configuración
- [x] Bloque 2: Verifier Layer (Componente 1)
- [x] Bloque 3: Self-consistency Check (Componente 2)
- [x] Bloque 4: Scoring y Pipeline (Componente 3 y Orquestación)
- [x] Integración en CLI y modo chat
- [x] Actualización de documentación (README.md y update1-2.md)
- [x] Mejora 1: Conectar HyDE al LLM real
- [x] Mejora 2: Score de relevancia por chunk visible
- [x] Mejora 3: Query rewriting
- [x] Mejora 4: Feedback loop mínimo

## Mejoras Implementadas

### Mejora 1 — HyDE con LLM real
- **Archivos modificados**: `rag_lab/retrieval/query_processor.py`
- **Descripción**: Se reemplazó la plantilla fija de HyDE por una llamada real al LLM usando un prompt técnico especializado.
- **Log**: `HyDE: hipótesis generada (188 tokens) para query: "¿Qué es SDMX?..."`
- **Commit**: `b37b8a2`

### Mejora 2 — Score de relevancia por chunk visible
- **Archivos modificados**: `rag_lab/verification/pipeline.py`
- **Descripción**: Se añadió una sección de fragmentos recuperados en el bloque de verificación, mostrando el score individual de cada chunk con barras visuales (`██████████░░░░░`).
- **Advertencia**: Si algún score < 0.60, se muestra `⚠ Algunos fragmentos tienen relevancia baja. Considera reformular la pregunta o activar HyDE con --hyde para mejorar la recuperación.`
- **Commit**: `5111e8c`

### Mejora 3 — Query rewriting
- **Archivos nuevos**: `rag_lab/retrieval/query_rewriter.py`
- **Archivos modificados**: `rag_lab/retrieval/query_processor.py`, `rag_lab/config.py`, `rag_lab/cli.py`
- **Descripción**: Se añadió un módulo de rewriting que reformula las preguntas del usuario para maximizar la recuperación semántica (expansión de siglas, terminología técnica).
- **Flag CLI**: `--rewrite`
- **Log**: `QueryRewriter: "¿Qué es DSD?" → "¿Qué es la Definición de Estructura de Datos (DSD)?"`
- **Commit**: `ea16178`

### Mejora 4 — Feedback loop mínimo
- **Archivos nuevos**: `rag_lab/feedback/feedback_store.py`, `rag_lab/feedback/analyze_feedback.py`, `rag_lab/feedback/__init__.py`
- **Archivos modificados**: `rag_lab/cli.py`
- **Descripción**: Se añadió un sistema de feedback con almacenamiento SQLite que permite al usuario calificar la utilidad de las respuestas. El script `analyze_feedback.py` genera un resumen estadístico.
- **Flag CLI**: `--no-feedback` para desactivar el prompt de feedback.
- **Comando de análisis**: `python -m rag_lab.feedback.analyze_feedback`
- **Commit**: `a4e3d9b`

## Pipeline de Consulta Completo
```
Pregunta original
      ↓
Query rewriting (si --rewrite)
      ↓
HyDE (si --hyde) — opera sobre la pregunta reescrita
      ↓
Query expansion (variantes)
      ↓
Embedding + búsqueda híbrida
      ↓
Reranking
      ↓
Generación LLM
      ↓
Verificación (citas + consistency + scoring)
      ↓
Feedback (si --no-feedback no está activo)
```

## Notas
- Fecha de inicio: 2026-04-28
- Versión anterior: v1.1 (completada)
- Total commits: 4 (b37b8a2, 5111e8c, ea16178, a4e3d9b)
