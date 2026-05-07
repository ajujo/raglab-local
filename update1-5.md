# RAG-Lab Update 1.5

## Descripción
Nueva fase de desarrollo para el sistema RAG-Lab. Este documento registra las mejoras, correcciones y nuevas funcionalidades que se implementan en esta versión.

## Contexto
- Fecha de inicio: 2026-05-06
- Versión objetivo: v1.5
- Dependencias: Ninguna nueva requerida

## Resumen de lo hecho hoy (2026-05-06)

### Gestor de Documentos (`rag_lab/doc_manager/`)
Se creó un módulo independiente para gestionar documentos ingestados en el sistema RAG:

**Archivos creados:**
- `rag_lab/doc_manager/__init__.py` — Paquete del gestor
- `rag_lab/doc_manager/doc_store.py` — Almacenamiento SQLite + lógica de etiquetas
- `rag_lab/doc_manager/cli.py` — CLI con Typer + Rich (comandos: list, add, delete, tag, untag, search, collections, info, migrate, interactive)
- `rag_lab/doc_manager/__main__.py` — Punto de entrada principal
- `rag_lab/doc_manager/interactive.py` — Interfaz interactiva con menú de opciones
- `tests/test_doc_manager/test_doc_manager.py` — 10 tests pasando

**Características implementadas:**
- Detección de duplicados por hash MD5 al añadir documentos
- Borrado de chunks de ChromaDB al eliminar un documento
- Gestión de etiquetas/colecciones para organizar documentos
- Comando `migrate` para importar documentos existentes desde `ingested.jsonl`
- Modo interactivo por defecto (`python -m rag_lab.doc_manager`)
- Modo CLI con argumentos (`python -m rag_lab.doc_manager list`)
- Opción "volver" / "n" para cancelar cualquier prompt y volver al menú
- Opción [9] "Borrar todos" con confirmación doble (escribir "BORRAR")

### Mejoras en Scoring y Pipeline de Verificación
**`scoring.py` — `retrieval_score` con normalización min-max:**
- Se reemplazó el enfoque anterior (sigmoid sobre logits) por normalización min-max.
- Los logits crudos del reranker se escalan a `[0, 1]`: mejor chunk → 1.0, peor → 0.0.
- Funciona independientemente de si los logits son positivos o negativos.
- Se usan solo los top-3 scores normalizados para calcular el promedio.

**`pipeline.py` — Barras de visualización y warning adaptativo:**
- Las barras de scores se muestran en escala `0-10` con ancho fijo de 10 caracteres.
- El warning de "relevancia baja" se activa cuando el spread (max - min) de los scores es `< 0.5`, indicando que el reranker no pudo distinguir relevancia entre chunks.

**Tests actualizados:**
- `test_raw_scores_fix.py`: Assertions actualizadas para reflejar la normalización min-max (ej. `0.4 < retrieval_score < 0.55`).
- `test_verification.py`: `test_score_all_valid` espera `coverage_score == 1.0` (antes era diferente).

### Mejoras en `DocStore`
- Se añadió `count_chunks(doc_id)` para contar chunks por documento
- Se añadió `close()` para cerrar conexiones

### Commits del día
1. `495ddeb` — feat: añadir gestor de documentos con CLI independiente
2. `1ef49cd` — feat: añadir __main__.py y comando migrate al gestor de documentos
3. `0c81ab8` — feat: añadir modo interactivo con menú de opciones al gestor de documentos
4. `ac056ed` — refactor: modo interactivo por defecto, CLI con argumentos
5. `088765b` — feat: añadir opción volver en borrar y borrar todos con confirmación doble
6. `7a23af2` — fix: aceptar 'volver' y 'n' en selección de documentos

## Tareas Pendientes de Updates Anteriores
- [ ] Re-ingesta completa (bloqueada por falta de VRAM — SGLang ocupando GPU)
- [ ] Verificar que los fixes de Update 1.4 estén aplicados en la base de datos

## Tareas Pendientes para continuar mañana
- [ ] Completar re-ingesta del documento principal (detener SGLang y ejecutar `python -m rag_lab.cli ingest`)
- [ ] Verificar recuperación de "Maintenance Agencies" con la pregunta de ejemplo
- [ ] Integrar el gestor de documentos con el pipeline RAG (filtrado por doc_id en búsqueda)
- [ ] Añadir soporte para múltiples documentos en el sistema RAG

## Notas
- El gestor es independiente del pipeline RAG principal
- Usa SQLite para metadatos y ChromaDB para chunks
- Las etiquetas permiten organizar y filtrar documentos
- La detección de duplicados por hash evita ingestas duplicadas
- Total tests: 10 (doc_manager) + 225 (resto del proyecto) = 235 tests
