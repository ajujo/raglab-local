# Guía de Uso — RAG-Lab

## ¿Qué es RAG-Lab?

RAG-Lab es un sistema de Generación Aumentada con Recuperación (RAG) que permite hacer consultas en lenguaje natural sobre documentos técnicos. Está diseñado para funcionar con múltiples documentos y ofrece una capa de verificación que garantiza la calidad de las respuestas.

---

## Comenzar

### 1. Entorno
```bash
conda activate rag-lab
```

### 2. Ingestar documentos
```bash
# Todos los documentos configurados en SOURCES
python -m rag_lab.cli ingest

# Un solo documento
python -m rag_lab.cli ingest --doc "otro_doc.md"
```

### 3. Consultar
```bash
# Consulta básica
python -m rag_lab.cli query "¿Qué es SDMX?"

# Con HyDE y query rewriting
python -m rag_lab.cli query "¿Qué es SDMX?" --hyde --rewrite

# Con métricas de rendimiento
python -m rag_lab.cli query "¿Qué es SDMX?" --profile

# Sin feedback (automatizado)
python -m rag_lab.cli query "¿Qué es SDMX?" --no-feedback
```

### 4. Modo Chat
```bash
python -m rag_lab.cli chat
```

---

## Comandos de Chat

| Comando | Uso | Descripción |
|---------|-----|-------------|
| `/help` | `/help` | Muestra ayuda |
| `/clear` | `/clear` | Limpia el historial |
| `/hyde` | `/hyde on` / `/hyde off` | Activa/desactiva HyDE |
| `/rewrite` | `/rewrite on` / `/rewrite off` | Activa/desactiva query rewriting |
| `/feedback` | `/feedback on` / `/feedback off` | Activa/desactiva feedback |
| `/docs` | `/docs doc1,doc2` | Filtra por documento |
| `/mode` | `/mode fast` | Cambia modo (fast, standard, hyde) |
| `/temp` | `/temp 0.1` | Cambia temperatura del LLM |
| `/topk` | `/topk 20` | Cambia número de chunks |
| `/quit` | `/quit` | Sale del chat |

---

## Flags de la CLI

| Flag | Propósito |
|------|-----------|
| `--hyde` | Activar HyDE (hipótesis con LLM) |
| `--rewrite` | Activar query rewriting |
| `--profile` | Mostrar métricas de rendimiento |
| `--no-feedback` | Desactivar prompt de feedback |
| `--cpu-embedding` | Ejecutar embedding en CPU |
| `--cpu-reranker` | Ejecutar reranker en CPU |
| `--fast` | Modo rápido (sin reranking) |
| `--top-k <n>` | Número de chunks a recuperar |

---

## Capa de Verificación

Cada respuesta incluye un bloque de verificación:

```
─────────────────────────────────────
Verificación de respuesta
  Fragmentos recuperados:
    [1] doc1 | Líneas 10-20  → 0.91 ████████████░░░
    [2] doc2 | Líneas 50-60  → 0.74 ██████████░░░░░
  Citas verificadas : 3/3 ✓
  Consistencia      : OK ✓
  Score de confianza: 0.87 — HIGH ✓
─────────────────────────────────────
```

**Niveles de confianza:**
- `HIGH` (≥ 0.75): Respuesta confiable
- `MEDIUM` (≥ 0.50): Respuesta aceptable
- `LOW` (< 0.50): Respuesta poco confiable

---

## Análisis de Feedback

```bash
python -m rag_lab.feedback.analyze_feedback
```

Muestra estadísticas de las respuestas evaluadas por los usuarios.

---

## Preguntas Frecuentes

**¿Cómo añado un nuevo documento?**
1. Añade la ruta a `SOURCES` en `config.py`
2. Ejecuta `python -m rag_lab.cli ingest --doc "nuevo_doc.md"`

**¿Qué es HyDE?**
HyDE (Hypothetical Document Embeddings) genera una respuesta hipotética con el LLM para mejorar la recuperación de chunks relevantes.

**¿Qué es query rewriting?**
Reformula la pregunta del usuario expandiendo siglas y usando terminología técnica para mejorar la búsqueda semántica.

**¿Cómo funciona el filtrado por documento?**
En modo chat, `/docs doc1,doc2` limita la búsqueda a esos documentos. `hybrid_search()` usa `where={"doc_id": {"$in": doc_ids}}` en ChromaDB.

---

## Estructura del Proyecto

```
RAG-Lab/
├── rag_lab/
│   ├── cli.py              # CLI principal
│   ├── cli_chat.py        # Modo chat
│   ├── config.py          # Configuración central
│   ├── ingest/            # Fase 1: Ingesta
│   ├── chunking/          # Fase 2: Chunking
│   ├── embedding/         # Fase 3: Embedding
│   ├── storage/           # Fase 4: Almacenamiento
│   ├── retrieval/         # Fase 5-6: Búsqueda
│   ├── generation/        # Fase 7: Generación
│   ├── verification/      # Fase 8: Verificación
│   ├── feedback/          # Fase 9: Feedback
│   └── performance/       # Métricas de rendimiento
├── tests/                  # Tests
├── data/                    # Datos procesados
└── storage/                 # Bases de datos
```

---

## Soporte

Para reportar bugs o pedir ayuda:
- GitHub Issues: https://github.com/ajujo/raglab-local/issues
- Documentación: README.md, AGENTS.md, QWEN.md
