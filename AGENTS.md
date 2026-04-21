# RAG-Lab: SDMX Technical Notes — Guía para Agentes de IA

## 1. Descripción del proyecto

Este proyecto implementa un sistema **RAG** (Retrieval-Augmented Generation) sobre el documento *SDMX Standards: Technical Notes v2.1*. El objetivo es permitir consultas en lenguaje natural sobre especificaciones técnicas de formatos de intercambio de datos estadísticos (SDMX-ML, SDMX-EDI, modelos de información, definiciones de estructura de datos).

**Stack tecnológico:**
- Python 3.x dentro del entorno conda `rag-lab`
- GPU: RTX 5090 (32GB VRAM)
- Embeddings: BGE-M3 (dense + sparse)
- Vector store: ChromaDB (HNSW, cosine similarity)
- Reranker: BGE-reranker-v2-m3 (cross-encoder)
- LLM: Qwen 3.6 35B A3B via SGLang
- CLI: Typer + Rich

---

## 2. Reglas de ejecución obligatorias

### Entorno Conda

```bash
# SIEMPRE activar el entorno conda antes de ejecutar cualquier comando Python
conda activate rag-lab

# Para instalar dependencias nuevas:
pip install <paquete>   # dentro del entorno rag-lab
```

**NUNCA** ejecutes `python` o `pip` fuera del entorno `rag-lab`. Este es el único entorno disponible.

### GPU

Todos los modelos ML (BGE-M3, reranker) deben correr en GPU (`device='cuda'`). Verificar disponibilidad:

```bash
conda activate rag-lab
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Estructura del proyecto

```
RAG-Lab/
├── AGENTS.md                          ← Este archivo (instrucciones para agentes)
├── Notas_Tecnicas_SDMX_2.1.md         ← Documento fuente (~3.3 MB, 2816 líneas)
├── .opencode/plans/rag-sdmx-plan.md   ← Plan de implementación detallado
│
├── rag_lab/                           ← Paquete principal (a implementar)
│   ├── __init__.py
│   ├── config.py                      ← Configuración central
│   ├── cli.py                         ← CLI con Typer
│   │
│   ├── ingest/                        ← Fase 1: Ingesta y limpieza
│   │   ├── cleaner.py                 ← Elimina imágenes base64
│   │   └── manifest.py                ← Genera ingested.jsonl
│   │
│   ├── chunking/                      ← Fase 2: Chunking semántico
│   │   ├── parser.py                  ← Parsea headings Markdown
│   │   └── splitter.py                ← Divide en chunks jerárquicos
│   │
│   ├── embedding/                     ← Fase 3: Embedding
│   │   └── encoder.py                 ← BGE-M3 dense + sparse
│   │
│   ├── storage/                       ← Fase 4: Almacenamiento
│   │   ├── vector_store.py            ← ChromaDB wrapper
│   │   ├── sparse_store.py            ← Sparse index
│   │   └── docstore.py                ← SQLite docstore
│   │
│   ├── retrieval/                     ← Fases 5-6: Retrieval híbrido
│   │   ├── query_processor.py         ← Query expansion + HyDE
│   │   ├── hybrid_search.py           ← Búsqueda + RRF fusion
│   │   └── reranker.py                ← Cross-encoder reranking
│   │
│   └── generation/                    ← Fase 7: Generación LLM
│       ├── prompt_builder.py          ← Construye prompts
│       ├── llm_client.py              ← Wrapper OpenAI-compatible
│       └── verifier.py                ← Verifica citas del LLM
│
├── data/                              ← Datos procesados (generados)
│   ├── ingested.jsonl                 ← Manifiesto de documentos
│   ├── cleaned/                       ← Documentos limpios
│   └── chunks.jsonl                   ← Chunks generados
│
├── storage/                           ← Bases de datos (generadas)
│   ├── chroma_db/                     ← ChromaDB persistente
│   ├── sparse_index.json              ← Índices sparse
│   └── docstore.sqlite                ← Docstore SQLite
│
├── tests/                             ← Tests (a implementar)
│   ├── conftest.py                    ← Fixtures compartidos
│   ├── test_ingest/
│   ├── test_chunking/
│   ├── test_embedding/
│   ├── test_storage/
│   ├── test_retrieval/
│   ├── test_generation/
│   ├── test_cli/
│   └── integration/
│
├── requirements.txt                   ← Dependencias Python
├── pyproject.toml                     ← Configuración del proyecto
├── .env.example                       ← Variables de entorno
└── .env                               ← Variables de entorno (no commit)
```

---

## 4. Comandos principales

### Setup inicial
```bash
conda activate rag-lab
pip install torch FlagEmbedding chromadb numpy huggingface_hub pytest
cp .env.example .env   # y editar con el endpoint del LLM
```

### Pipeline completo
```bash
conda activate rag-lab
python -m rag_lab.cli ingest      # Ingesta + chunking + embedding + almacenamiento
python -m rag_lab.cli query "tu pregunta aquí"   # Hacer una consulta
```

### Opciones de consulta
```bash
python -m rag_lab.cli query "Pregunta" --hyde     # Con HyDE activado
python -m rag_lab.cli query "Pregunta" --fast      # Sin reranker (más rápido)
python -m rag_lab.cli query "Pregunta" --top-k 10  # Más chunks recuperados
```

### Tests
```bash
conda activate rag-lab
pytest tests/                          # Todos los tests
pytest tests/test_chunking/            # Solo tests de chunking
pytest tests/ -v --tb=short            # Detallado con stack trace
pytest tests/ -m "gpu"                 # Solo tests que requieren GPU
```

---

## 5. Excepciones personalizadas del sistema

El código debe usar estas excepciones para errores específicos:

| Excepción | Cuándo se usa |
|-----------|--------------|
| `RAGLabError` | Error base del sistema |
| `DocumentIngestionError` | Documento corrupto/inválido |
| `ChunkingError` | Problemas al dividir chunks |
| `EmbeddingError` | OOM GPU, fallo en embedding |
| `RetrievalError` | Fallo en búsqueda/retrieval |
| `LLMConnectionError` | Servidor LLM no disponible |

---

## 6. Logging

Todo el sistema debe usar logging centralizado:

```python
import logging
logger = logging.getLogger("rag_lab")

# Niveles:
logger.debug("Detalle técnico para debugging")
logger.info("Operación exitosa (ingesta completada, etc.)")
logger.warning("Advertencia no crítica (tabla grande, chunk truncado)")
logger.error("Error que afecta una operación específica")
logger.critical("Error fatal del sistema")
```

---

## 7. Convenciones de código

### Estilo
- **PEP 8** para Python
- **Type hints** en todas las funciones públicas
- **Docstrings** con Google style en cada módulo y clase pública
- **f-strings** para interpolación de strings

### Ejemplo de función:
```python
def chunk_document(text: str, max_tokens: int = 400, overlap: int = 80) -> list[dict]:
    """Divide un documento en chunks semánticos.

    Args:
        text: Texto completo del documento limpio.
        max_tokens: Tokens máximos por chunk (default: 400).
        overlap: Tokens de superposición entre chunks (default: 80).

    Returns:
        Lista de dicts con keys: chunk_id, doc_id, text, heading_path, tipo, posicion_relativa.

    Raises:
        ChunkingError: Si el texto está vacío o es inválido.
    """
    if not text or not text.strip():
        raise ChunkingError("El documento de entrada está vacío")
    ...
```

---

## 8. Flujo de trabajo recomendado para agentes

### Para implementar una nueva funcionalidad:

1. **Leer el plan** en `.opencode/plans/rag-sdmx-plan.md` para entender la arquitectura
2. **Identificar la fase** correspondiente (Fase 1-7)
3. **Crear los tests primero** (TDD) en `tests/test_<modulo>/`
4. **Implementar el código** en `rag_lab/<modulo>/`
5. **Ejecutar tests**: `pytest tests/test_<modulo>/ -v`
6. **Verificar que todo corre en GPU**

### Para hacer cambios:

1. Siempre verificar que los tests existentes pasan antes de modificar código
2. Añadir tests para cualquier bug fix nuevo
3. Actualizar este AGENTS.md si se cambian comandos o estructura

---

## 9. Configuración central (config.py)

Todos los parámetros deben estar en `rag_lab/config.py`:

```python
# Ejemplo de configuración esperada:
DATA_DIR = "data"
STORAGE_DIR = "storage"
CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP = 80
CHUNK_MIN_TOKENS = 50
EMBEDDING_BATCH_SIZE = 64
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5
LLM_BASE_URL = "http://localhost:30000/v1"
LLM_MODEL = "qwen-3.6-35b-a3b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
HYDE_ENABLED = False
FAST_MODE = False
```

---

## 10. Referencias rápidas

| Recurso | Ubicación |
|---------|-----------|
| Plan detallado | `.opencode/plans/rag-sdmx-plan.md` |
| Documento fuente | `Notas_Tecnicas_SDMX_2.1.md` |
| Configuración LLM | `.env` (crear desde `.env.example`) |
| Dependencias | `requirements.txt` (al crearlo) |
| Documentación SDMX | Contenido en `Notas_Tecnicas_SDMX_2.1.md` |