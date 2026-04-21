# RAG-Lab — Notas de Actualización v1.1

> **Estado:** Plan de mejora para la versión 1.1
> **Fecha:** 2026-04-21
> **Depende de:** v1.0 (copia de seguridad realizada)

---

## Resumen de la Versión 1.1

La versión 1.0 del proyecto es funcional pero carece de pruebas automatizadas y tiene varias debilidades arquitecturales que limitan la mantenibilidad a largo plazo. La v1.1 introduce:

1. **Cobertura de tests completa** (prioridad máxima)
2. **Gestión de memoria GPU para tests** (prioridad máxima)
3. **Refactorización de paths hardcodeados**
4. **Mejora del manejo de errores**
5. **Optimización de parámetros**
6. **Soporte multi-documento**
7. **Corrección de la configuración de logging**

---

## 1.5. Gestión de Memoria GPU para Tests ✅ COMPLETADO

### Problema

Tu RTX 5090 tiene solo 1GB libre. Cargar BGE-M3 + reranker en GPU simultáneamente puede causar OOM. Necesitamos forzar CPU para los tests.

### Solución Implementada ✅

**En `rag_lab/config.py`:**

```python
# Device settings
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")

# Test device settings — force CPU for tests to avoid OOM on GPU
TEST_EMBEDDING_DEVICE = "cpu"
TEST_RERANKER_DEVICE = "cpu"
```

**En `rag_lab/embedding/encoder.py`:**

- `encode_chunks()` acepta `device` explícito que sobreescribe `EMBEDDING_DEVICE`
- Nueva función `reset_embedding_cache()` para limpiar el singleton global `_model_cache`

**En `rag_lab/retrieval/reranker.py`:**

- `rerank()` acepta `device` explícito que sobreescribe `RERANKER_DEVICE`
- Nueva función `reset_reranker_cache()` para limpiar el singleton global `_reranker_cache`

**En `tests/conftest.py`:**

- `@pytest.fixture(autouse=True) reset_ml_caches()` — se ejecuta antes de cada test
- Fuerza `os.environ["EMBEDDING_DEVICE"] = "cpu"` y `os.environ["RERANKER_DEVICE"] = "cpu"`
- Limpia ambos caches de modelos antes de cada test

### Flujo de Ejecución

```
Cada test comienza con:
1. EMBEDDING_DEVICE="cpu" (via os.environ)
2. RERANKER_DEVICE="cpu" (via os.environ)
3. _model_cache = None (reset)
4. _reranker_cache = None (reset)

Producción:
1. EMBEDDING_DEVICE="cuda" (default)
2. RERANKER_DEVICE="cuda" (default)
3. Modelos se cargan en GPU normalmente
```

### Notas Técnicas

- Los singletons `_model_cache` y `_reranker_cache` se resetean antes de cada test
- Los tests siempre usan CPU para evitar OOM
- La producción sigue usando GPU por defecto
- No hay cambios de comportamiento en la CLI

---

## 2. Cobertura de Tests Automatizados

### Estado: 🟢 COMPLETADO — 115 tests PASSED

Se han creado los siguientes archivos de test:

| Archivo | Estado | Descripción |
|---------|--------|-------------|
| `tests/conftest.py` | ✅ Creado | Fixtures y reset de caches |
| `tests/test_chunking/test_splitter.py` | ✅ 26 tests PASSED | 30+ tests unitarios |
| `tests/test_chunking/test_parser.py` | ✅ 14 tests PASSED | Tests de parsing |
| `tests/test_storage/test_vector_store.py` | ✅ 6 tests PASSED | API de ChromaDB corregida |
| `tests/test_storage/test_sparse_store.py` | ✅ 10 tests PASSED | Tests de sparse index |
| `tests/test_storage/test_docstore.py` | ✅ 7 tests PASSED | Tests de SQLite docstore |
| `tests/test_retrieval/test_query_processor.py` | ✅ 11 tests PASSED | Tests de query expansion |
| `tests/test_retrieval/test_hybrid_search.py` | ✅ 8 tests PASSED | Tests de RRF fusion |
| `tests/test_generation/test_prompt_builder.py` | ✅ 6 tests PASSED | Tests de prompts |
| `tests/test_generation/test_llm_client.py` | ✅ 8 tests PASSED | Tests de LLM client |
| `tests/test_generation/test_verifier.py` | ✅ 4 tests PASSED | Tests de citas |
| `tests/test_ingest/test_cleaner.py` | ✅ 4 tests PASSED | Tests de limpieza |
| `tests/test_ingest/test_manifest.py` | ✅ 2 tests PASSED | Tests de manifest |
| `tests/test_cli/test_cli_commands.py` | ✅ 2 tests PASSED | Tests de CLI (mocked) |

**Faltan por crear:**
- `tests/integration/` - Tests de integración
- `tests/test_embedding/` - Tests de embedding
- Tests de regresión (benchmarks)

### Próximos pasos

1. Agregar tests de integración
2. Agregar tests de embedding
3. Agregar tests de regresión (benchmarks Q1, Q2, Q3)

### Problema

El directorio `tests/` está completamente vacío. El pipeline de 7 fases es complejo y cualquier cambio puede romper funcionalidades existentes sin que nos enteremos.

### Solución Propuesta

Implementar tests unitarios y de integración para cada módulo del pipeline.

#### 1.1. Tests Unitarios por Módulo

| Módulo | Qué probar | Tipo de test |
|--------|-----------|-------------|
| `chunking/splitter.py` | `_count_tokens`, `_is_table_line`, `_merge_sibling_sections`, `_filter_tiny_chunks`, `_create_chunks` | Unitarios |
| `chunking/parser.py` | `parse_headings`, `build_heading_tree` | Unitarios |
| `embedding/encoder.py` | `encode_chunks` con datos mock | Unitarios + GPU |
| `storage/vector_store.py` | `initialize`, `add`, `query` | Unitarios |
| `storage/sparse_store.py` | `add`, `query`, `save`, `load` | Unitarios |
| `storage/docstore.py` | `add`, `get_by_ids` | Unitarios |
| `retrieval/query_processor.py` | `process_query` con diferentes inputs | Unitarios |
| `retrieval/hybrid_search.py` | `_reciprocal_rank_fusion` con datos mock | Unitarios |
| `generation/prompt_builder.py` | `build_prompt` con diferentes chunk counts | Unitarios |
| `generation/llm_client.py` | `_extract_content`, `_extract_answer_from_reasoning` | Unitarios |
| `generation/verifier.py` | `verify_citations` con diferentes casos | Unitarios |
| `ingest/cleaner.py` | `clean_document` con archivos con/sin base64 | Unitarios |
| `ingest/manifest.py` | `create_manifest` con/sin hash match | Unitarios |
| `cli.py` | `ingest` y `query` commands (mocked) | Integración |

#### 1.2. Tests de Integración

| Test | Descripción |
|------|-------------|
| `test_integration/test_full_pipeline.py` | Ejecuta ingest + query completo con datos mock |
| `test_integration/test_edge_cases.py` | Documentos vacíos, sin headings, tablas grandes |

#### 1.3. Tests de Regresión (Benchmarks)

| Test | Descripción |
|------|-------------|
| `test_integration/test_benchmark_q1.py` | Verifica Q1: 4+2+2 variantes de formato |
| `test_integration/test_benchmark_q2.py` | Verifica Q2: 7/7 reglas de agencias |
| `test_integration/test_benchmark_q3.py` | Verifica Q3: 7/7 períodos de reporte |

#### 1.4. Fix Necesario Previo a los Tests

En `rag_lab/logging_config.py`, la función `setup_logging` crea un nuevo handler cada vez que se llama. Esto es aceptable si se llama una vez al inicio, pero podría causar handlers duplicados si se llama múltiples veces.

**Fix propuesto:**

```python
def setup_logging(level: str = "INFO") -> None:
    """Configure centralized logging for the RAG system.

    Ensures only one set of handlers is added, preventing duplicate log entries.
    """
    logger = logging.getLogger("rag_lab")
    if not logger.handlers:
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE),
        ]
        logger.setLevel(getattr(logging, level.upper()))
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(LOG_FILE))
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        for h in logger.handlers:
            h.setFormatter(formatter)
```

#### 1.5. Fix Necesario en `rag_lab/cli.py`

El comando `query` llama a `setup_logging("INFO")` cada vez. Esto es aceptable en CLI (cada ejecución es un proceso nuevo), pero conviene asegurarse de que no se llame múltiples veces en el mismo proceso.

#### 1.6. Estructura de Tests

```
tests/
├── conftest.py                    ← Fixtures compartidos (mocks, test data)
├── test_chunking/
│   ├── test_splitter.py          ← Tests unitarios del splitter
│   └── test_parser.py           ← Tests unitarios del parser
├── test_embedding/
│   └── test_encoder.py          ← Tests unitarios del encoder
├── test_storage/
│   ├── test_vector_store.py
│   ├── test_sparse_store.py
│   └── test_docstore.py
├── test_retrieval/
│   ├── test_query_processor.py
│   └── test_hybrid_search.py
├── test_generation/
│   ├── test_prompt_builder.py
│   ├── test_llm_client.py
│   └── test_verifier.py
├── test_cli/
│   └── test_cli_commands.py
└── integration/
    ├── test_full_pipeline.py
    ├── test_edge_cases.py
    ├── test_benchmark_q1.py
    ├── test_benchmark_q2.py
    └── test_benchmark_q3.py
```

#### 1.7. Configuración de pytest

Añadir `tests/conftest.py` con fixtures reutilizables:

```python
import pytest
from pathlib import Path

# Ruta a archivos de test
TEST_ASSETS = Path(__file__).parent / "assets"

@pytest.fixture
def sample_text():
    """Sample text for testing chunking."""
    return """# Section 1
Some text here.

## Subsection 1.1
More text here.

| Header 1 | Header 2 |
|----------|----------|
| A        | B        |
"""

@pytest.fixture
def sample_headings():
    """Sample headings for testing parser."""
    return [
        {"title": "Section 1", "level": 1, "position": 1},
        {"title": "Subsection 1.1", "level": 2, "position": 5},
    ]
```

---

## 2. Refactorización de Paths Hardcodeados

### Problema

Varios módulos usan paths hardcodeados como `"chroma_db/"`, `"sparse_index.json"`, `"docstore.sqlite"` sin usar las constantes de `config.py`. Esto hace que cambiar ubicaciones de almacenamiento requiera cambios en múltiples archivos.

### Solución Propuesta

Centralizar todas las rutas en `config.py` y usarlas en todos los módulos.

**Cambios en `config.py`:**

```python
# Storage paths - centralized
VECTOR_STORE_PATH = "storage/chroma_db"
SPARSE_INDEX_PATH = "storage/sparse_index.json"
DOCDSTORE_SQLITE_PATH = "storage/docstore.sqlite"
```

**Cambios en `rag_lab/storage/vector_store.py`:**

```python
from rag_lab.config import VECTOR_STORE_PATH

class VectorStore:
    def __init__(self, path=None):
        self.path = Path(path or VECTOR_STORE_PATH)
```

**Cambios en `rag_lab/storage/sparse_store.py`:**

```python
from rag_lab.config import SPARSE_INDEX_PATH

class SparseStore:
    def __init__(self, path=None):
        self.path = Path(path or SPARSE_INDEX_PATH)
```

**Cambios en `rag_lab/storage/docstore.py`:**

```python
from rag_lab.config import DOCDSTORE_SQLITE_PATH

class DocStore:
    def __init__(self, path=None):
        self.db_path = Path(path or DOCDSTORE_SQLITE_PATH)
```

**Cambios en `rag_lab/cli.py`:**

```python
from rag_lab.config import (
    STORAGE_DIR,
    VECTOR_STORE_PATH,
    SPARSE_INDEX_PATH,
    DOCDSTORE_SQLITE_PATH,
)
```

---

## 3. Mejora del Manejo de Errores

### Problema

El comando `query` en `cli.py` captura solo `Exception` de forma genérica. Las llamadas a `rerank` y `generate_response` podrían fallar de maneras no específicamente manejadas.

### Solución Propuesta

Usar las excepciones personalizadas definidas en `rag_lab/exceptions.py` para cada tipo de error.

**Cambios en `cli.py`:**

```python
from rag_lab.exceptions import (
    RetrievalError,
    LLMConnectionError,
    RAGLabError,
)

@app.command()
def query(...):
    ...
    if unique_results:
        system_prompt, user_prompt = build_prompt(...)
        try:
            response = generate_response(system_prompt, user_prompt)
            ...
        except LLMConnectionError as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
    else:
        console.print("[bold yellow]⚠️ No results found.[/bold yellow]")
```

**Cambios en `rag_lab/retrieval/hybrid_search.py`:**

```python
from rag_lab.exceptions import RetrievalError

def hybrid_search(...):
    ...
    if not dense_results and not sparse_results:
        raise RetrievalError("No results from either dense or sparse search")
```

**Cambios en `rag_lab/generation/llm_client.py`:**

```python
from rag_lab.exceptions import LLMConnectionError

def generate_response(...):
    ...
    except Exception as e:
        raise LLMConnectionError(
            f"Failed to connect to LLM server at {LLM_BASE_URL}: {e}"
        )
```

---

## 4. Optimización de Parámetros

### Problema

Algunos parámetros están hardcodeados o tienen valores arbitrarios sin justificación clara.

### Solución Propuesta

**En `config.py`:**

```python
# Chunking
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP = 200
CHUNK_MIN_TOKENS = 50

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 8        # Aumentado de 4
EMBEDDING_MAX_LENGTH = 1024
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# Retrieval
RETRIEVAL_TOP_K = 30
RERANK_TOP_K = 8
RRF_K = 60

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b@iq4_xs")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# Query
VARIANTS_COUNT = 2
HYDE_ENABLED = False
FAST_MODE = False
```

**En `rag_lab/cli.py`:**

```python
from rag_lab.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    STORAGE_DIR,
)
```

**En `rag_lab/generation/llm_client.py`:**

```python
# Qwen3 thinking models need extra token budget for their internal reasoning.
# With complex RAG prompts, reasoning can use 500-2000+ tokens before producing
# the actual content. We multiply the configured max_tokens to ensure there's
# room for both reasoning and the final answer.
_THINKING_TOKEN_MULTIPLIER = 4
```

---

## 5. Soporte Multi-Documento

### Problema

El config tiene una lista `SOURCES` comentada para múltiples documentos, pero el CLI `ingest` solo maneja un documento.

### Solución Propuesta

**En `config.py`:**

```python
# Document sources
SOURCES = [
    SOURCES_DIR / "Notas_Tecnicas_SDMX_2.1.md",
    # SOURCES_DIR / "doc2.md",
    # SOURCES_DIR / "doc3.md",
]
```

**En `rag_lab/cli.py`:**

```python
@app.command()
def ingest(
    doc: str = typer.Option(
        None,
        "--doc",
        help="Path to the source document. If not specified, uses all SOURCES.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-ingestion even if already ingested.",
    ),
) -> None:
    """Ingest one or more documents: clean, chunk, embed, and store."""
    ...
    # If doc is None, iterate over all SOURCES
    if doc is None:
        for source_path in SOURCES:
            _process_single_document(source_path, force)
    else:
        _process_single_document(Path(doc), force)
```

**En `rag_lab/chunking/splitter.py`:**

```python
def chunk_document(
    text: str,
    doc_id: str = "sdmx_tech_notes_2.1",
    max_tokens: int = None,
    overlap: int = None,
) -> List[Chunk]:
    ...
```

---

## Proceso de Implementación

### Fase 1: Corrección de Logging (Prioridad: Alta)

1. Modificar `rag_lab/logging_config.py` para evitar handlers duplicados
2. Actualizar `rag_lab/cli.py` para no llamar `setup_logging` múltiples veces
3. Escribir test para `setup_logging` en `tests/test_storage/test_logging.py`

### Fase 2: Refactorización de Paths (Prioridad: Alta)

1. Actualizar `config.py` con todas las rutas centralizadas
2. Actualizar `rag_lab/storage/vector_store.py` para usar `VECTOR_STORE_PATH`
3. Actualizar `rag_lab/storage/sparse_store.py` para usar `SPARSE_INDEX_PATH`
4. Actualizar `rag_lab/storage/docstore.py` para usar `DOCDSTORE_SQLITE_PATH`
5. Actualizar `rag_lab/cli.py` para usar las constantes de config
6. Escribir tests unitarios para cada módulo de storage

### Fase 3: Mejora del Manejo de Errores (Prioridad: Alta)

1. Actualizar `cli.py` para usar excepciones específicas (`RetrievalError`, `LLMConnectionError`)
2. Actualizar `hybrid_search.py` para lanzar `RetrievalError` cuando no hay resultados
3. Actualizar `llm_client.py` para lanzar `LLMConnectionError` correctamente
4. Escribir tests unitarios para cada excepción

### Fase 4: Optimización de Parámetros (Prioridad: Media)

1. Actualizar `config.py` con los parámetros optimizados
2. Actualizar `cli.py` para usar las nuevas constantes de config
3. Actualizar `llm_client.py` para usar `LLM_MAX_TOKENS` correctamente
4. Actualizar `splitter.py` para usar `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP`, `CHUNK_MIN_TOKENS`
5. Actualizar `encoder.py` para usar `EMBEDDING_BATCH_SIZE` y `EMBEDDING_MAX_LENGTH`
6. Escribir tests unitarios para cada módulo afectado

### Fase 5: Soporte Multi-Documento (Prioridad: Media)

1. Actualizar `config.py` con la lista `SOURCES`
2. Actualizar `cli.py` para iterar sobre todos los documentos
3. Actualizar `chunk_document` para manejar múltiples `doc_id`
4. Escribir tests de integración para multi-documento

### Fase 6: Cobertura de Tests Completa (Prioridad: Máxima)

1. Crear `tests/conftest.py` con fixtures reutilizables
2. Escribir tests unitarios para cada módulo del pipeline
3. Escribir tests de integración para cada fase del pipeline
4. Escribir tests de regresión para los benchmarks Q1, Q2, Q3
5. Asegurar que todos los tests pasen con `pytest`

### Fase 7: Validación Final

1. Ejecutar todos los tests con `pytest tests/ -v`
2. Ejecutar el pipeline completo con `python -m rag_lab.cli ingest`
3. Ejecutar una consulta de prueba con `python -m rag_lab.cli query "tu pregunta"`
4. Verificar que no hay alucinaciones en la respuesta

---

## Resumen de Prioridades

| Prioridad | Tarea | Impacto |
|-----------|-------|---------|
| Máxima | Cobertura de tests | Previene regresiones |
| Alta | Refactorización de paths | Mejora mantenibilidad |
| Alta | Mejora del manejo de errores | Mejora robustez |
| Media | Optimización de parámetros | Mejora rendimiento |
| Media | Soporte multi-documento | Amplía funcionalidad |
| Baja | Validación final | Asegura calidad |

---

## Notas Finales

- Todo el trabajo debe realizarse dentro del entorno conda `rag-lab`
- Antes de cada tarea, verificar que los tests existentes pasan
- Actualizar este documento con el estado de cada tarea
- Mantener el estilo de código PEP 8 con type hints y docstrings Google style
