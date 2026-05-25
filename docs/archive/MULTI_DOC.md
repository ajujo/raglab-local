# RAG-Lab — Soporte Multi-Documento

## Propósito

Permitir la ingesta y consulta de múltiples documentos simultáneamente, en lugar de un solo documento fuente.

---

## Cómo Funciona

### Flujo de Ingesta Multi-Documento

```
SOURCES = [doc1.md, doc2.md, doc3.md]
  ↓
Para cada documento:
  1. Limpiar (clean_document)
  2. Chunking (chunk_document)
  3. Embedding (encode_chunks)
  4. Almacenamiento (VectorStore, SparseStore, DocStore)
  ↓
Todos los chunks de todos los documentos en los mismos almacenes
```

### Flujo de Consulta Multi-Documento

```
Consulta del usuario
  ↓
Búsqueda híbrida en todos los almacenes
  ↓
Reranking de resultados de múltiples documentos
  ↓
Generación de respuesta con citas de múltiples fuentes
```

---

## Cambios Requeridos

### 1. `config.py`

```python
# Fuentes de documentos — lista de rutas
SOURCES = [
    SOURCES_DIR / "Notas_Tecnicas_SDMX_2.1.md",
    SOURCES_DIR / "doc2.md",
    SOURCES_DIR / "doc3.md",
]
```

### 2. `cli.py` — Comando `ingest`

```python
@app.command()
def ingest(
    doc: str = typer.Option(None, "--doc", help="Path to a single document"),
    force: bool = typer.Option(False, "--force", help="Force re-ingestion"),
) -> None:
    if doc is None:
        # Ingestir TODOS los documentos en SOURCES
        for source_path in SOURCES:
            _process_single_document(source_path, force)
    else:
        # Ingestir un solo documento
        _process_single_document(Path(doc), force)
```

### 3. `chunking/splitter.py`

Cada chunk lleva un `doc_id` que identifica de qué documento proviene:

```python
chunks = chunk_document(text, doc_id="doc1")
# Cada chunk tiene: chunk_id, doc_id, text, heading_path, tipo, posicion_relativa
```

### 4. `storage/` — Almacenes compartidos

Todos los documentos comparten los mismos almacenes:
- `VectorStore`: Todos los embeddings densos en una sola colección
- `SparseStore`: Todos los índices dispersos en un solo JSON
- `DocStore`: Todos los chunks en una sola base SQLite

### 5. `retrieval/` — Búsqueda multi-documento

La búsqueda híbrida recupera chunks de todos los documentos:

```python
results = hybrid_search(
    query,
    vector_store,
    sparse_store,
    doc_store,
    query_dense=...,
    query_sparse=...,
    top_k=30,
)
# Los resultados incluyen chunks de múltiples documentos
```

### 6. `generation/` — Citas multi-documento

El verificador de citas debe manejar referencias de múltiples fuentes:

```python
verified = verify_citations(response, unique_results)
# Las citas pueden provenir de cualquier documento en SOURCES
```

---

## Ejemplo de Uso

```bash
# Ingestir todos los documentos en SOURCES
python -m rag_lab.cli ingest

# Ingestir un solo documento
python -m rag_lab.cli ingest --doc "mi_documento.md"

# Consultar el sistema (busca en todos los documentos)
python -m rag_lab.cli query "¿Qué es SDMX?"
```

---

## Beneficios

1. **Escalabilidad:** Añadir nuevos documentos es tan fácil como agregar una ruta a `SOURCES`.
2. **Búsqueda unificada:** Una consulta busca en todos los documentos simultáneamente.
3. **Citas precisas:** El sistema puede citar de múltiples fuentes en una sola respuesta.

---

## Consideraciones

- **Rendimiento:** Más documentos = más chunks = más tiempo de ingesta.
- **Memoria:** Los almacenes crecen con cada documento.
- **Colisiones de ID:** Cada chunk necesita un `chunk_id` único (se usa hash MD5 del texto).
- **Metadata:** Cada chunk almacena su `doc_id` para rastrear el origen.
