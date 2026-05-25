# RAG-Lab — Contrato de Frontmatter Markdown (v1.19)

Este documento describe el contrato canónico de metadatos YAML para documentos Markdown en RAG-Lab.

---

## Propósito

El frontmatter YAML es el mecanismo oficial para asociar metadatos de clasificación a un documento antes de ingestarlo. Permite:

- Identificar un documento de forma estable (`doc_id`).
- Clasificar el documento por dominio, tipo de fuente, idioma y versión.
- Asignar etiquetas explícitas que persisten en la base de datos.
- Filtrar el corpus en retrieval sin modificar el algoritmo de ranking.

El frontmatter es **exclusivo de documentos Markdown**. JSON está reservado para artefactos internos del sistema (benchmarks, auditorías, configuraciones). No existe soporte para datasets tabulares (CSV, Parquet, DuckDB) ni para loaders automáticos de PDF/DOCX/HTML.

---

## Ejemplo completo recomendado

```yaml
---
doc_id: sdmx_user_guide_2_1
title: SDMX User Guide 2.1
domain: sdmx
source_type: manual
language: en
version: "2.1"
tags:
  - sdmx
  - technical_notes
  - metadata
---

# SDMX User Guide 2.1

Contenido del documento...
```

---

## Ejemplo mínimo aceptado

Solo `doc_id` es estrictamente obligatorio. Los demás campos generan WARNs si están ausentes, pero no bloquean la ingesta:

```yaml
---
doc_id: sdmx_glossary
---

# SDMX Glossary

Contenido del documento...
```

Este documento pasará validación sin errores pero emitirá WARNs por `title`, `domain`, `source_type` y `language` ausentes. Con `--strict`, la ingesta quedaría bloqueada.

---

## Ejemplo de documento antiguo (compatible)

Un documento sin frontmatter es **válido** (no ERROR), pero genera un WARN `frontmatter_missing`. Es compatible con todos los sistemas existentes: se ingesta, se indexa y se recupera normalmente.

```markdown
# SDMX Glossary (legacy)

Contenido del documento sin frontmatter...
```

En este caso `doc_id` se deriva del nombre del archivo (comportamiento heredado). Los campos `domain`, `source_type`, `language` y `version` quedan vacíos en la base de datos.

---

## Campos del contrato

### `doc_id` — obligatorio

Identificador único del documento en el sistema. Debe ser estable, reproducible y no contener espacios ni caracteres especiales. Se usa como clave primaria en la tabla `documents`.

```yaml
doc_id: sdmx_user_guide_2_1
```

- Ausente → ERROR `frontmatter_missing_doc_id`. La ingesta no procede.
- Se persiste en `documents.doc_id` (PRIMARY KEY).

### `title` — recomendado

Título legible del documento. Si está ausente, se usa el primer H1 encontrado en el contenido.

```yaml
title: SDMX User Guide 2.1
```

- Ausente → WARN `frontmatter_missing_title`.
- Se persiste en `documents.title`.
- Visible en `docs show` (sección Classification) y `diagnose --doc-id`.

### `domain` — recomendado

Dominio temático del documento. Valor en minúsculas.

```yaml
domain: sdmx
```

- Ausente → WARN `frontmatter_missing_domain`.
- Se persiste en `documents.domain`.
- Genera automáticamente el tag derivado `domain:sdmx`.
- Filtrable en retrieval con `FilterSpec(domain="sdmx")`.

### `source_type` — recomendado

Tipo de fuente del documento. Valores habituales: `manual`, `spec`, `training`, `glossary`, `notes`.

```yaml
source_type: manual
```

- Ausente → WARN `frontmatter_missing_source_type`.
- Se persiste en `documents.source_type`.
- Genera automáticamente el tag derivado `source_type:manual`.
- Filtrable en retrieval con `FilterSpec(source_type="manual")`.

### `language` — recomendado

Código de idioma ISO 639-1 del documento (`en`, `es`, `fr`, …).

```yaml
language: en
```

- Ausente → WARN `frontmatter_missing_language`.
- Se persiste en `documents.language`.
- Genera automáticamente el tag derivado `lang:en` (prefijo `lang:`, no `language:`).
- Filtrable en retrieval con `FilterSpec(language="en")`.

### `version` — opcional

Versión del estándar o documento. Siempre entre comillas si contiene puntos, para evitar que YAML lo interprete como número flotante.

```yaml
version: "2.1"
```

- Ausente → sin WARN.
- Se persiste en `documents.version`.
- Genera automáticamente el tag derivado `version:2.1`.
- Filtrable en retrieval con `FilterSpec(version="2.1")`.

### `tags` — opcional

Lista de etiquetas explícitas asignadas al documento. Deben ser strings no vacíos.

```yaml
tags:
  - sdmx
  - technical_notes
  - metadata
```

- `tags` no es lista → ERROR `frontmatter_tags_not_list`.
- Elemento no string → ERROR `frontmatter_tag_not_string`.
- Elemento vacío → WARN `frontmatter_tag_empty`.
- Elemento solo espacios → WARN `frontmatter_tag_whitespace`.
- Elemento duplicado → WARN `frontmatter_tag_duplicate`.
- Se persisten en la tabla `document_tags`.
- Filtrables en retrieval con `FilterSpec(tags_include=["sdmx"])`.

---

## Tags explícitos vs tags derivados

RAG-Lab distingue dos tipos de tags en un documento ingestado:

### Tags explícitos

Los definidos en el campo `tags:` del frontmatter. Se almacenan literalmente en `document_tags`.

```yaml
tags:
  - sdmx
  - metadata
  - smoke
```

### Tags derivados

Generados automáticamente durante la ingesta a partir de los campos de clasificación. El mapeo es:

| Campo | Tag derivado | Ejemplo |
|---|---|---|
| `domain: sdmx` | `domain:sdmx` | `domain:sdmx` |
| `source_type: manual` | `source_type:manual` | `source_type:manual` |
| `language: en` | `lang:en` | `lang:en` |
| `version: "2.1"` | `version:2.1` | `version:2.1` |

Los tags derivados se auto-importan en `document_tags` junto con los explícitos durante la ingesta. Permiten filtrar por clasificación usando la misma infraestructura de tags existente.

Para ver ambos tipos tras la ingesta:

```bash
rag-lab docs show <doc_id>
# Sección Classification:
#   tags (explicit)   metadata, sdmx, smoke
#   tags (derived)    domain:sdmx, lang:en, source_type:manual, version:2.1
```

---

## Campos prohibidos

Los campos `dataset` y `dataset_id` están **explícitamente prohibidos** en el frontmatter:

```yaml
# INCORRECTO — esto producirá un ERROR:
dataset: sdmx_codelist
dataset_id: cl_freq_001
```

Error generado: `frontmatter_scope_violation` (ERROR). La ingesta no procede.

**Razón:** RAG-Lab es un sistema RAG sobre documentos Markdown. No existe soporte para datos tabulares, datasets, CSV, Parquet ni DuckDB. El campo `dataset` no tiene semántica en este sistema y su presencia indica un error de clasificación del documento. Si en el futuro hace falta procedencia más fina que `source_type`, se evaluará un campo `source` separado — pero eso es una decisión futura, no una ruta de escape para datos tabulares.

---

## Cómo validar un documento

### Validación normal

```bash
rag-lab docs validate path/to/doc.md
```

- Sale con código 0 y `✓ OK` si no hay errores.
- Sale con código 0 y lista de WARNs si hay advertencias.
- Sale con código 1 si hay ERRORs (bloquea la ingesta).

### Validación estricta (WARNs bloquean)

```bash
rag-lab docs validate --strict path/to/doc.md
```

Útil para pipelines CI o antes de ingesta masiva. Trata WARNs como ERRORs.

### Inspeccionar estructura y frontmatter

```bash
rag-lab docs inspect path/to/doc.md
```

Muestra el frontmatter parseado con todos los campos (incluyendo `derived_tags`), estructura de encabezados, estimación de tokens y chunks, y resultado de validación.

Ejemplo de salida:

```
Inspect: doc.md

  Frontmatter
  doc_id                   sdmx_user_guide_2_1
  title                    SDMX User Guide 2.1
  domain                   sdmx
  source_type              manual
  language                 en
  version                  2.1
  tags                     sdmx, technical_notes, metadata
  derived_tags             domain:sdmx, source_type:manual, lang:en, version:2.1

  Structure
  file_size                42.3 KB
  total_lines              1847
  total_tokens (~)         9823
  estimated_chunks (~)     14
  validation               OK
```

---

## Cómo comprobar los metadatos tras la ingesta

### Ver clasificación en la base de datos

```bash
rag-lab docs show <doc_id>
```

Muestra las secciones Classification (title, domain, source_type, language, version, tags explícitos y derivados) y Technical (path, hash, timestamps, chunks).

### Ver metadatos en diagnose

```bash
rag-lab diagnose --doc-id <doc_id> --query "tu pregunta" --explain
```

Muestra el bloque `Document metadata for '<doc_id>'` con todos los campos de clasificación y los tags, y ejecuta una query de prueba filtrada a ese documento.

### Verificar consistencia de stores

```bash
rag-lab reconcile --check
```

Comprueba que DocStore, ChromaDB, FTS5 y Sparse BLOBs están sincronizados. A partir de v1.19, también verifica la consistencia de tags derivados: si un documento tiene `domain=sdmx` en `documents.domain` pero no el tag `domain:sdmx` en `document_tags`, lo reporta como inconsistencia.

---

## Flujo completo de incorporación de un documento nuevo

```bash
# 1. Validar el Markdown y su frontmatter
rag-lab docs validate path/to/doc.md

# 2. Inspeccionar estructura (frontmatter, tokens, chunks estimados)
rag-lab docs inspect path/to/doc.md

# 3. Preview de chunks sin escribir nada
rag-lab docs preview-chunks path/to/doc.md

# 4. Ingestar
rag-lab ingest --doc path/to/doc.md

# 5. Verificar que los metadatos se han persistido correctamente
rag-lab docs show <doc_id>
rag-lab diagnose --doc-id <doc_id> --query "pregunta de prueba" --explain

# 6. Confirmar integridad de stores
rag-lab reconcile --check
rag-lab doctor
```

---

## Filtrado por metadatos en retrieval

Los campos de clasificación del frontmatter se resuelven internamente como tags derivados antes de la búsqueda. El algoritmo de ranking (RRF, MMR, reranker) no cambia — solo se restringe el pool de candidatos.

Ejemplos de uso programático:

```python
from rag_lab.retrieval.filters import FilterSpec

# Solo documentos del dominio sdmx
FilterSpec(domain="sdmx")

# Solo manuales en inglés
FilterSpec(source_type="manual", language="en")

# Versión específica
FilterSpec(version="2.1")

# Combinado con tags explícitos
FilterSpec(domain="sdmx", tags_include=["technical_notes"])
```

Internamente, `FilterSpec(domain="sdmx")` se traduce a `tags_include=["domain:sdmx"]` antes de la resolución de doc_ids. No existe ninguna ruta de código especial por campo de clasificación — todo pasa por el mismo sistema de tags.

---

## Notas sobre evolución futura

- **Campo `source`:** Si en el futuro hace falta registrar la procedencia más fina de un documento (URL de origen, repositorio, organismo emisor), se evaluará añadir un campo `source` separado de `source_type`. Esta decisión está pendiente de casos de uso reales; no existe ni está planificada en v1.19.

- **Tipos de `source_type`:** Los valores habituales actuales son `manual`, `spec`, `training`, `glossary`, `notes`. No existe una lista cerrada — cualquier string en minúsculas es aceptado.

- **Idiomas adicionales:** `language` acepta cualquier código ISO 639-1 de dos caracteres. El corpus actual usa principalmente `en` y `es`.
