# RAG-Lab Update 1.2

## Descripción
Nueva fase de desarrollo para el sistema RAG-Lab. Este documento registra las mejoras, correcciones y nuevas funcionalidades que se implementarán en esta versión.

## Tareas Pendientes
- [ ] (Añadiremos las tareas a medida que las vayas proponiendo)

## Cambios Realizados
### 1. Rango de líneas en metadatos de chunks
- **Archivos modificados**: `rag_lab/chunking/splitter.py`, `rag_lab/generation/prompt_builder.py`
- **Descripción**: Se añadieron los campos `line_start` y `line_end` al dataclass `Chunk` y se propagaron por todo el pipeline de chunking.
- **Impacto**: Las citas ahora incluyen el rango de líneas del documento fuente (ej. `Líneas: 1543-1580`), mejorando drásticamente la precisión de las referencias.
- **Formato en prompt**: `[N] Fuente: <doc_id> | Sección: <heading_path> | Líneas: <line_start>-<line_end>`

### 2. Nuevo System Prompt genérico
- **Archivos modificados**: `rag_lab/config.py`
- **Descripción**: Se reemplazó el prompt específico de SDMX por un prompt genérico de "análisis documental" que funciona con cualquier tipo de documento.
- **Impacto**: El sistema ahora puede usarse con cualquier colección de documentos, no solo SDMX. Las reglas de citación ahora usan el formato con rango de líneas.

### 3. Actualización del User Prompt y Verificador
- **Archivos modificados**: `rag_lab/config.py`, `rag_lab/generation/verifier.py`, `rag_lab/storage/docstore.py`, `rag_lab/chunking/splitter.py`
- **Descripción**: 
  - Se actualizó `USER_PROMPT_TEMPLATE` con una estructura más clara y explícita sobre el formato de citas.
  - Se unificó el formato de citas en `[[N] Fuente: ... | Líneas: ...]` eliminando el soporte legacy.
  - Se actualizaron las tablas de `DocStore` para incluir `line_start` y `line_end`.
  - Se corrigió el cálculo de rangos de líneas en `_create_chunks` para que los índices de segmentos se mapeen correctamente a las líneas del documento original.
- **Por qué se actualizó**: 
  - El cálculo anterior de rangos de líneas era inexacto, lo que provocaba que algunos chunks tuvieran rangos superpuestos o incorrectos.
  - La corrección asegura que cada chunk tenga un rango de líneas preciso y no superpuesto con otros chunks.
- **Impacto**: Las citas ahora son 100% precisas en cuanto a localización. El código es más limpio y el mantenimiento es más sencillo al eliminar el soporte legacy.

## Notas
- Fecha de inicio: 2026-04-28
- Versión anterior: v1.1 (completada)
