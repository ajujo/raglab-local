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

---

# Plan de Implementación: Verification Layer (Capa de Verificación)

## Objetivo
Diseñar e implementar una capa de verificación que se ejecute DESPUÉS de que el LLM genere su respuesta, antes de devolvérsela al usuario. Esta capa tiene tres componentes: Verifier Layer, Self-consistency Check y Scoring.

## Estructura de Archivos
```
rag_lab/
└── verification/
    ├── __init__.py
    ├── verifier.py       # Componente 1: Verificación de citas
    ├── consistency.py    # Componente 2: Coherencia interna (Faithfulness)
    ├── scoring.py        # Componente 3: Puntuación de confianza
    └── pipeline.py       # Orquestación de los tres componentes
```

---

## Bloque 1: Estructura Base y Configuración
- **Tareas:**
  - Crear el directorio `rag_lab/verification/` y el archivo `__init__.py`.
  - Definir las clases base y los tipos de datos compartidos (ej. `CitationResult`, `VerificationResult`).
  - Añadir configuración en `config.py` para activar/desactivar el consistency check (`ENABLE_CONSISTENCY_CHECK = True`).
- **Tests:**
  - Verificar que los módulos se importan correctamente.
  - Probar que la configuración se lee adecuadamente.

## Bloque 2: Verifier Layer (Componente 1)
- **Tareas:**
  - Implementar la extracción de citas usando regex para el formato `[[N] Fuente: ... | Sección: ... | Líneas: ...]`.
  - Implementar la lógica de búsqueda en los chunks recuperados para verificar `doc_id`, `heading_path`, `line_start` y `line_end`.
  - Clasificar las citas en `VALID`, `PARTIAL` o `INVALID`.
  - Generar advertencias para las citas `INVALID`.
- **Tests:**
  - Probar la extracción de citas con diferentes formatos.
  - Verificar la clasificación de citas con chunks válidos, parciales e inválidos.
  - Probar la generación de advertencias.

## Bloque 3: Self-consistency Check (Componente 2)
- **Tareas:**
  - Implementar la función `check_consistency(response, chunks, llm_client)`.
  - Construir el prompt de evaluación que pide al LLM que responda en JSON con `has_unsupported_claims`, `has_contradictions`, `has_hallucinations` y `details`.
  - Parsear la respuesta JSON y manejar errores de formato.
  - Hacerlo opcional mediante el flag `enable_consistency_check`.
- **Tests:**
  - Probar la generación del prompt de evaluación.
  - Simular respuestas del LLM (mock) para verificar el parseo del JSON.
  - Verificar que el check se puede desactivar sin romper el pipeline.

## Bloque 4: Scoring y Pipeline (Componente 3 y Orquestación)
- **Tareas:**
  - Implementar el cálculo de los cuatro sub-scores: `citation_score`, `retrieval_score`, `consistency_score`, `coverage_score`.
  - Implementar la fórmula de puntuación final: `final_score = citation_score * 0.35 + retrieval_score * 0.30 + consistency_score * 0.25 + coverage_score * 0.10`.
  - Asignar el nivel de confianza (`HIGH`, `MEDIUM`, `LOW`).
  - Crear `pipeline.py` con la función `verify_and_score` que orquesta los tres componentes.
  - Integrar el output final con el bloque de metadatos de verificación.
- **Tests:**
  - Probar el cálculo de cada sub-score por separado.
  - Verificar la fórmula de puntuación final y la asignación de niveles.
  - Probar el pipeline completo con datos de entrada simulados.
  - Medir la latencia para asegurar que se mantiene por debajo de 2 segundos (sin consistency check).
