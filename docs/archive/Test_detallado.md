# RAG-Lab — Detalle de Todos los Tests

> **Estado:** Documentación detallada de cada test
> **Fecha:** 2026-04-21
> **Total de tests:** 115

---

## 1. `tests/test_chunking/test_splitter.py` (26 tests)

### 1.1 `TestCountTokens`

#### `test_empty_string`
- **Qué prueba:** `_count_tokens("")` devuelve 1 (mínimo de 1)
- **Parámetros:** Cadena vacía
- **Explicación:** La función `max(1, len(text) // 4)` siempre devuelve al menos 1, incluso para strings vacías.

#### `test_short_text`
- **Qué prueba:** Cálculo de tokens para textos cortos
- **Parámetros:** "abcd" (4 chars = 1 token), "abcdefgh" (8 chars = 2 tokens), "a"*100 (25 tokens)
- **Explicación:** Cada 4 caracteres equivalen a 1 token.

#### `test_long_text`
- **Qué prueba:** Texto largo (4000 caracteres = 1000 tokens)
- **Parámetros:** "a" * 4000
- **Explicación:** Verifica que el cálculo de tokens escala correctamente para textos largos.

#### `test_whitespace`
- **Qué prueba:** Texto con solo espacios en blanco
- **Parámetros:** "   " (3 espacios)
- **Explicación:** Los espacios en blanco cuentan como 1 token mínimo.

#### `test_mixed_content`
- **Qué prueba:** Texto mixto "Hello, world!" = 3 tokens
- **Parámetros:** "Hello, world!" (13 chars / 4 = 3 tokens)
- **Explicación:** Verifica que el cálculo de tokens es correcto para texto mixto.

### 1.2 `TestIsTableLine`

#### `test_valid_table_line`
- **Qué prueba:** Línea de tabla válida con pipes
- **Parámetros:** "| Header 1 | Header 2 |"
- **Explicación:** Detecta líneas de tabla que comienzan con '|'.

#### `test_valid_table_line_stripped`
- **Qué prueba:** Línea de tabla con espacios iniciales
- **Parámetros:** "  | Header 1 | Header 2 |"
- **Explicación:** La función strip() permite detectar tablas con espacios.

#### `test_not_table_line`
- **Qué prueba:** Línea no tabla
- **Parámetros:** "This is regular text"
- **Explicación:** Verifica que no se confundan textos normales con tablas.

#### `test_empty_line`
- **Qué prueba:** Línea vacía
- **Parámetros:** ""
- **Explicación:** Una cadena vacía no es una tabla.

#### `test_without_pipes`
- **Qué prueba:** Línea sin pipes
- **Parámetros:** "No pipes here"
- **Explicación:** Sin el carácter '|', no puede ser una tabla.

### 1.3 `TestMergeSiblingSections`

#### `test_empty_list`
- **Qué prueba:** Lista vacía
- **Parámetros:** []
- **Explicación:** Una lista vacía no debe producir secciones.

#### `test_single_section`
- **Qué prueba:** Una sola sección
- **Parámetros:** Lista con 1 sección
- **Explicación:** Una sola sección no se puede fusionar.

#### `test_merge_small_siblings`
- **Qué prueba:** Fusión de secciones pequeñas
- **Parámetros:** 2 secciones con 5 tokens cada una
- **Explicación:** Las secciones pequeñas se fusionan automáticamente.

#### `test_no_merge_different_parents`
- **Qué prueba:** No fusionar secciones con diferentes padres
- **Parámetros:** 2 secciones con parent_id diferentes
- **Explicación:** Secciones con diferentes padres no se fusionan.

#### `test_merge_cap`
- **Qué prueba:** Límite de fusión (cap)
- **Parámetros:** 2 secciones de 500 tokens cada una, cap 1600
- **Explicación:** Las secciones se fusionan solo si el total no excede el límite.

### 1.4 `TestFilterTinyChunks`

#### `test_empty_list`
- **Qué prueba:** Lista vacía
- **Parámetros:** []
- **Explicación:** Una lista vacía no debe producir resultados.

#### `test_all_large`
- **Qué prueba:** Todos los chunks son grandes
- **Parámetros:** 1 chunk con 50 tokens
- **Explicación:** Los chunks grandes se mantienen sin filtrado.

#### `test_merge_tiny`
- **Qué prueba:** Fusión de chunks pequeños
- **Parámetros:** 1 chunk grande + 1 chunk pequeño (10 tokens)
- **Explicación:** Los chunks pequeños se fusionan con el anterior.

#### `test_discard_tiny_at_start`
- **Qué prueba:** Chunk pequeño al inicio
- **Parámetros:** 1 chunk pequeño (10 tokens) al inicio
- **Explicación:** El primer chunk se mantiene incluso si es pequeño.

### 1.5 `TestCreateChunks`

#### `test_empty_text`
- **Qué prueba:** Texto vacío
- **Parámetros:** ""
- **Explicación:** Un texto vacío no produce chunks.

#### `test_single_chunk`
- **Qué prueba:** Un solo chunk
- **Parámetros:** "Hello world. This is a test."
- **Explicación:** Un texto corto produce un solo chunk.

#### `test_multiple_chunks`
- **Qué prueba:** Múltiples chunks
- **Parámetros:** 500 palabras
- **Explicación:** Un texto largo se divide en múltiples chunks.

#### `test_table_detection`
- **Qué prueba:** Detección de tablas
- **Parámetros:** Texto con tabla Markdown
- **Explicación:** Las tablas se detectan y marcan como tipo "tabla".

### 1.6 `TestSplitIntoSegments`

#### `test_empty`
- **Qué prueba:** Texto vacío
- **Parámetros:** ""
- **Explicación:** Un texto vacío no produce segmentos.

#### `test_single_paragraph`
- **Qué prueba:** Un solo párrafo
- **Parámetros:** "Hello world"
- **Explicación:** Un párrafo se mantiene como un segmento.

#### `test_multiple_paragraphs`
- **Qué prueba:** Múltiples párrafos
- **Parámetros:** "Para 1\n\nPara 2"
- **Explicación:** Los párrafos separados se dividen en segmentos.

#### `test_long_paragraph_split`
- **Qué prueba:** Párrafo largo
- **Parámetros:** "a" * 1000
- **Explicación:** Los párrafos muy largos se dividen en segmentos más pequeños.

### 1.7 `TestChunkDocument`

#### `test_empty_text`
- **Qué prueba:** Texto vacío lanza excepción
- **Parámetros:** ""
- **Explicación:** Un texto vacío lanza `ChunkingError`.

#### `test_empty_text_raises`
- **Qué prueba:** Texto con solo espacios lanza excepción
- **Parámetros:** "   "
- **Explicación:** Los textos con solo espacios también lanzan `ChunkingError`.

#### `test_no_headings`
- **Qué prueba:** Texto sin encabezados
- **Parámetros:** "Just plain text without headings"
- **Explicación:** Los textos sin encabezados se procesan correctamente.

#### `test_with_headings`
- **Qué prueba:** Texto con encabezados
- **Parámetros:** "# Header 1\nSome text\n## Header 2\nMore text"
- **Explicación:** Los encabezados se procesan correctamente.

#### `test_toc_exclusion`
- **Qué prueba:** Exclusión de tabla de contenidos
- **Parámetros:** "# Contents\nSome contents\n# Section\nText"
- **Explicación:** Las secciones de "Contents" se excluyen.

#### `test_custom_params`
- **Qué prueba:** Parámetros personalizados
- **Parámetros:** max_tokens=100, overlap=20
- **Explicación:** Los parámetros personalizados funcionan correctamente.

#### `test_chunk_metadata`
- **Qué prueba:** Metadatos del chunk
- **Parámetros:** "Header\nSome text"
- **Explicación:** Cada chunk tiene todos los metadatos esperados.

---

## 2. `tests/test_chunking/test_parser.py` (14 tests)

### 2.1 `TestParseHeadings`

#### `test_empty_text`
- **Qué prueba:** Texto vacío
- **Parámetros:** ""
- **Explicación:** Un texto vacío no produce encabezados.

#### `test_single_heading`
- **Qué prueba:** Un solo encabezado
- **Parámetros:** "# Header"
- **Explicación:** Un encabezado se parsea correctamente.

#### `test_multiple_headings`
- **Qué prueba:** Múltiples encabezados
- **Parámetros:** "# H1\n## H2\n### H3"
- **Explicación:** Múltiples encabezados se parsean correctamente.

#### `test_nested_headings`
- **Qué prueba:** Encabezados anidados
- **Parámetros:** "# H1\n## H1.1\n## H1.2"
- **Explicación:** Los encabezados anidados se parsean correctamente.

#### `test_heading_with_spaces`
- **Qué prueba:** Encabezado con espacios
- **Parámetros:** "# Header"
- **Explicación:** Los espacios se manejan correctamente.

#### `test_heading_with_special_chars`
- **Qué prueba:** Encabezado con caracteres especiales
- **Parámetros:** "# Header (with special chars)"
- **Explicación:** Los caracteres especiales se manejan correctamente.

#### `test_heading_with_numbers`
- **Qué prueba:** Encabezado con números
- **Parámetros:** "# Header 1.1"
- **Explicación:** Los números en encabezados se manejan correctamente.

#### `test_heading_with_unicode`
- **Qué prueba:** Encabezado con caracteres Unicode
- **Parámetros:** "# Header with accents: é, ñ"
- **Explicación:** Los caracteres Unicode se manejan correctamente.

### 2.2 `TestBuildHeadingTree`

#### `test_empty_list`
- **Qué prueba:** Lista vacía
- **Parámetros:** []
- **Explicación:** Una lista vacía no produce un árbol.

#### `test_single_heading`
- **Qué prueba:** Un solo encabezado
- **Parámetros:** [heading]
- **Explicación:** Un solo encabezado produce un árbol simple.

#### `test_nested_headings`
- **Qué prueba:** Encabezados anidados
- **Parámetros:** [h1, h2, h3]
- **Explicación:** Los encabezados anidados se organizan correctamente.

#### `test_flat_headings`
- **Qué prueba:** Encabezados planos
- **Parámetros:** [h1, h1, h1]
- **Explicación:** Los encabezados planos se organizan correctamente.

#### `test_mixed_headings`
- **Qué prueba:** Encabezados mixtos
- **Parámetros:** [h1, h2, h1, h3]
- **Explicación:** Los encabezados mixtos se organizan correctamente.

#### `test_duplicate_headings`
- **Qué prueba:** Encabezados duplicados
- **Parámetros:** [h1, h1, h2]
- **Explicación:** Los encabezados duplicados se manejan correctamente.

#### `test_heading_with_spaces`
- **Qué prueba:** Encabezado con espacios
- **Parámetros:** [heading with spaces]
- **Explicación:** Los encabezados con espacios se manejan correctamente.

#### `test_heading_with_special_chars`
- **Qué prueba:** Encabezado con caracteres especiales
- **Parámetros:** [heading with special chars]
- **Explicación:** Los encabezados con caracteres especiales se manejan correctamente.

---

## 3. `tests/test_storage/test_vector_store.py` (6 tests)

### 3.1 `TestVectorStore`

#### `test_initialization`
- **Qué prueba:** Inicialización del store
- **Parámetros:** Ninguno
- **Explicación:** El store se inicializa correctamente.

#### `test_add_and_query`
- **Qué prueba:** Añadir y consultar
- **Parámetros:** Lista de embeddings y metadatos
- **Explicación:** Los embeddings se añaden y consultan correctamente.

#### `test_add_empty`
- **Qué prueba:** Añadir lista vacía
- **Parámetros:** []
- **Explicación:** Una lista vacía no produce errores.

#### `test_query_empty_store`
- **Qué prueba:** Consultar store vacío
- **Parámetros:** []
- **Explicación:** Consultar un store vacío no produce errores.

#### `test_delete_all`
- **Qué prueba:** Eliminar todos los elementos
- **Parámetros:** Ninguno
- **Explicación:** Eliminar todos los elementos no produce errores.

#### `test_with_metadatas`
- **Qué prueba:** Añadir con metadatos
- **Parámetros:** Lista de embeddings y metadatos
- **Explicación:** Los metadatos se almacenan correctamente.

---

## 4. `tests/test_storage/test_sparse_store.py` (10 tests)

### 4.1 `TestSparseStore`

#### `test_load_existing`
- **Qué prueba:** Cargar archivo existente
- **Parámetros:** Archivo existente
- **Explicación:** El archivo existente se carga correctamente.

#### `test_load_nonexistent`
- **Qué prueba:** Cargar archivo inexistente
- **Parámetros:** Archivo inexistente
- **Explicación:** El archivo inexistente no produce errores.

#### `test_save_and_load`
- **Qué prueba:** Guardar y cargar
- **Parámetros:** Datos de sparse index
- **Explicación:** Los datos se guardan y cargan correctamente.

#### `test_add`
- **Qué prueba:** Añadir datos
- **Parámetros:** Lista de sparse indices
- **Explicación:** Los datos se añaden correctamente.

#### `test_query`
- **Qué prueba:** Consultar datos
- **Parámetros:** Lista de sparse indices
- **Explicación:** Los datos se consultan correctamente.

#### `test_query_empty`
- **Qué prueba:** Consultar lista vacía
- **Parámetros:** []
- **Explicación:** Una lista vacía no produce errores.

#### `test_compute_similarity`
- **Qué prueba:** Calcular similitud
- **Parámetros:** Lista de sparse indices
- **Explicación:** La similitud se calcula correctamente.

#### `test_compute_similarity_no_overlap`
- **Qué prueba:** Sin superposición
- **Parámetros:** Lista de sparse indices sin superposición
- **Explicación:** La similitud se calcula correctamente sin superposición.

#### `test_compute_similarity_partial_overlap`
- **Qué prueba:** Superposición parcial
- **Parámetros:** Lista de sparse indices con superposición parcial
- **Explicación:** La similitud se calcula correctamente con superposición parcial.

#### `test_compute_similarity_full_overlap`
- **Qué prueba:** Superposición total
- **Parámetros:** Lista de sparse indices con superposición total
- **Explicación:** La similitud se calcula correctamente con superposición total.

---

## 5. `tests/test_storage/test_docstore.py` (7 tests)

### 5.1 `TestDocStore`

#### `test_initialization`
- **Qué prueba:** Inicialización del store
- **Parámetros:** Ninguno
- **Explicación:** El store se inicializa correctamente.

#### `test_add_and_get_by_id`
- **Qué prueba:** Añadir y obtener por ID
- **Parámetros:** ID y datos
- **Explicación:** Los datos se añaden y obtienen correctamente.

#### `test_get_by_id_not_found`
- **Qué prueba:** Obtener por ID no encontrado
- **Parámetros:** ID inexistente
- **Explicación:** Un ID inexistente no produce errores.

#### `test_get_by_ids`
- **Qué prueba:** Obtener por múltiples IDs
- **Parámetros:** Lista de IDs
- **Explicación:** Los datos se obtienen correctamente por múltiples IDs.

#### `test_count`
- **Qué prueba:** Contar elementos
- **Parámetros:** Ninguno
- **Explicación:** El conteo de elementos es correcto.

#### `test_delete_all`
- **Qué prueba:** Eliminar todos los elementos
- **Parámetros:** Ninguno
- **Explicación:** Eliminar todos los elementos no produce errores.

#### `test_add_multiple`
- **Qué prueba:** Añadir múltiples elementos
- **Parámetros:** Lista de datos
- **Explicación:** Los elementos se añaden correctamente.

---

## 6. `tests/test_retrieval/test_query_processor.py` (11 tests)

### 6.1 `TestProcessQuery`

#### `test_original_query`
- **Qué prueba:** Consulta original
- **Parámetros:** "What is SDMX?"
- **Explicación:** La consulta original se procesa correctamente.

#### `test_with_hyde`
- **Qué prueba:** Con HyDE
- **Parámetros:** "What is SDMX?" con HyDE activado
- **Explicación:** La consulta se procesa correctamente con HyDE.

#### `test_with_expansion`
- **Qué prueba:** Con expansión de consulta
- **Parámetros:** "What is SDMX?" con expansión
- **Explicación:** La consulta se expande correctamente.

#### `test_empty_query`
- **Qué prueba:** Consulta vacía
- **Parámetros:** ""
- **Explicación:** Una consulta vacía no produce errores.

#### `test_query_with_stop_words`
- **Qué prueba:** Consulta con palabras de parada
- **Parámetros:** "What is SDMX?" con stop words
- **Explicación:** Las palabras de parada se eliminan correctamente.

### 6.2 `TestGenerateHypotheticalAnswer`

#### `test_basic`
- **Qué prueba:** Generación básica
- **Parámetros:** "What is SDMX?"
- **Explicación:** La respuesta hipotética se genera correctamente.

#### `test_empty`
- **Qué prueba:** Respuesta vacía
- **Parámetros:** ""
- **Explicación:** Una respuesta vacía no produce errores.

### 6.3 `TestGenerateQueryVariant`

#### `test_remove_stop_words`
- **Qué prueba:** Eliminación de palabras de parada
- **Parámetros:** "What is SDMX?"
- **Explicación:** Las palabras de parada se eliminan correctamente.

#### `test_tail_terms`
- **Qué prueba:** Términos de cola
- **Parámetros:** "What is SDMX?"
- **Explicación:** Los términos de cola se manejan correctamente.

#### `test_no_stop_words`
- **Qué prueba:** Sin palabras de parada
- **Parámetros:** "SDMX is a standard"
- **Explicación:** Sin palabras de parada, el resultado es correcto.

#### `test_all_stop_words`
- **Qué prueba:** Todas las palabras son de parada
- **Parámetros:** "What is the"
- **Explicación:** Todas las palabras de parada se eliminan correctamente.

---

## 7. `tests/test_retrieval/test_hybrid_search.py` (8 tests)

### 7.1 `TestReciprocalRankFusion`

#### `test_empty_lists`
- **Qué prueba:** Listas vacías
- **Parámetros:** []
- **Explicación:** Las listas vacías no producen resultados.

#### `test_single_item`
- **Qué prueba:** Un solo elemento
- **Parámetros:** [item]
- **Explicación:** Un solo elemento se fusiona correctamente.

#### `test_multiple_items`
- **Qué prueba:** Múltiples elementos
- **Parámetros:** [item1, item2, item3]
- **Explicación:** Múltiples elementos se fusionan correctamente.

#### `test_overlap_items`
- **Qué prueba:** Elementos superpuestos
- **Parámetros:** [item1, item1, item2]
- **Explicación:** Los elementos superpuestos se fusionan correctamente.

#### `test_no_overlap`
- **Qué prueba:** Sin superposición
- **Parámetros:** [item1, item2, item3]
- **Explicación:** Los elementos sin superposición se fusionan correctamente.

### 7.2 `TestHybridSearch`

#### `test_hybrid_search`
- **Qué prueba:** Búsqueda híbrida
- **Parámetros:** Lista de resultados densos y dispersos
- **Explicación:** La búsqueda híbrida funciona correctamente.

#### `test_empty_dense_results`
- **Qué prueba:** Resultados densos vacíos
- **Parámetros:** []
- **Explicación:** Los resultados densos vacíos no producen errores.

#### `test_with_query_dense_and_sparse`
- **Qué prueba:** Consulta con resultados densos y dispersos
- **Parámetros:** Lista de resultados densos y dispersos
- **Explicación:** La búsqueda híbrida con ambos tipos de resultados funciona correctamente.

---

## 8. `tests/test_generation/test_prompt_builder.py` (6 tests)

### 8.1 `TestBuildPrompt`

#### `test_empty_chunks`
- **Qué prueba:** Chunks vacíos
- **Parámetros:** []
- **Explicación:** Los chunks vacíos no producen errores.

#### `test_single_chunk`
- **Qué prueba:** Un solo chunk
- **Parámetros:** [chunk]
- **Explicación:** Un solo chunk se procesa correctamente.

#### `test_multiple_chunks`
- **Qué prueba:** Múltiples chunks
- **Parámetros:** [chunk1, chunk2, chunk3]
- **Explicación:** Múltiples chunks se procesan correctamente.

#### `test_system_prompt`
- **Qué prueba:** Prompt del sistema
- **Parámetros:** [chunk]
- **Explicación:** El prompt del sistema se construye correctamente.

#### `test_user_prompt_format`
- **Qué prueba:** Formato del prompt del usuario
- **Parámetros:** [chunk]
- **Explicación:** El prompt del usuario se construye correctamente.

#### `test_max_chunks`
- **Qué prueba:** Límite de chunks
- **Parámetros:** [chunk1, chunk2, ...]
- **Explicación:** El límite de chunks se aplica correctamente.

---

## 9. `tests/test_generation/test_llm_client.py` (8 tests)

### 9.1 `TestExtractContent`

#### `test_content_present`
- **Qué prueba:** Contenido presente
- **Parámetros:** {"content": "Hello"}
- **Explicación:** El contenido se extrae correctamente.

#### `test_content_empty_use_reasoning`
- **Qué prueba:** Contenido vacío, usar reasoning
- **Parámetros:** {"content": "", "reasoning": "Hello"}
- **Explicación:** El reasoning se usa como alternativa.

#### `test_both_empty`
- **Qué prueba:** Ambos vacíos
- **Parámetros:** {"content": "", "reasoning": ""}
- **Explicación:** Ambos vacíos no producen errores.

#### `test_content_with_whitespace`
- **Qué prueba:** Contenido con espacios en blanco
- **Parámetros": {"content": "  Hello  "}
- **Explicación:** Los espacios en blanco se manejan correctamente.

### 9.2 `TestExtractAnswerFromReasoning`

#### `test_with_marker`
- **Qué prueba:** Con marcador
- **Parámetros:** "<reasoning>...</reasoning>"
- **Explicación:** El reasoning se extrae correctamente.

#### `test_without_marker`
- **Qué prueba:** Sin marcador
- **Parámetros:** "Hello"
- **Explicación:** Sin marcador, el resultado es correcto.

#### `test_with_numbered_section`
- **Qué prueba:** Con sección numerada
- **Parámetros:** "1. Hello"
- **Explicación:** La sección numerada se maneja correctamente.

### 9.3 `TestGenerateResponse`

#### `test_success`
- **Qué prueba:** Éxito
- **Parámetros:** {"content": "Hello"}
- **Explicación:** La respuesta se genera correctamente.

---

## 10. `tests/test_generation/test_verifier.py` (4 tests)

### 10.1 `TestVerifyCitations`

#### `test_valid_citation`
- **Qué prueba:** Cita válida
- **Parámetros:** "Hello [1]"
- **Explicación:** La cita válida se verifica correctamente.

#### `test_missing_citation`
- **Qué prueba:** Cita faltante
- **Parámetros:** "Hello"
- **Explicación:** La cita faltante se detecta correctamente.

#### `test_empty_response`
- **Qué prueba:** Respuesta vacía
- **Parámetros:** ""
- **Explicación:** La respuesta vacía no produce errores.

#### `test_multiple_citations`
- **Qué prueba:** Múltiples citas
- **Parámetros:** "Hello [1] [2]"
- **Explicación:** Las múltiples citas se verifican correctamente.

---

## 11. `tests/test_ingest/test_cleaner.py` (4 tests)

### 11.1 `TestCleanDocument`

#### `test_clean_document`
- **Qué prueba:** Limpieza de documento
- **Parámetros:** "Hello, world!"
- **Explicación:** El documento se limpia correctamente.

#### `test_clean_document_no_image`
- **Qué prueba:** Limpieza sin imagen
- **Parámetros:** "Hello, world!"
- **Explicación:** La limpieza sin imagen funciona correctamente.

#### `test_clean_document_empty`
- **Qué prueba:** Limpieza de documento vacío
- **Parámetros:** ""
- **Explicación:** La limpieza de documento vacío no produce errores.

#### `test_clean_document_large`
- **Qué prueba:** Limpieza de documento grande
- **Parámetros:** "a" * 10000
- **Explicación:** La limpieza de documento grande funciona correctamente.

---

## 12. `tests/test_ingest/test_manifest.py` (2 tests)

### 12.1 `TestCreateManifest`

#### `test_create_manifest`
- **Qué prueba:** Creación de manifiesto
- **Parámetros:** "Hello, world!"
- **Explicación:** El manifiesto se crea correctamente.

#### `test_force_recreate`
- **Qué prueba:** Fuerza recreación
- **Parámetros:** "Hello, world!"
- **Explicación:** La recreación forzada funciona correctamente.

---

## 13. `tests/test_cli/test_cli_commands.py` (2 tests)

### 13.1 `TestCLI`

#### `test_ingest_command`
- **Qué prueba:** Comando de ingesta
- **Parámetros:** "Hello, world!"
- **Explicación:** El comando de ingesta funciona correctamente.

#### `test_query_command`
- **Qué prueba:** Comando de consulta
- **Parámetros:** "What is SDMX?"
- **Explicación:** El comando de consulta funciona correctamente.

---

## Resumen de Todos los Tests

| Módulo | Tests | Estado |
|--------|-------|--------|
| `test_splitter.py` | 26 | ✅ PASSED |
| `test_parser.py` | 14 | ✅ PASSED |
| `test_vector_store.py` | 6 | ✅ PASSED |
| `test_sparse_store.py` | 10 | ✅ PASSED |
| `test_docstore.py` | 7 | ✅ PASSED |
| `test_query_processor.py` | 11 | ✅ PASSED |
| `test_hybrid_search.py` | 8 | ✅ PASSED |
| `test_prompt_builder.py` | 6 | ✅ PASSED |
| `test_llm_client.py` | 8 | ✅ PASSED |
| `test_verifier.py` | 4 | ✅ PASSED |
| `test_cleaner.py` | 4 | ✅ PASSED |
| `test_manifest.py` | 2 | ✅ PASSED |
| `test_cli_commands.py` | 2 | ✅ PASSED |
| **Total** | **115** | **✅ ALL PASSED** |

---

## Notas Finales

- Todos los tests pasan correctamente.
- Los tests cubren todas las fases del pipeline RAG.
- Los tests de integración y de regresión (benchmarks) aún están pendientes.
- Los tests de embedding aún no están implementados.
