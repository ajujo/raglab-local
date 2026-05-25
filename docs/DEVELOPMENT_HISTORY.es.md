# RAG-Lab — Historia de desarrollo

Este documento explica cómo y por qué evolucionó el sistema desde su primera versión hasta
el estado actual en v1.19.1. Cubre decisiones de diseño, resultados de benchmarks y lecciones
aprendidas en cada sprint.

---

## MVP — v1.0: RAG básico funcional

El MVP estableció el pipeline completo de extremo a extremo sobre un único documento Markdown.

**Componentes implementados:**
- Ingesta con limpieza básica (strip de imágenes base64)
- Chunking por headings H2+ con respeto de tablas
- Embedding denso con BGE-M3 (dense + sparse simultáneos)
- Almacenamiento en ChromaDB (vectores) y JSON (sparse index)
- Retrieval con RRF (Reciprocal Rank Fusion) sobre dense + sparse
- Generación con cliente OpenAI-compatible local

**Lo que faltaba en v1.0:**
- Tests automatizados
- Transacciones de ingesta (un fallo a mitad dejaba el sistema inconsistente)
- Validación de la calidad del documento de entrada
- Soporte multi-documento
- Métricas de evaluación

---

## v1.1: Diversidad documental y tests

La limitación más importante de v1.0 para un corpus real era que los documentos grandes
acaparaban todos los slots del top-5 en retrieval. Con un solo documento largo, todos los
chunks recuperados venían del mismo archivo.

**MMR (Maximal Marginal Relevance):** Se añadió diversidad en la selección de chunks para que
el top-K representara distintas secciones y documentos. El A/B benchmark de diversidad mostró
hybrid_mmr +14.6pp R@5 respecto al retrieval sin diversidad.

**Suite de tests automatizados:** Primera suite de tests con cobertura de los módulos
principales. A partir de v1.1 todos los cambios se validan antes de merge.

**Multi-doc:** Soporte para ingestar y consultar múltiples documentos simultáneamente con
filtrado por `ACTIVE_DOCS`.

---

## v1.2: Capa de verificación

El sistema generaba respuestas pero no había ningún mecanismo para evaluar su fiabilidad.

**Citation check:** Verificación por regex del formato `[[N] ...]` en la respuesta generada.
Comprueba que los chunks citados existen en el contexto enviado al LLM.

**Self-consistency:** Segunda llamada al LLM para detectar alucinaciones. Caro en latencia pero
detectable: si el LLM genera afirmaciones contradictorias con el contexto, el score baja.

**Trust score:** Puntuación ponderada combinando citation (35%), retrieval (30%), consistency
(25%) y coverage (10%). Normalización min-max sobre los scores de retrieval antes de mostrar.

El objetivo no era bloquear respuestas sino dar al usuario una señal de confianza.

---

## v1.3: Metadatos y etiquetas

Con múltiples documentos y crecimiento del corpus, se necesitaba filtrar por colección o tipo
de documento sin modificar el ranking.

**Tabla `documents`:** Source of truth de documentos ingestados con campos de clasificación.

**Sistema de tags:** Etiquetas persistentes por documento, filtrables en retrieval. El mecanismo
de `FilterSpec` restringe el pool de candidatos antes de la búsqueda vectorial.

**Fuentes (`sources`):** Asociación de documentos a fuentes de datos para gestión del ciclo
de vida.

---

## v1.4: Transacciones de ingesta

Antes de v1.4, un fallo durante la ingesta (error de embedding, disco lleno, interrupción)
dejaba el sistema en estado inconsistente: ChromaDB podía tener chunks sin correspondencia
en DocStore o viceversa.

**IngestTransaction:** Clase que agrupa todas las escrituras de una ingesta y hace rollback
completo si cualquier paso falla. La ingesta se convierte en atómica: o todo se escribe, o
nada se escribe.

Esto también hizo posible el `rag-lab ingest rollback <run_id>` — deshacer una ingesta completa
de forma limpia.

---

## v1.5: Eliminación de datasets/CSV

Durante el desarrollo temprano habían entrado referencias a datasets tabulares (CSV, Parquet,
DuckDB) en el código. RAG-Lab es un sistema RAG sobre documentos Markdown — no tiene ninguna
semántica para datos tabulares.

**Scope guard:** Se eliminaron todos los loaders tabulares, referencias a datasets y tests
asociados. Los stores se aislaron en tests para evitar que una suite contamine a otra.

Esta decisión fue deliberada y no se revierte. Los datos tabulares no son un caso de uso de
RAG-Lab.

---

## v1.6: Validación Markdown y quality gate

Un problema emergente al crecer el corpus: documentos con estructura defectuosa (headings
vacíos, frontmatter mal formado, tablas rotas) generaban chunks de baja calidad que degradaban
el retrieval sin que hubiera ninguna señal de advertencia.

**`rag-lab docs validate`:** Comando de validación con códigos de error y advertencia antes de
ingestar. Detecta: frontmatter inválido, headings vacíos, tablas sin cabecera, secciones
excesivamente largas.

**`--strict`:** Flag que trata los WARNs como ERRORs. Útil para pipelines CI o ingesta masiva
donde se quiere garantía de calidad.

**`rag-lab docs inspect`:** Muestra la estructura del documento (frontmatter, tokens estimados,
chunks estimados) sin ingestar.

**`rag-lab docs preview-chunks`:** Genera los chunks que se crearían sin escribir nada en los
stores. Permite auditar el resultado del chunking antes de comprometerse.

---

## v1.7: Limpieza

Sprint de deuda técnica. Sin features nuevas, solo correcciones.

**Eliminar `generation/verifier.py`:** Existía un fichero duplicado de verificación que había
quedado huérfano tras la reorganización de v1.2. Se eliminó para evitar confusión.

**Fix reranker device cache:** El cache global del reranker no respetaba cambios de dispositivo
(CPU/GPU) entre tests. En tests siempre se debe usar CPU independientemente del valor de la
variable de entorno. El fix garantiza que `conftest.py` puede resetear el cache de dispositivo.

---

## v1.8: Framework de benchmark

Sin métricas reproducibles era imposible saber si un cambio mejoraba o empeoraba el retrieval.
El "benchmark" antes de v1.8 era ad-hoc: se probaba a mano con unas pocas preguntas.

**Eval queries:** Conjunto oficial de preguntas con relevance judgments sobre el corpus SDMX.
Cada query tiene chunks relevantes identificados por `doc_id` y posición.

**Baselines JSON:** El resultado de cada variante benchmarked se persiste en JSON para comparación
histórica.

**Compare guard:** Test que falla si una variante propuesta degrada las métricas principales
(R@5, MRR, nDCG@10) por encima del umbral configurado.

A partir de v1.8 cualquier cambio en retrieval se valida contra el benchmark antes de merge.

---

## v1.9: Tokenizer real

El chunking hasta v1.8 usaba una estimación de tokens por longitud de caracteres (`len(text)/4`).
Esta aproximación es imprecisa para texto técnico con muchos símbolos, tablas y términos
en inglés cortos.

**BAAI/bge-m3 tokenizer:** Se integró el tokenizer real del modelo de embedding para contar
tokens. El límite `CHUNK_MAX_TOKENS=800` ahora se aplica con precisión sobre los tokens reales
que verá el modelo.

Impacto: algunos chunks que antes pasaban el límite ahora se dividen; algunos que se dividían
innecesariamente ahora permanecen juntos. Mejora la coherencia semántica de los chunks.

---

## v1.10: Heading context en reranker

El reranker (BGE-reranker-v2-m3) recibe pares (query, chunk) para puntuar relevancia. Antes
de v1.10, el texto del chunk se enviaba tal cual, sin contexto del documento o sección al que
pertenecía.

**Prefix añadido al chunk:** `"Document: {título}\nSection: {heading_path}\n\n{texto}"` antes
de enviar al reranker.

**Resultado del benchmark:**
- R@5: +2.1pp
- MRR: +2.6pp

**Regresión conocida:** La query q070 (mezcla español/inglés) mostró -0.5pp en MRR. El efecto
es minoritario y el beneficio global es claro.

Este fue el cambio con mayor retorno sobre inversión en toda la historia del proyecto.

---

## v1.11: Query variants cleanup y CI baseline

En v1.10 se habían dejado activos mecanismos de expansión de queries (variantes sinónimas,
paráfrasis generadas por LLM). El A/B benchmark mostró:

- Variantes de query: **0 beneficio** en R@5, MRR, nDCG@10
- Latencia: **×2** (cada variante requiere un pase de embedding adicional)

**Decisión:** Desactivar variantes por defecto. El código se mantiene pero no se ejecuta salvo
flag explícito.

**v1.11 promovido como baseline de CI:** A partir de v1.11, el benchmark compara siempre contra
el estado de v1.11 como línea base. Esto hace reproducible la comparación histórica.

---

## v1.12: HyDE + query rewriting

HyDE (Hypothetical Document Embeddings) es una técnica que genera un fragmento de texto
hipotético con el LLM ("¿cómo sería la respuesta a esta pregunta?") y usa ese fragmento como
query de embedding en lugar del texto original.

**Implementación:** v1.12 implementó HyDE y query rewriting como opciones opt-in (`--hyde`,
`--rewrite`).

**Benchmark de HyDE en corpus SDMX:**
- R@5: -3.8pp respecto a baseline
- Latencia: ×12.5
- Veredicto: **neto negativo**

**Explicación:** HyDE ayuda cuando las queries son cortas o ambiguas y el corpus es heterogéneo.
El corpus SDMX es técnico y específico. Las queries ya son suficientemente informativas para el
embedding denso sin necesidad de expandirlas.

**Query rewriting:** Implementado pero sin benchmark oficial. Estado: disponible como `--rewrite`
pero no recomendado en producción.

---

## v1.13: Auditoría HNSW

ChromaDB usa HNSW para el índice vectorial. Los parámetros HNSW (M, ef_construction, ef_search)
controlan el trade-off entre calidad del índice y velocidad de búsqueda.

**Hallazgo:** En ChromaDB 1.x todos los parámetros HNSW son **build-time** — se fijan al crear
la colección y no pueden modificarse sin reconstruir el índice desde cero.

**Benchmark de perfiles:**
- M=8: recall -12pp respecto a M=16
- M=16 (default): baseline
- M=32: +0.3pp recall, +40% tiempo de construcción

**Decisión:** Mantener M=16 (default de ChromaDB). No hay beneficio en aumentar, y M=8 degrada
significativamente. El parámetro queda documentado pero no expuesto como configurable.

---

## v1.14: Query cache

Las queries repetidas (mismo texto, mismo corpus) regeneraban embedding, hacían búsqueda y
llamaban al LLM en cada ejecución. En uso interactivo esto es latencia innecesaria.

**Cache SQLite:** Caché persistente en SQLite keyed por fingerprint de query + configuración.
TTL de 7 días por defecto.

**Corpus fingerprint:** El fingerprint incluye un hash del estado del corpus (número de chunks,
IDs). Si se ingesta o elimina un documento, el fingerprint cambia y todas las entradas del
caché quedan invalidadas automáticamente.

**v1.14.1:** Fix de invalidación: operaciones de tags y `docs delete` también invalidan el
caché (se actualizaba el corpus pero el fingerprint no se recalculaba).

---

## v1.15: Feedback store

Sin datos de uso real no se puede saber qué queries fallan, qué chunks se recuperan
innecesariamente o qué documentos no son útiles para los usuarios.

**FeedbackEntry:** Registro por query de: query text, flag hyde, chunk_id recuperados, score
de retrieval, tipo de feedback (`useful`, `not_useful`, `relevant`, `irrelevant`, `wrong_doc`,
`outdated`, `duplicate`, `bad_citation`).

**Puramente observacional:** No afecta al ranking ahora ni en el futuro próximo. El riesgo de
overfitting a pocos eventos es real. El umbral para activarlo como señal de ranking es >50
eventos con patrón claro.

**CLI:** `rag-lab feedback add/list/stats/export/clear`.

---

## v1.16: Batch/resumable ingest

Con corpus más grandes o ingesta de muchos documentos a la vez, la ingesta secuencial podía
tardar minutos. Un fallo interrumpía todo el proceso y había que reiniciar.

**Workers paralelos:** `--workers N` permite ingestar N documentos en paralelo.

**IngestTransaction por batch:** Cada documento tiene su propia transacción. Un fallo en un
documento no afecta al resto del batch.

**Tracking de batches y runs:** `rag-lab ingest batches` y `rag-lab ingest runs` muestran
el historial de ingestas con estado (success, failed, partial). `rag-lab ingest retry <run_id>`
reintenta solo los documentos fallidos de un run anterior.

**`--resume`:** Continúa una ingesta interrumpida desde el último checkpoint.
**`--retry-failed`:** Reintenta automáticamente los documentos con estado `failed` del run más
reciente.

---

## v1.17: Release candidate audit

Sprint de calidad antes de v1.18. Sin features nuevas.

**Store isolation guard tests:** Tests que verifican que ninguna suite de tests deja estado en
los stores compartidos. Detectan casos donde un test crea chunks en ChromaDB o DocStore y no
los limpia correctamente.

**`rag-lab doctor`:** Comando de health check que ejecuta una serie de comprobaciones sobre el
estado del sistema: configuración, DocStore, ChromaDB, FTS5, sparse coverage, reconciliación,
ingest health y query de prueba.

**`rag-lab reconcile`:** Verifica y repara inconsistencias entre stores (DocStore, ChromaDB,
FTS5, Sparse BLOBs). `--check` solo reporta; `--repair` corrige.

**`rag-lab diagnose`:** Ejecuta una query de diagnóstico con traza detallada del pipeline:
qué chunks se recuperaron, qué reranker puntuó, qué filtros se aplicaron.

---

## v1.18: Verification hardening

La capa de verificación de v1.2 tenía cuatro bugs silenciosos que habían pasado inadvertidos
porque los tests no cubrían casos edge del parsing de citas.

**4 bugs corregidos:**
1. El regex de citation no reconocía el formato `[[N]]` (corchetes dobles sin espacio).
2. El score de coverage era siempre 1.0 si no había citas (debería ser 0.0).
3. El `evidence_map` no se construía si la respuesta estaba vacía.
4. La comparación de consistency fallaba silenciosamente si el segundo LLM call devolvía JSON
   malformado.

**evidence_map:** Mapa de evidencia explícito que asocia cada afirmación de la respuesta con
el chunk que la soporta. Visible en el verbose trace.

**Verbose trace:** `--profile` muestra ahora el trace completo de verificación: qué chunks se
citaron, qué puntuó la consistencia, por qué el trust score tiene ese valor.

---

## v1.18.1: E2E audit

Tras el hardening de v1.18, se realizó una auditoría E2E completa con LLM real (no mock).

**10/10 PASS:** Las 10 preguntas del conjunto de evaluación E2E produjeron respuestas con
citation check correcto, consistency check sin alucinaciones detectadas, y trust score > 0.7.

**Script de auditoría:** `scripts/e2e_audit.py` — ejecuta el conjunto completo y produce un
reporte de resultados. Reproducible con `python scripts/e2e_audit.py`.

---

## v1.18.2: Eliminación SparseStore JSON

El sparse index original (v1.0) usaba un fichero JSON para almacenar vectores sparse. En v1.16
se migró a SQLite BLOBs como formato canónico. En v1.18.2 se eliminó el código legacy de
lectura/escritura del JSON sparse.

**SparseStore JSON:** Dead code eliminado. El formato canónico es SQLite BLOBs en
`storage/docstore.sqlite`.

La migración `python -m rag_lab.maintenance.migrate_to_v2` queda disponible para instalaciones
que aún tengan el JSON antiguo.

---

## v1.19: Contrato de frontmatter

Con el corpus creciendo en número de documentos y dominios, la clasificación manual por nombre
de archivo era insuficiente. Se necesitaba un mecanismo estructurado para asociar metadatos de
clasificación a cada documento.

**YAML frontmatter contract:** Los campos `doc_id`, `title`, `domain`, `source_type`,
`language`, `version` y `tags` se leen del frontmatter YAML y se persisten en la tabla
`documents`.

**Tags derivados:** Cada campo de clasificación genera automáticamente un tag derivado
(`domain:sdmx`, `source_type:manual`, `lang:en`, `version:2.1`). Esto permite filtrar por
clasificación usando la misma infraestructura de tags existente sin código especial.

**FilterSpec:** `FilterSpec(domain="sdmx")` se resuelve internamente a
`tags_include=["domain:sdmx"]` antes del retrieval. El algoritmo de ranking no cambia.

**Campos prohibidos:** `dataset` y `dataset_id` producen ERROR `frontmatter_scope_violation`.
RAG-Lab no soporta datos tabulares.

---

## v1.19.1: Documentación

Sprint de documentación. Reorganización completa de `/docs/`:

- `ROADMAP.es.md` / `ROADMAP.en.md` — Hoja de ruta con filosofía explícita
- `DEVELOPMENT_HISTORY.es.md` / `DEVELOPMENT_HISTORY.en.md` — Este documento
- `API_REFERENCE.es.md` / `API_REFERENCE.en.md` — Referencia completa de CLI
- `FRONTMATTER.en.md` — Traducción del contrato de frontmatter existente

---

## Decisiones descartadas y por qué

### HyDE desactivado por defecto

Benchmarked en v1.12: R@5 -3.8pp, latencia ×12.5. El corpus SDMX es técnico y específico —
las queries ya son suficientemente informativas para embedding denso. HyDE queda disponible
como `--hyde` para experimentación pero no se activa por defecto.

### PDF pospuesto indefinidamente

La extracción de texto desde PDF pierde estructura documental. Los headings desaparecen o se
convierten en texto plano, las tablas se deforman, el orden de texto puede alterarse. El chunker
de RAG-Lab depende de la estructura Markdown. Sin un pipeline de conversión auditado que
produzca Markdown de calidad, los PDFs generarían chunks de baja calidad.

### Feedback frozen como señal de ranking

Con <50 eventos no hay señal estadística. Ajustar pesos de ranking a pocos datos produce
overfitting. El feedback existe como herramienta de observación y se activará como señal de
ranking cuando haya evidencia suficiente.

### CSV/datos tabulares eliminados (v1.5)

RAG-Lab es un sistema RAG sobre documentos Markdown. No tiene semántica para datos tabulares.
Los loaders CSV/Parquet/DuckDB que habían entrado en el código temprano se eliminaron como
scope guard.

### Sparse global descartado

Un scan sparse global sin WAND/early termination es O(N) sobre el corpus completo. SQLite no
tiene estos mecanismos de optimización para índices invertidos. A 610 chunks es tolerable, pero
no escala. La solución correcta requiere un motor dedicado (Elasticsearch, Qdrant). La
arquitectura actual (sparse solo sobre el pool de candidatos dense) es un compromiso correcto
para el tamaño de corpus actual.

### Query variants desactivadas (v1.11)

A/B benchmark: 0 beneficio en R@5/MRR/nDCG@10, latencia ×2. El coste no justifica el beneficio
nulo. Las variantes están implementadas pero desactivadas por defecto.

---

*Última actualización: v1.19.1*
