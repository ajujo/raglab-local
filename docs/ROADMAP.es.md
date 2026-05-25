# RAG-Lab — Hoja de ruta (v1.19.1)

> **Filosofía:** No añadir features por inercia. Solo con evidencia, benchmark o dolor real de uso.
> Si no hay un problema concreto que resolver o una métrica que mejorar, la funcionalidad no entra.

---

## Estado actual

**Versión:** v1.19.1
**Estado:** Sistema estable, en uso local activo
**Corpus:** 610 chunks de documentación SDMX
**Tests:** 1031 tests, todos en verde
**Retrieval:** R@5=0.821, R@10=0.896, MRR=0.939, nDCG@10=0.837

El sistema está completo para su uso actual como herramienta CLI local sobre un corpus Markdown
de documentación técnica SDMX. No hay deuda técnica conocida activa.

---

## Próximo razonable (basado en uso real)

Estas tareas tienen justificación directa en el uso actual del sistema.

### Añadir frontmatter al corpus SDMX existente

La mayoría de documentos del corpus son anteriores a v1.19 y no tienen frontmatter. Añadir el
contrato YAML (`doc_id`, `domain`, `source_type`, `language`, `version`) manualmente documento
a documento es trabajo de documentación, no de ingeniería.

- **Por qué ahora:** El sistema ya puede aprovechar los metadatos para filtrado en retrieval.
  Sin frontmatter, los filtros `FilterSpec(domain="sdmx")` no funcionan sobre documentos legacy.
- **Coste:** Trabajo manual. Se puede hacer incrementalmente.
- **Riesgo:** Bajo. El proceso es `validate → inspect → ingest --force`.

### Benchmark de calidad de respuesta E2E

El benchmark actual mide calidad de retrieval (R@5, MRR, nDCG@10). No mide si la respuesta
generada es correcta, completa o útil. Para evaluar el pipeline completo se necesita un
conjunto de pares (pregunta, respuesta esperada) y una métrica de evaluación de texto.

- **Por qué ahora:** Sin esta métrica, mejoras en retrieval no se traducen necesariamente en
  mejores respuestas percibidas por el usuario.
- **Coste:** Alto. Requiere crear el dataset de evaluación y definir la métrica (ROUGE, BERTScore,
  evaluación LLM-as-judge, o evaluación humana).
- **Bloqueante:** No hay conjunto de "respuestas correctas" para el corpus SDMX. La tarea de
  creación de ese dataset es previa.

### Vigilancia de escala: sparse O(N) en pool de candidatos

El scan sparse actual es O(N) sobre el pool de candidatos (top-K del vector search). Con 610
chunks es trivialmente rápido. Si el corpus crece varios órdenes de magnitud, esta decisión de
diseño tendrá que revisarse.

- **Señal para actuar:** Latencia de retrieval > 2s en benchmark, o corpus > 50.000 chunks.
- **Solución futura:** Motor con WAND/early termination (Elasticsearch, Qdrant). No SQLite.
- **Hoy:** No hacer nada. El problema no existe.

---

## Mejoras pospuestas (esperando datos reales)

Funcionalidades implementadas o diseñadas pero deliberadamente no activadas hasta tener evidencia.

### Feedback como señal de re-ranking

El store de feedback existe desde v1.15 y registra eventos (útil, no útil, incorrecto, etc.).
Actualmente es **puramente observacional** — no afecta al ranking ni a la selección de chunks.

- **Por qué pospuesto:** Con menos de 50 eventos no hay señal estadística. Ajustar pesos de
  ranking a pocos datos produce overfitting a casos ruidosos, degradando el rendimiento global.
- **Condición para activar:** >50 eventos con patrón claro y reproducible de fallos evitables.
  Se necesita un A/B benchmark que demuestre mejora antes de activar.
- **Riesgo si se activa prematuramente:** Un par de eventos negativos sobre un chunk legítimamente
  relevante pueden suprimir ese chunk en futuras queries.

### Query rewriting activado por defecto

El rewriter de queries está implementado desde v1.12 como opción `--rewrite`. No se ha hecho
benchmark oficial de su efecto en retrieval.

- **Por qué pospuesto:** Sin benchmark no sabemos si ayuda o perjudica. HyDE se implementó con
  las mismas expectativas y resultó negativo (-3.8pp R@5, latencia ×12.5).
- **Condición para activar:** Benchmark oficial que muestre mejora neta en R@5 y MRR con
  latencia aceptable.
- **Estado actual:** Disponible como `rag-lab query "..." --rewrite`. No recomendado en
  producción hasta benchmark.

---

## Mejoras congeladas (no hacer salvo evidencia fuerte)

Decisiones tomadas con datos. Reabrir solo si los datos cambian.

### HyDE activado por defecto

HyDE (Hypothetical Document Embeddings) genera un fragmento hipotético con el LLM antes de
buscar en el corpus. Se implementó en v1.12 y se benchmarked en el corpus SDMX oficial.

**Resultado del benchmark:**
- R@5: -3.8pp respecto a la baseline sin HyDE
- Latencia: ×12.5 (el LLM añade latencia de generación antes del retrieval)
- Veredicto: neto negativo en este corpus

**Estado:** Disponible como `rag-lab query "..." --hyde` para experimentación. No activar por
defecto a menos que un corpus nuevo muestre un beneficio claro.

**Por qué puede funcionar en otros corpora:** HyDE ayuda cuando las queries son muy cortas o
ambiguas y el corpus es muy heterogéneo. El corpus SDMX es técnico y específico: las queries
ya son suficientemente informativas para el retrieval denso.

### Búsqueda sparse global (sin pool previo de dense)

La arquitectura actual hace sparse solo sobre el pool de candidatos del vector search (dense
primero, luego sparse sobre ese subconjunto). No hay WAND ni early termination en SQLite.

**Por qué congelado:** Un scan sparse global sobre el corpus completo con SQLite es O(N) sobre
todos los documentos. A 610 chunks es tolerable, pero la arquitectura no escala. Hacer esto
bien requiere un motor dedicado (Elasticsearch, Qdrant con soporte sparse nativo). SQLite no
es el lugar correcto.

---

## Posibles evoluciones de producto (largo plazo)

Estas no son tareas previstas — son caminos posibles si el uso del sistema evoluciona.

### API REST / multiusuario / autenticación

Hoy RAG-Lab es un CLI local. Si se quiere convertir en un servicio accesible desde otros
sistemas o por múltiples usuarios simultáneos, haría falta:
- Servidor FastAPI/Flask sobre la lógica de retrieval y generación
- Autenticación (API keys, OAuth)
- Gestión de sesiones y contexto por usuario
- Persistencia de conversaciones

**Señal para actuar:** Necesidad real de acceso multi-usuario o integración con otros sistemas.

### Soporte de LLM providers en la nube

El sistema hoy asume un endpoint OpenAI-compatible local (`http://localhost:8000/v1`). Para
portabilidad a OpenAI, Anthropic, Mistral AI, etc., haría falta:
- Abstracción del cliente LLM (hoy es un wrapper fino sobre la API OpenAI)
- Gestión de costes y rate limiting
- Configuración de provider por entorno

**Señal para actuar:** Necesidad de usar el sistema fuera del entorno local o en CI con
provider externo.

### Soporte PDF/DOCX/HTML

Hoy RAG-Lab solo ingesta Markdown. La razón es que la extracción de texto desde PDF/DOCX pierde
estructura: los headings se pierden, las tablas se deforman, el orden del texto puede alterarse.
La calidad del chunking depende de la calidad del Markdown de entrada.

Si se quiere añadir estos loaders, el prerrequisito es un pipeline de conversión auditado que
preserve la estructura documental (no solo extraiga texto plano).

---

## No hacer todavía

Lista explícita de cosas que se han considerado y se han descartado para el futuro próximo.

### PDF/DOCX/HTML loaders

La extracción pierde estructura. No hasta tener un pipeline de conversión auditado que preserve
headings, tablas y orden de contenido. La calidad del chunking depende directamente de la calidad
del Markdown de entrada. Un pipeline de conversión que produzca Markdown de baja calidad
degradaría el retrieval.

### AutoML / datos tabulares

Fuera del scope. RAG-Lab es para documentos, no para datasets. Los loaders CSV/Parquet/DuckDB
se eliminaron en v1.5 como "scope guard". Esta decisión no se revierte.

### Fine-tuning del modelo de embedding

Alto coste computacional, riesgo de regresión sobre el corpus existente, y necesita un corpus de
entrenamiento anotado (pares query-documento relevantes). Sin ese corpus, el fine-tuning es una
apuesta a ciegas. La mejora que dio mayor retorno sobre inversión (heading context en v1.10, +2.1pp
R@5) fue de ingeniería de features, no de fine-tuning.

### Indexación incremental sin re-ingesta completa

Hoy añadir o modificar un documento requiere ingestar ese documento (`rag-lab ingest --doc`).
Hacer actualizaciones granulares a nivel de chunk sin re-ingestar es complejo y frágil. El coste
de re-ingestar un documento es bajo. No hay necesidad demostrada de indexación incremental.

---

## Señales que justificarían abrir v1.20

No hay sprint planificado. v1.20 se abrirá cuando se cumpla al menos una de estas condiciones:

1. **Feedback con patrón claro:** >50 eventos de feedback con un tipo de fallo recurrente y
   evitable que el sistema pudiera corregir con un cambio de ranking o retrieval.

2. **Corpus nuevo con problema de calidad:** Se incorpora un corpus nuevo (documentación de otro
   estándar, manuales técnicos) y revela problemas de chunking, frontmatter o retrieval que el
   sistema actual no maneja bien.

3. **Pregunta que el sistema falla sistemáticamente:** Una clase de preguntas que deberían tener
   respuesta en el corpus y el sistema no puede responder correctamente de forma repetida.
   Identificada por el benchmark E2E o por uso directo.

4. **Benchmark E2E disponible:** Una vez que exista el dataset de pares (pregunta, respuesta
   esperada), el primer resultado del benchmark E2E probablemente revelará mejoras concretas.

5. **Necesidad de escala:** El corpus crece hasta un tamaño donde la latencia actual sea
   inaceptable (>2s en retrieval) y haya un beneficio claro en cambiar la arquitectura de
   almacenamiento.

---

*Última actualización: v1.19.1*
