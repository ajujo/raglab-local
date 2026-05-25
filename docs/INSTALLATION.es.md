# Instalación de RAG-Lab

RAG-Lab es una herramienta CLI local para consultar documentos Markdown mediante un pipeline RAG (Retrieval-Augmented Generation). Este documento cubre la instalación completa desde cero.

---

## 1. Requisitos previos

### Python 3.11

RAG-Lab requiere **Python 3.11**. Se recomienda usar [conda](https://docs.conda.io/) para aislar el entorno y evitar conflictos de dependencias.

```bash
# Verificar la versión disponible
python --version
conda --version
```

Si no tienes conda, instala [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/).

### GPU NVIDIA (recomendada)

El entorno de desarrollo usa una RTX 5090. Una GPU NVIDIA con soporte CUDA acelera significativamente el embedding y el reranking. Sin embargo, **el modo CPU está disponible** y es el modo que usan los tests automáticamente.

Si no dispones de GPU, consulta la sección [Modo solo CPU](#6-modo-solo-cpu) más adelante.

### Servidor LLM propio

RAG-Lab **no incluye un LLM**. La generación de respuestas se delega a un servidor externo con una API compatible con OpenAI. Algunas opciones habituales:

- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (con servidor HTTP)
- [Ollama](https://ollama.com/)

El servidor debe estar corriendo y accesible antes de lanzar consultas. El embedding y el reranking se realizan localmente con modelos de HuggingFace; el LLM es el único componente externo.

### Git

```bash
git --version
```

---

## 2. Instalación del entorno

### Clonar el repositorio

```bash
git clone https://github.com/ajujo/raglab-local.git
cd raglab-local
```

### Crear y activar el entorno conda

```bash
conda create -n rag-lab python=3.11
conda activate rag-lab
```

### Instalar RAG-Lab en modo editable

```bash
pip install -e .
```

Este comando instala el paquete y registra el comando `rag-lab` en el entorno. A partir de aquí, todos los comandos se invocan con el prefijo `rag-lab`.

---

## 3. Configuración (.env)

RAG-Lab lee su configuración de un fichero `.env` en la raíz del proyecto.

```bash
cp .env.example .env
```

Abre `.env` con tu editor y ajusta las variables:

```dotenv
# URL base del servidor LLM (API compatible con OpenAI)
LLM_BASE_URL=http://localhost:8000/v1

# Nombre del modelo tal como lo expone el servidor
LLM_MODEL=your-model-name

# Dispositivo para el modelo de embedding (cuda o cpu)
EMBEDDING_DEVICE=cuda

# Dispositivo para el modelo de reranking (cuda o cpu)
RERANKER_DEVICE=cuda
```

### Descripción de cada variable

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `LLM_BASE_URL` | Endpoint del servidor LLM. Debe incluir `/v1` al final si el servidor lo requiere. | `http://localhost:8000/v1` |
| `LLM_MODEL` | Identificador del modelo servido. Debe coincidir con lo que expone el servidor. | — |
| `EMBEDDING_DEVICE` | Dispositivo para BAAI/bge-m3. Usa `cuda` con GPU o `cpu` sin ella. | `cuda` |
| `RERANKER_DEVICE` | Dispositivo para BAAI/bge-reranker-v2-m3. Independiente de `EMBEDDING_DEVICE`. | `cuda` |

> **Nota:** Los modelos de embedding y reranker se descargan automáticamente desde HuggingFace la primera vez que se usan. No es necesario descargarlos manualmente.

---

## 4. Modelos necesarios

### Embedding: BAAI/bge-m3

RAG-Lab usa [BGE-M3](https://huggingface.co/BAAI/bge-m3) para generar vectores densos y dispersos simultáneamente. Este modelo se descarga automáticamente al ejecutar el primer `rag-lab ingest`.

- Tamaño aproximado: ~570 MB
- Cargado perezosamente y cacheado en memoria durante la sesión

### Reranker: BAAI/bge-reranker-v2-m3

El [BGE-Reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) reordena los candidatos recuperados antes de enviárselos al LLM. También se descarga automáticamente al primer uso.

- Tamaño aproximado: ~570 MB
- Se puede omitir con `--fast` en consultas (sin reranking)

### LLM (servidor externo)

El modelo de lenguaje se sirve **fuera de RAG-Lab**. No hay descarga automática del LLM; debes gestionar el servidor por tu cuenta. RAG-Lab solo se comunica con él vía HTTP usando la API compatible con OpenAI.

Ejemplo con vLLM:

```bash
# En otra terminal o proceso
python -m vllm.entrypoints.openai.api_server \
  --model sakamakismile/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000
```

---

## 5. Verificar la instalación

Una vez configurado el `.env` y con el servidor LLM activo, ejecuta:

```bash
rag-lab doctor
```

El comando `doctor` comprueba todos los subsistemas: configuración, stores (ChromaDB, SQLite, índice disperso), conectividad con el LLM, estado de FTS5, y más. La salida indica claramente qué pasa y qué falla.

```bash
# Ver todos los comandos disponibles
rag-lab --help

# Ver la ayuda de un subcomando
rag-lab query --help
rag-lab ingest --help
```

Si `doctor` devuelve errores de stores vacíos (ChromaDB, docstore), es normal en una instalación nueva. Necesitas ingestar al menos un documento antes de consultar:

```bash
rag-lab ingest --doc path/to/your/document.md
```

---

## 6. Modo solo CPU

Si no dispones de GPU NVIDIA o prefieres ejecutar en CPU:

```dotenv
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

El rendimiento es considerablemente más lento en CPU, especialmente el embedding y el reranking. Para uso ocasional o en máquinas sin GPU es perfectamente funcional.

> **Los tests siempre corren en CPU.** El fichero `conftest.py` fija `CUDA_VISIBLE_DEVICES=""` automáticamente, por lo que no hace falta GPU para ejecutar la suite de tests.

También puedes sobreescribir el dispositivo por sesión desde la línea de comandos:

```bash
rag-lab query "mi pregunta" --cpu-embedding --cpu-reranker
```

---

## 7. Errores frecuentes

### CUDA not available

```
RuntimeError: CUDA not available
```

**Causa:** `EMBEDDING_DEVICE=cuda` o `RERANKER_DEVICE=cuda` pero no hay GPU disponible (o el driver no está instalado).

**Solución:** Cambia a `cpu` en `.env`:

```dotenv
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

---

### Connection refused al LLM

```
LLMConnectionError: Connection refused to http://localhost:8000/v1
```

**Causa:** El servidor LLM no está arrancado o está escuchando en un puerto distinto.

**Solución:**

1. Verifica que el servidor LLM está corriendo.
2. Comprueba que `LLM_BASE_URL` apunta al host y puerto correctos.
3. Si el servidor requiere `/v1` en la URL base, asegúrate de incluirlo.

---

### ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rag_lab'
```

**Causa:** El paquete no está instalado en el entorno activo, o se está usando un entorno distinto.

**Solución:**

```bash
# Asegurarse de estar en el entorno correcto
conda activate rag-lab

# Reinstalar desde el directorio raíz del repositorio
cd /ruta/a/raglab-local
pip install -e .
```

---

### Stores vacíos tras la instalación

```
Warning: ChromaDB collection is empty
Warning: DocStore has 0 chunks
```

**Causa:** Es el comportamiento esperado en una instalación nueva. Los stores se crean al ingestar.

**Solución:** Ingestar al menos un documento:

```bash
rag-lab docs validate path/to/doc.md
rag-lab ingest --doc path/to/doc.md
```

Consulta la guía [USAGE.es.md](USAGE.es.md) para el flujo completo de ingesta.

---

## Estructura de directorios tras la instalación

```
raglab-local/
├── .env                    # Tu configuración (no en git)
├── .env.example            # Plantilla de configuración
├── rag_lab/                # Código fuente del pipeline
├── data/
│   └── ingested.jsonl      # Manifiesto de documentos ingestados
├── storage/
│   ├── chroma_db/          # Base de datos vectorial (ChromaDB)
│   ├── sparse_index.json   # Índice disperso (BM25-like)
│   └── docstore.sqlite     # SQLite: texto y metadatos de chunks
└── tests/                  # Suite de tests
```

Los directorios `storage/` y `data/` se crean automáticamente al primer ingest.
