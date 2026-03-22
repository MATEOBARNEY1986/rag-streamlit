# RAG con Streamlit + ChromaDB + Groq

Aplicación de Retrieval-Augmented Generation (RAG) con interfaz web en Streamlit.
Indexa PDFs y archivos de texto, responde preguntas usando los documentos como contexto.

## Stack

| Componente | Tecnología |
|------------|-----------|
| UI | Streamlit |
| Embeddings | `sentence-transformers` · `paraphrase-multilingual-MiniLM-L12-v2` (local, 384-dim) |
| Generación | Groq API · `llama-3.3-70b-versatile` |
| Vector store | ChromaDB (HTTP API v2) |
| PDF parsing | PyMuPDF |

## Características

- **Embeddings locales** — el modelo corre en el container, sin llamadas externas para vectorizar
- **Indexación paralela** — extracción de texto multi-hilo (6 workers) + batch encoding
- **Multilingüe** — el modelo soporta español, inglés y 50+ idiomas
- **Multi-colección** — configurable por variable de entorno
- **Incremental** — omite archivos ya indexados (cache por hash MD5)

## Requisitos

- Docker + Docker Compose
- [ChromaDB](https://docs.trychroma.com/) corriendo en `localhost:8000` (o configurable)
- API key de [Groq](https://console.groq.com/)

## Inicio rápido

```bash
# 1. Clonar
git clone https://github.com/tu-usuario/rag-streamlit
cd rag-streamlit

# 2. Configurar claves
cp .env.example .env
# editar .env y poner tu GROQ_API_KEY

# 3. Crear carpeta de documentos
mkdir docs
cp mis_documentos/*.pdf docs/

# 4. Levantar ChromaDB (si no tienes uno)
docker run -d -p 8000:8000 chromadb/chroma

# 5. Construir y arrancar
docker compose up -d --build

# 6. Abrir
open http://localhost:8501
```

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Requerida.** API key de Groq |
| `CHROMA_URL` | `http://host.docker.internal:8000` | URL de ChromaDB |
| `CHROMA_COLLECTION` | `rag_docs` | Nombre de la colección |
| `DOCS_FOLDER` | `/docs` | Carpeta de documentos dentro del container |

## Red Docker

- **Linux**: descomenta `network_mode: host` en `docker-compose.yml` para que el container acceda a ChromaDB en `localhost:8000`
- **Mac / Windows**: usa `host.docker.internal:8000` (default)

## Formato de documentos soportados

`.pdf` `.txt` `.md`

## Uso

1. Sube documentos desde la barra lateral o copia archivos en `docs/`
2. Pulsa **⚡ Indexar / Actualizar**
3. Escribe tu pregunta en el chat

## Cambiar el modelo LLM

En `app.py` edita:

```python
GEN_MODEL = "llama-3.3-70b-versatile"  # cualquier modelo de Groq
```

Modelos disponibles en Groq: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, etc.

## Cambiar el modelo de embeddings

En `app.py` y `Dockerfile`:

```python
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 384-dim
```

Si cambias el modelo, borra la colección en ChromaDB antes de re-indexar (las dimensiones deben coincidir).
