import os
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as _req
import streamlit as st
from dotenv import load_dotenv
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Paths & constants ─────────────────────────────────────────────────────────
DOCS_FOLDER  = Path(os.getenv("DOCS_FOLDER", str(Path(__file__).parent.parent / "RAG_buconos")))
CACHE_FOLDER = DOCS_FOLDER / ".cache"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384
GEN_MODEL        = "llama-3.3-70b-versatile"
SUPPORTED        = {".pdf", ".txt", ".md"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CHROMA_URL   = os.getenv("CHROMA_URL", "http://localhost:8000")
CHROMA_COLL  = os.getenv("CHROMA_COLLECTION", "buconos_rag")
_CHROMA_COLLS = (
    f"{CHROMA_URL}/api/v2/tenants/default_tenant"
    f"/databases/default_database/collections"
)

PDF_WORKERS = 6   # páginas extraídas en paralelo por PDF
FILE_WORKERS = 4  # archivos procesados en paralelo


# ── ChromaDB thin wrapper ─────────────────────────────────────────────────────
class ChromaIndex:
    def __init__(self):
        colls = _req.get(_CHROMA_COLLS, timeout=10).json()
        match = next((c for c in colls if c["name"] == CHROMA_COLL), None)
        if match:
            self._cid = match["id"]
        else:
            r = _req.post(
                _CHROMA_COLLS,
                json={"name": CHROMA_COLL, "configuration": {}},
                timeout=10,
            )
            r.raise_for_status()
            self._cid = r.json()["id"]
        self._base = f"{_CHROMA_COLLS}/{self._cid}"

    def add(self, ids, embeddings, documents, metadatas):
        _req.post(
            f"{self._base}/add",
            json={"ids": ids, "embeddings": embeddings,
                  "documents": documents, "metadatas": metadatas},
            timeout=60,
        ).raise_for_status()

    def query(self, embedding, top_k):
        resp = _req.post(
            f"{self._base}/query",
            json={"query_embeddings": [embedding], "n_results": top_k,
                  "include": ["documents", "metadatas", "distances"]},
            timeout=15,
        )
        resp.raise_for_status()
        r = resp.json()
        ids       = (r.get("ids")       or [[]])[0]
        docs      = (r.get("documents") or [[]])[0]
        metas     = (r.get("metadatas") or [[]])[0]
        distances = (r.get("distances") or [[]])[0]
        return [
            {"id": doc_id, "metadata": metas[i] if i < len(metas) else {},
             "text": docs[i] if i < len(docs) else "",
             "score": round(1 - distances[i], 4) if i < len(distances) else 0}
            for i, doc_id in enumerate(ids)
        ]

    def count(self):
        try:
            return _req.get(f"{self._base}/count", timeout=10).json()
        except Exception:
            return 0


# ── Clients (cached across reruns) ───────────────────────────────────────────
@st.cache_resource
def get_clients():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = ChromaIndex()
    return model, index


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Codifica una lista de textos en un solo batch (rápido)."""
    return model.encode(texts, batch_size=64, show_progress_bar=False).tolist()


# ── PDF: extracción paralela de texto ────────────────────────────────────────
def _extract_page(args) -> tuple[int, str]:
    pdf_path, page_idx = args
    doc  = fitz.open(str(pdf_path))
    text = doc[page_idx].get_text().strip()
    doc.close()
    return page_idx, text


def pdf_texts_parallel(pdf_path: Path) -> list[str]:
    """Extrae texto de todas las páginas en paralelo y devuelve lista ordenada."""
    doc        = fitz.open(str(pdf_path))
    n_pages    = len(doc)
    doc.close()

    results = {}
    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as ex:
        futures = {ex.submit(_extract_page, (pdf_path, i)): i for i in range(n_pages)}
        for fut in as_completed(futures):
            idx, text = fut.result()
            results[idx] = text

    return [results[i] or pdf_path.stem for i in range(n_pages)]


# ── File hash ─────────────────────────────────────────────────────────────────
def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:10]


# ── Ingest de un solo archivo ─────────────────────────────────────────────────
def _ingest_file(fpath: Path, fh: str, model, index: ChromaIndex):
    ext = fpath.suffix.lower()
    ids, embs, docs, metas = [], [], [], []

    if ext == ".pdf":
        texts = pdf_texts_parallel(fpath)           # extracción paralela
        embeddings = embed_texts(model, texts)       # batch encoding
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            ids.append(f"{fh}_p{i+1}")
            embs.append(emb)
            docs.append(text[:2000])
            metas.append({"source": fpath.name, "page": i+1,
                          "text": text[:2000], "type": "pdf_page"})

    elif ext in {".txt", ".md"}:
        raw    = fpath.read_text(encoding="utf-8", errors="ignore")
        chunks = [raw[i:i+1500] for i in range(0, len(raw), 1200)]
        embeddings = embed_texts(model, chunks)
        for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            ids.append(f"{fh}_c{ci}")
            embs.append(emb)
            docs.append(chunk)
            metas.append({"source": fpath.name, "page": ci+1,
                          "text": chunk, "type": "text"})

    # Subir a ChromaDB en lotes de 100
    for i in range(0, len(ids), 100):
        index.add(ids[i:i+100], embs[i:i+100], docs[i:i+100], metas[i:i+100])

    return len(ids)


# ── Ingestion principal ───────────────────────────────────────────────────────
def ingest(model, index: ChromaIndex):
    files = sorted(f for f in DOCS_FOLDER.iterdir() if f.suffix.lower() in SUPPORTED)
    if not files:
        st.warning("Suelta documentos en `RAG_buconos/` y vuelve a pulsar.")
        return

    state_path = CACHE_FOLDER / "indexed.json"
    indexed    = json.loads(state_path.read_text()) if state_path.exists() else {}

    pending = [(f, file_hash(f)) for f in files if indexed.get(f.name) != file_hash(f)]
    skip    = len(files) - len(pending)

    status = st.empty()
    bar    = st.progress(0.0)

    if skip:
        status.info(f"⏭  {skip} archivo(s) ya indexados, procesando {len(pending)} nuevos…")

    errors = []

    def process(item):
        fpath, fh = item
        n = _ingest_file(fpath, fh, model, index)
        indexed[fpath.name] = fh
        state_path.write_text(json.dumps(indexed))
        return fpath.name, n

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as ex:
        futures = {ex.submit(process, item): item[0].name for item in pending}
        done = 0
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                name, n_vecs = fut.result()
                done += 1
                bar.progress(done / max(len(pending), 1))
                status.info(f"✅ {name} → {n_vecs} vectores  ({done}/{len(pending)})")
            except Exception as e:
                errors.append(f"{fname}: {e}")
                st.error(f"❌ {fname}: {e}")

    bar.empty()
    if errors:
        status.warning(f"Completado con {len(errors)} error(es).")
    else:
        status.success("✅ Indexación completa")


# ── Retrieval + generation ────────────────────────────────────────────────────
def answer(model, index: ChromaIndex, question: str, top_k: int) -> dict:
    q_emb   = embed_texts(model, [question])[0]
    matches = index.query(q_emb, top_k)

    if not matches:
        return {"answer": "No encontré información relevante en los documentos indexados.",
                "sources": []}

    ctx_blocks = []
    sources    = []
    for m in matches:
        meta = m["metadata"]
        ctx_blocks.append(f"[{meta['source']} · pág. {meta['page']}]\n{meta['text']}")
        sources.append({**meta, "score": m["score"]})

    context = "\n\n---\n\n".join(ctx_blocks)
    prompt  = (
        "Eres un asistente experto. Responde la pregunta basándote ÚNICAMENTE en el contexto dado.\n"
        "Si la respuesta no está en el contexto, dilo claramente.\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\nRESPUESTA:"
    )

    resp = _req.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GEN_MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2},
        timeout=30,
    )
    resp.raise_for_status()
    return {"answer": resp.json()["choices"][0]["message"]["content"], "sources": sources}


# ── SiAM scraper ─────────────────────────────────────────────────────────────
_INAT_PROJECT = "tiburones-rayas-y-quimeras-de-colombia"
_INAT_HEADERS = {"User-Agent": "RAG-Buconos/1.0", "Accept": "application/json"}

def _inat_get(url: str) -> dict:
    resp = _req.get(url, headers=_INAT_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()

def _fetch_all_observations() -> list[dict]:
    obs, page = [], 1
    while True:
        data = _inat_get(
            f"https://api.inaturalist.org/v1/observations"
            f"?project_id={_INAT_PROJECT}&per_page=200&page={page}"
        )
        batch = data.get("results", [])
        obs.extend(batch)
        if len(obs) >= data["total_results"] or not batch:
            break
        page += 1
        time.sleep(0.4)
    return obs

def _fmt_obs(o: dict) -> str:
    t    = o.get("taxon") or {}
    name = t.get("preferred_common_name") or t.get("name") or o.get("species_guess") or "?"
    sci  = t.get("name") or ""
    lines = [
        f"Avistamiento #{o['id']}",
        f"Especie: {name}" + (f" ({sci})" if sci else ""),
        f"Lugar: {o.get('place_guess') or 'No especificado'}",
        f"Fecha: {o.get('observed_on') or o.get('created_at','')[:10]}",
        f"Calidad: {o.get('quality_grade','casual')}",
    ]
    if o.get("location"):
        lat, lon = o["location"].split(",")
        lines.append(f"Coordenadas: {float(lat):.4f}, {float(lon):.4f}")
    if (o.get("description") or "").strip():
        lines.append(f"Descripción: {o['description'].strip()[:400]}")
    return "\n".join(lines)

def _species_sheet(taxon: dict, obs_list: list[dict]) -> str:
    name = taxon.get("preferred_common_name") or taxon.get("name", "")
    sci  = taxon.get("name", "")
    cs   = (taxon.get("conservation_status") or {}).get("status_name") or "No evaluado"
    anc  = " > ".join(
        a["name"] for a in (taxon.get("ancestors") or [])
        if a.get("rank") in ("class", "order", "family")
    )
    places = sorted({o.get("place_guess","") for o in obs_list if o.get("place_guess")})
    dates  = sorted(o.get("observed_on","") for o in obs_list if o.get("observed_on"))
    lines  = [
        f"=== {name} ({sci}) ===",
        f"Clasificación: {anc}" if anc else "",
        f"Estado de conservación: {cs}",
        f"Observaciones en Colombia: {len(obs_list)}",
        f"Lugares: {', '.join(places[:10])}" if places else "",
        f"Período: {dates[0]} a {dates[-1]}" if dates else "",
        f"\n{(taxon.get('wikipedia_summary') or '')[:800]}" if taxon.get("wikipedia_summary") else "",
    ]
    return "\n".join(l for l in lines if l)

def fetch_siam_data(status_placeholder) -> tuple[Path, Path]:
    """Descarga datos de iNaturalist y guarda dos archivos en DOCS_FOLDER."""
    status_placeholder.info("Descargando observaciones de iNaturalist…")
    obs = _fetch_all_observations()

    by_taxon: dict[int, list] = defaultdict(list)
    taxon_obj: dict[int, dict] = {}
    for o in obs:
        t = o.get("taxon") or {}
        tid = t.get("id")
        if tid:
            by_taxon[tid].append(o)
            taxon_obj.setdefault(tid, t)

    # Avistamientos
    lines_obs = [
        "AVISTAMIENTOS — TIBURONES, RAYAS Y QUIMERAS EN COLOMBIA",
        f"Fuente: iNaturalist · {_INAT_PROJECT}",
        f"Total: {len(obs)} observaciones", "=" * 60, "",
        *[_fmt_obs(o) + "\n" for o in obs],
    ]
    obs_path = DOCS_FOLDER / "siam_avistamientos.txt"
    obs_path.write_text("\n".join(lines_obs), encoding="utf-8")

    # Fichas de especie (enriquece las 40 más observadas)
    status_placeholder.info(f"Enriqueciendo fichas de {len(by_taxon)} especies…")
    lines_sp = [
        "FICHAS DE ESPECIES — TIBURONES, RAYAS Y QUIMERAS DE COLOMBIA",
        f"Fuente: iNaturalist API", "=" * 60, "",
    ]
    for i, (tid, olist) in enumerate(sorted(by_taxon.items(), key=lambda x: -len(x[1]))):
        t = taxon_obj[tid]
        if i < 40:
            try:
                full = _inat_get(f"https://api.inaturalist.org/v1/taxa/{tid}")
                if full.get("results"):
                    t = {**t, **full["results"][0]}
                time.sleep(0.3)
            except Exception:
                pass
        lines_sp.append(_species_sheet(t, olist) + "\n")
    sp_path = DOCS_FOLDER / "siam_especies.txt"
    sp_path.write_text("\n".join(lines_sp), encoding="utf-8")

    # Invalidar cache para forzar re-indexación
    state_path = CACHE_FOLDER / "indexed.json"
    if state_path.exists():
        indexed = json.loads(state_path.read_text())
        indexed.pop("siam_avistamientos.txt", None)
        indexed.pop("siam_especies.txt", None)
        state_path.write_text(json.dumps(indexed))

    return obs_path, sp_path


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="RAG Buconos", page_icon="🧠", layout="wide")
    model, index = get_clients()

    with st.sidebar:
        st.title("🧠 RAG Buconos")
        st.caption(f"Carpeta: `{DOCS_FOLDER}`")

        docs = [f for f in DOCS_FOLDER.iterdir() if f.suffix.lower() in SUPPORTED]
        st.metric("Documentos en carpeta", len(docs))

        if docs:
            with st.expander("Archivos detectados"):
                for d in docs:
                    st.text(f"• {d.name}")

        st.divider()

        uploaded = st.file_uploader(
            "Sube documentos",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            for uf in uploaded:
                (DOCS_FOLDER / uf.name).write_bytes(uf.getvalue())
            st.success(f"✅ Guardado: {', '.join(u.name for u in uploaded)}")
            st.rerun()

        st.divider()

        if st.button("⚡ Indexar / Actualizar", use_container_width=True):
            ingest(model, index)
            st.rerun()

        st.divider()
        st.caption("🌊 Datos externos")
        if st.button("🔄 Actualizar SiAM / iNaturalist", use_container_width=True):
            _status = st.empty()
            try:
                obs_p, sp_p = fetch_siam_data(_status)
                _status.success(
                    f"✅ Descargado: `{obs_p.name}` y `{sp_p.name}`  \n"
                    "Pulsa **⚡ Indexar / Actualizar** para incorporarlos."
                )
            except Exception as e:
                _status.error(f"❌ Error: {e}")

        st.metric("Vectores en ChromaDB", index.count())
        st.divider()
        top_k = st.slider("Fragmentos a recuperar (top-k)", 2, 8, 4)

        if st.button("🗑 Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("💬 Chat con tus documentos")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    if user_input := st.chat_input("Pregunta algo sobre tus documentos…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en tus documentos…"):
                result = answer(model, index, user_input, top_k)
            st.markdown(result["answer"])
            _render_sources(result["sources"])

        st.session_state.messages.append({
            "role": "assistant", "content": result["answer"],
            "sources": result["sources"],
        })


def _render_sources(sources: list):
    if not sources:
        return
    with st.expander("📎 Fuentes", expanded=False):
        for s in sources:
            st.caption(f"• **{s['source']}** | pág. {s['page']} | score: {s.get('score','—')}")


if __name__ == "__main__":
    main()
