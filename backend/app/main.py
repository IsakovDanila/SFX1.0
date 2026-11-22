"""SFX Smart Search — FastAPI backend.

Implements the team's frontend contract (client.ts):
  POST /api/scan?directory=... | {"folder": ...}
  POST /api/search {"query", "n_results"|"top_k"}
  POST /api/search/audio (multipart file)
  GET  /api/files
  PUT  /api/files/{id}/tags  |  PUT /api/files/{id}/description
  GET  /api/files/{id}/audio
plus extras: /api/health, /api/library, /api/similar, /api/evaluate ...

Run (from backend/):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
import tempfile

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import env_loader

env_loader.load_env()   # .env с HF_TOKEN и настройками (в git не попадает)

from . import audio_features as af
from . import clap_runtime
from . import embeddings as E
from .store import JsonStore, get_store

app = FastAPI(title="SFX Smart Search API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

store: JsonStore = get_store()

scan_state: dict = {"status": "idle", "total": 0, "done": 0, "current": "",
                    "result": None}


def _register_index_cleanup() -> None:
    """Стирать index.json при остановке бэкенда.

    Векторы в индексе привязаны к текущему режиму (13-d эвристика против
    512-d CLAP) и к mtime файлов. После остановки это кэш без ценности, а
    при смене режима — источник ошибок размерности (cosine 13 vs 512 ->
    HTTP 500). Поэтому по умолчанию удаляем; SFX_INDEX_KEEP=1 отключает.
    """
    def _cleanup(*_args) -> None:
        if os.getenv("SFX_INDEX_KEEP") == "1":
            return
        try:
            if os.path.exists(store.path):
                os.unlink(store.path)
                print(f"[store] index.json удалён при остановке: {store.path}",
                      flush=True)
        except OSError as e:  # noqa: BLE001 - не мешаем завершению процесса
            print(f"[store] не удалось удалить индекс: {e}", flush=True)

    atexit.register(_cleanup)
    # SIGTERM: docker stop / kill. SIGINT обычно перехватывает uvicorn,
    # поэтому дублируем через atexit.
    for sig in (signal.SIGTERM,):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, lambda *a: (_cleanup(), prev(*a)
                                           if callable(prev) else sys.exit(0)))
        except (ValueError, OSError):
            pass   # не главный поток — atexit хватит


@app.on_event("startup")
def _startup() -> None:
    """CLAP грузим в фоне: сервер должен отвечать сразу, а не через 5 минут.

    Когда модель готова, стор переключается на неё и переиндексирует
    библиотеку (у CLAP-векторов другая размерность).
    """
    _register_index_cleanup()
    clap_runtime.RUNTIME.start()
    if not clap_runtime.ClapRuntime.enabled():
        return

    def _watch() -> None:
        import time
        while True:
            time.sleep(2)
            st = clap_runtime.RUNTIME.state
            if st == "ready":
                emb = clap_runtime.RUNTIME.embedder()
                if emb is not None and emb.name != store.embedder.name:
                    store.use_embedder(emb)
                return
            if st in ("failed", "disabled"):
                print(f"[main] CLAP недоступен ({st}): "
                      f"{clap_runtime.RUNTIME.error or 'нет причины'}. "
                      f"Работаем на эвристике.", flush=True)
                return

    import threading
    threading.Thread(target=_watch, name="clap-watch", daemon=True).start()


# ---------- schemas ----------
class ScanRequest(BaseModel):
    folder: str = ""


class SearchRequest(BaseModel):
    query: str = ""
    top_k: int = 10
    n_results: int | None = None  # frontend sends n_results


class TagsRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


class DescriptionRequest(BaseModel):
    description: str = ""


class AnnotateRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


# ---------- meta ----------
@app.get("/api/health")
def health():
    """Короткий контракт сохранён (status/embedder/indexed) + блок clap."""
    health = {"status": "ok", "embedder": store.embedder.name,
              "indexed": len(store.items), "version": app.version,
              "clap": clap_runtime.RUNTIME.status()}
    # диагностика: почему индекс пуст/отброшен и что сломалось при переиндексации
    if getattr(store, "index_reset_reason", ""):
        health["index_reset_reason"] = store.index_reset_reason
    if getattr(store, "reindex_error", ""):
        health["reindex_error"] = store.reindex_error
    return health


@app.get("/api/clap/status")
def clap_status():
    """Состояние CLAP для плашки в UI: loading / ready / failed / disabled."""
    st = clap_runtime.RUNTIME.status()
    st["active_embedder"] = store.embedder.name
    st["indexed"] = len(store.items)
    return st


@app.post("/api/clap/retry")
def clap_retry():
    """Повторить загрузку (например, после того как пользователь скачал веса)."""
    if not clap_runtime.ClapRuntime.enabled():
        raise HTTPException(400, "CLAP выключен: запустите с SFX_USE_CLAP=1 "
                                 "(см. start-backend-clap.sh)")
    if clap_runtime.RUNTIME.state == "loading":
        return {"status": "already_loading", **clap_runtime.RUNTIME.status()}
    clap_runtime.RUNTIME.state = "disabled"
    clap_runtime.RUNTIME.start()
    return {"status": "restarted", **clap_runtime.RUNTIME.status()}


@app.get("/")
def root():
    return {"service": "SFX Smart Search API", "docs": "/docs",
            "health": "/api/health"}


# ---------- scan / index ----------
def _run_scan(folder: str):
    scan_state.update(status="running", total=0, done=0, current="", result=None)

    def cb(done, total, current):
        scan_state.update(total=total, done=done, current=current)

    try:
        res = store.scan_folder(folder, progress_cb=cb)
        scan_state.update(status="done", result=res)
    except Exception as e:  # noqa: BLE001 - report to UI
        scan_state.update(status="error", result={"error": str(e)})


@app.post("/api/scan")
def scan(bg: BackgroundTasks, directory: str | None = None,
         req: ScanRequest | None = None):
    """Dual mode: ?directory=... scans synchronously (frontend contract,
    returns {added}); JSON {"folder"} scans in background (progress via
    /api/scan/status)."""
    folder = (directory or (req.folder if req else "") or "").strip()
    if not folder:
        raise HTTPException(400, "Provide ?directory=... or JSON {folder}")
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise HTTPException(400, f"Folder not found: {folder}")
    if scan_state["status"] == "running":
        raise HTTPException(409, "Scan already running")
    if directory is not None:
        _run_scan(folder)
        res = scan_state.get("result") or {}
        if scan_state["status"] == "error":
            raise HTTPException(500, str(res.get("error", "scan failed")))
        return {"status": "ok", "added": res.get("indexed", 0),
                "new": res.get("added", 0), "reused": res.get("reused", 0)}
    bg.add_task(_run_scan, folder)
    return {"accepted": True, "folder": folder}


@app.get("/api/scan/status")
def scan_status():
    return scan_state


@app.post("/api/index/clear")
def clear_index():
    """Drop the whole index (e.g. after deleting the demo folder)."""
    n = len(store.items)
    store.clear()
    store.save()
    scan_state.update(status="idle", total=0, done=0, current="", result=None)
    return {"status": "ok", "cleared": n}


# ---------- files ----------
@app.get("/api/files")
def files():
    return {"files": [JsonStore.their_shape(it) for it in store.items],
            "count": len(store.items)}


@app.get("/api/library")
def library(category: str = "", q: str = ""):
    return {"count": len(store.items), "items": store.library(category, q),
            "categories": store.categories()}


def _serve_audio(item_id: str):
    it = store.get(item_id)
    if not it or not os.path.exists(it["path"]):
        raise HTTPException(404, "Audio not found")
    ext = os.path.splitext(it["path"])[1].lower()
    media = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
             "ogg": "audio/ogg", "oga": "audio/ogg", "m4a": "audio/mp4",
             "aif": "audio/aiff", "aiff": "audio/aiff"}.get(ext.lstrip("."), "audio/wav")
    return FileResponse(it["path"], media_type=media, filename=it["filename"])


@app.get("/api/files/{item_id}/audio")
def file_audio(item_id: str):
    return _serve_audio(item_id)


@app.get("/api/audio/{item_id}")
def audio(item_id: str):
    return _serve_audio(item_id)


# ---------- search (text + audio) ----------
@app.post("/api/search")
def search(req: SearchRequest):
    if not store.items:
        raise HTTPException(400, "Library is empty — scan a folder first")
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    k = int(req.n_results or req.top_k or 10)
    k = max(1, min(k, 50))
    return store.search_text(req.query, k)


def _vector_from_upload(content: bytes, suffix: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        feats = af.analyze(tmp_path)
        if isinstance(store.embedder, E.CLAPEmbedder):
            return np.asarray(store.embedder.audio_vector_file(tmp_path))
        return np.asarray(store.embedder.audio_vector(feats))
    finally:
        os.unlink(tmp_path)


def _search_by_upload(content: bytes, suffix: str, top_k: int,
                      query: str = "") -> list[dict]:
    """Поиск по загруженному файлу. `query` намеренно не используется.

    Складывание референса и описания ухудшало результат: при неточном
    описании неверный файл поднимался выше правильного (вес текста 0.8
    против 1.0 у звука). Параметр оставлен в сигнатуре, чтобы старый
    фронтенд не получал 422.
    """
    return store.search_vector(_vector_from_upload(content, suffix), top_k)


@app.post("/api/search/audio")
async def search_audio(file: UploadFile = File(...), n_results: int = 10,
                       query: str = ""):
    if not store.items:
        raise HTTPException(400, "Library is empty — scan a folder first")
    suffix = os.path.splitext(file.filename or "ref.wav")[1] or ".wav"
    return {"results": _search_by_upload(await file.read(), suffix,
                                         max(1, min(n_results, 50)), query)}


@app.post("/api/search-by-audio")
async def search_by_audio(file: UploadFile = File(...), top_k: int = 10):
    if not store.items:
        raise HTTPException(400, "Library is empty — scan a folder first")
    suffix = os.path.splitext(file.filename or "ref.wav")[1] or ".wav"
    results = _search_by_upload(await file.read(), suffix, max(1, min(top_k, 50)))
    return {"mode": store.embedder.name, "top_k": top_k, "results": results}


@app.get("/api/similar/{item_id}")
def similar(item_id: str, top_k: int = 10):
    """'Find similar' (deck risk mitigation: user trust)."""
    if not store.get(item_id):
        raise HTTPException(404, "Item not found")
    return {"item_id": item_id, "results": store.similar(item_id, top_k)}


# ---------- annotate ----------
@app.put("/api/files/{item_id}/tags")
def put_tags(item_id: str, req: TagsRequest):
    if not store.annotate(item_id, req.tags):
        raise HTTPException(404, "Item not found")
    return {"status": "ok"}


@app.put("/api/files/{item_id}/description")
def put_description(item_id: str, req: DescriptionRequest):
    if store.set_description(item_id, req.description) is None:
        raise HTTPException(404, "Item not found")
    return {"status": "ok"}


@app.post("/api/annotate/{item_id}")
def annotate(item_id: str, req: AnnotateRequest):
    it = store.annotate(item_id, req.tags)
    if not it:
        raise HTTPException(404, "Item not found")
    return it


# ---------- evaluation (Оценка screen, HADI-3 / R2) ----------
@app.post("/api/evaluate")
def evaluate(top_k: int = 10):
    """Run expert queries, return Recall@5/Recall@10 (deck gate >= 0.5)."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..", "scripts")))
    import evaluate_recall as ev
    return ev.run(store=store, top_k=top_k)
