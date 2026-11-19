"""
SFX Smart Search — Backend API
FastAPI + ChromaDB + CLAP + Librosa

Установка:
    pip install fastapi uvicorn chromadb librosa torch laion-clap pydub numpy

Запуск:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import uuid
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings

# === Опциональные импорты (требуют установки) ===
# import librosa
# import torch
# import laion_clap

app = FastAPI(title="SFX Smart Search API", version="0.1.0")

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "app://."),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ChromaDB ===
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
))

collection = chroma_client.get_or_create_collection(
    name="sfx_audio",
    metadata={"hnsw:space": "cosine"},
)


# === Модели ===
class SearchRequest(BaseModel):
    query: str
    n_results: int = 10

class TagUpdateRequest(BaseModel):
    tags: list[str]

class DescriptionUpdateRequest(BaseModel):
    description: str

class FileMetadata(BaseModel):
    id: str
    filename: str
    path: str
    duration: float
    sample_rate: int
    channels: int
    category: str
    tags: list[str]
    description: str
    spectral_features: dict


# === CLAP модель (ленивая загрузка) ===
_clap_model = None

def get_clap_model():
    """Ленивая загрузка CLAP модели"""
    global _clap_model
    if _clap_model is None:
        import laion_clap
        import torch
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        _clap_model.load_ckpt()  # загрузить предобученные веса
    return _clap_model


def get_audio_embedding(audio_path: str) -> list[float]:
    """Получить CLAP embedding из аудиофайла"""
    model = get_clap_model()
    embedding = model.get_audio_embed_from_file(audio_path)
    return embedding.tolist()


def get_text_embedding(text: str) -> list[float]:
    """Получить CLAP embedding из текста"""
    model = get_clap_model()
    embedding = model.get_text_embed(text)
    return embedding.tolist()


# === Librosa анализ ===
def analyze_audio(audio_path: str) -> dict:
    """Спектральный анализ аудио через Librosa"""
    import librosa
    
    y, sr = librosa.load(audio_path, sr=None)
    
    # Высота тона (dominant frequency)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if spectral_centroid < 500:
        pitch = "низкий"
    elif spectral_centroid < 2000:
        pitch = "средний"
    else:
        pitch = "высокий"
    
    # Энергия (RMS)
    rms = np.mean(librosa.feature.rms(y=y))
    if rms < 0.05:
        energy = "low"
    elif rms < 0.15:
        energy = "medium"
    else:
        energy = "high"
    
    # Шум (spectral flatness)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    has_noise = flatness > 0.3
    
    # Реверб (простая эвристика через decay)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    has_reverb = float(np.std(onset_env)) < 0.5
    
    # Длительность
    duration = float(librosa.get_duration(y=y, sr=sr))
    
    return {
        "pitch": pitch,
        "energy": energy,
        "has_noise": has_noise,
        "has_reverb": has_reverb,
        "duration": duration,
        "sample_rate": int(sr),
    }


# === Endpoints ===

@app.post("/api/scan")
async def scan_directory(directory: str):
    """Сканировать директорию и добавить аудиофайлы в ChromaDB"""
    audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.aiff'}
    added = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in audio_extensions:
                continue
            
            filepath = os.path.join(root, filename)
            file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))
            
            # Проверяем, есть ли уже в базе
            existing = collection.get(ids=[file_id])
            if existing['ids']:
                continue
            
            # Анализ аудио
            try:
                features = analyze_audio(filepath)
                embedding = get_audio_embedding(filepath)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
            
            # Добавляем в ChromaDB
            collection.add(
                ids=[file_id],
                embeddings=[embedding],
                metadatas=[{
                    "filename": filename,
                    "path": filepath,
                    "duration": features["duration"],
                    "sample_rate": features["sample_rate"],
                    "channels": 1,  # TODO: определить через librosa
                    "category": "Без категории",
                    "tags": "",
                    "description": "",
                    "pitch": features["pitch"],
                    "energy": features["energy"],
                    "has_noise": features["has_noise"],
                    "has_reverb": features["has_reverb"],
                }],
                documents=[filename],  # базовый документ
            )
            added += 1
    
    return {"status": "ok", "added": added}


@app.post("/api/search")
async def search_by_text(request: SearchRequest):
    """Семантический поиск по текстовому описанию"""
    query_embedding = get_text_embedding(request.query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.n_results,
    )
    
    # Форматируем ответ
    items = []
    if results['ids'] and results['ids'][0]:
        for i, file_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if results.get('distances') else 0
            # cosine distance -> similarity score
            score = 1 - distance
            
            items.append({
                "id": file_id,
                "score": score,
                "file": {
                    "id": file_id,
                    "filename": metadata["filename"],
                    "duration": metadata["duration"],
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "category": metadata["category"],
                    "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                    "description": metadata["description"],
                    "spectral_features": {
                        "pitch": metadata["pitch"],
                        "energy": metadata["energy"],
                        "has_noise": metadata["has_noise"],
                        "has_reverb": metadata["has_reverb"],
                    }
                }
            })
    
    return {"results": items, "query": request.query}


@app.post("/api/search/audio")
async def search_by_audio(file: UploadFile = File(...), n_results: int = 10):
    """Поиск по референсному аудиофайлу"""
    import tempfile
    
    # Сохраняем загруженный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Получаем embedding референса
        ref_embedding = get_audio_embedding(tmp_path)
        
        # Ищем похожие
        results = collection.query(
            query_embeddings=[ref_embedding],
            n_results=n_results,
        )
        
        items = []
        if results['ids'] and results['ids'][0]:
            for i, file_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if results.get('distances') else 0
                score = 1 - distance
                
                items.append({
                    "id": file_id,
                    "score": score,
                    "file": {
                        "id": file_id,
                        "filename": metadata["filename"],
                        "duration": metadata["duration"],
                        "sample_rate": metadata["sample_rate"],
                        "channels": metadata["channels"],
                        "category": metadata["category"],
                        "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                        "description": metadata["description"],
                    }
                })
        
        return {"results": items}
    finally:
        os.unlink(tmp_path)


@app.get("/api/files")
async def get_all_files():
    """Получить все файлы из базы"""
    all_data = collection.get()
    
    files = []
    for i, file_id in enumerate(all_data['ids']):
        metadata = all_data['metadatas'][i]
        files.append({
            "id": file_id,
            "filename": metadata["filename"],
            "duration": metadata["duration"],
            "sample_rate": metadata["sample_rate"],
            "channels": metadata["channels"],
            "category": metadata["category"],
            "tags": metadata["tags"].split(",") if metadata["tags"] else [],
            "description": metadata["description"],
            "spectral_features": {
                "pitch": metadata["pitch"],
                "energy": metadata["energy"],
                "has_noise": metadata["has_noise"],
                "has_reverb": metadata["has_reverb"],
            }
        })
    
    return {"files": files}


@app.put("/api/files/{file_id}/tags")
async def update_tags(file_id: str, request: TagUpdateRequest):
    """Обновить теги файла"""
    collection.update(
        ids=[file_id],
        metadatas=[{"tags": ",".join(request.tags)}],
    )
    return {"status": "ok"}


@app.put("/api/files/{file_id}/description")
async def update_description(file_id: str, request: DescriptionUpdateRequest):
    """Обновить описание файла (и переиндексировать embedding)"""
    # Обновляем метаданные
    collection.update(
        ids=[file_id],
        metadatas=[{"description": request.description}],
        documents=[request.description],
    )
    return {"status": "ok"}


@app.get("/api/files/{file_id}/audio")
async def get_audio(file_id: str):
    """Получить аудиофайл для стриминга в плеер"""
    results = collection.get(ids=[file_id])
    if not results['ids']:
        raise HTTPException(status_code=404, detail="File not found")
    
    filepath = results['metadatas'][0]['path']
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    from fastapi.responses import FileResponse
    return FileResponse(filepath, media_type="audio/wav")


# === Запуск ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
