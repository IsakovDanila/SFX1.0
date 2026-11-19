# SFX Smart Search — Умный каталог аудиоэффектов

## 🚀 Как запустить проект

### Frontend (этот репозиторий)

```bash
# 1. Установить зависимости
npm install

# 2. Запустить dev-сервер (горячая перезагрузка)
npm run dev

# 3. Собрать production-версию
npm run build

# 4. Предпросмотр production-сборки
npm run preview
```

После `npm run dev` откройте http://localhost:5173

### Desktop (Electron)

```bash
# Установить Electron (одноразово)
npm install -D electron electron-builder

# Запустить в desktop-режиме
npx electron electron/main.js

# Собрать .exe / .dmg / .AppImage
npx electron-builder
```

---

## 🏗 Архитектура

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND (React)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Каталог  │ │  Поиск   │ │ Разметка │ │ Оценка │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │            │            │            │       │
│  ┌────┴────────────┴────────────┴────────────┴────┐ │
│  │              WaveSurfer.js Player               │ │
│  └──────────────────────┬─────────────────────────┘ │
└─────────────────────────┼───────────────────────────┘
                          │ REST API (fetch)
                          ▼
┌─────────────────────────────────────────────────────┐
│                 BACKEND (FastAPI)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ /api/search  │  │ /api/annotate│  │ /api/scan │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│         │                 │                 │       │
│  ┌──────┴─────────────────┴─────────────────┴─────┐ │
│  │              CLAP / AudioCLIP Model             │ │
│  │         (векторные эмбеддинги аудио)            │ │
│  └──────────────────────┬─────────────────────────┘ │
│                         │                           │
│  ┌──────────────────────┴─────────────────────────┐ │
│  │              ChromaDB (векторная БД)            │ │
│  │   embeddings + metadata + audio file paths      │ │
│  └────────────────────────────────────────────────┘ │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │         Librosa (спектральный анализ)           │ │
│  │   pitch, energy, noise, reverb detection        │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 🗄 ChromaDB — схема данных

```python
# Пример коллекции в ChromaDB
collection = chroma_client.get_or_create_collection(
    name="sfx_audio",
    metadata={"hnsw:space": "cosine"}  # косинусное сходство
)

# Добавление аудиофайла
collection.add(
    ids=["sfx_hit_01"],
    embeddings=[0.12, -0.34, ...],  # CLAP embedding (512 dim)
    metadatas=[{
        "filename": "sfx_hit_heavy_metal_01.wav",
        "category": "Боевые эффекты",
        "duration": 1.2,
        "sample_rate": 44100,
        "pitch": "низкий",
        "energy": "high",
        "has_noise": True,
        "has_reverb": True,
        "tags": "удар,металл,тяжёлый,эхо",
        "description": "Тяжёлый металлический удар с длинным эхом"
    }],
    documents=["Тяжёлый металлический удар с длинным эхом, подходит для сцен сражений"]
)

# Семантический поиск
results = collection.query(
    query_texts=["гулкий шаг по деревянному полу"],
    n_results=10
)
```

---

## 🔌 API Endpoints (FastAPI)

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/scan` | Сканирование директории с аудио |
| POST | `/api/search` | Семантический поиск по тексту |
| POST | `/api/search/audio` | Поиск по референсному .wav |
| GET | `/api/files` | Получить список всех файлов |
| PUT | `/api/files/{id}/tags` | Обновить теги файла |
| PUT | `/api/files/{id}/description` | Обновить описание |
| GET | `/api/files/{id}/audio` | Получить аудио для плеера |
| POST | `/api/evaluate` | Сохранить оценку качества |

---

## 📦 Стек технологий

### Frontend
- **React 18** + TypeScript
- **Zustand** — стейт-менеджмент
- **WaveSurfer.js** — waveform-плеер
- **Tailwind CSS** — стилизация
- **Lucide React** — иконки
- **Vite** — сборщик

### Backend (планируется)
- **Python 3.10+** + FastAPI
- **CLAP** (LAION Audio) — эмбеддинги аудио
- **Librosa** — спектральный анализ
- **ChromaDB** — векторная база данных
- **Pydub** — обработка аудио

### Desktop (опционально)
- **Electron** — обёртка в десктопное приложение
- **electron-builder** — сборка .exe/.dmg/.AppImage
