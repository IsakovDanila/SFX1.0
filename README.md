# SFX Smart Search — MVP v0.1 (слияние готово ✅)

Локальный семантический поиск по библиотеке звуковых эффектов.
**Фронтенд команды** (React + TS + Tailwind + Zustand + WaveSurfer v8)
подключён к **рабочему бэкенду** (FastAPI → CLAP/heuristic → JSON/ChromaDB → Librosa).

## Статус (по MoSCoW из презы)

| Задача | Статус |
|---|---|
| D1 Окружение и pipeline | ✅ работает на CPU, без GPU |
| R2 Проверка поиска, Recall@10 ≥ 0.5 | ✅ **1.0** (24 запроса: EN + RU) |
| D3 ChromaDB + FastAPI | ✅ FastAPI по контракту команды; ChromaDB — опционально |
| D2 Прототип Web-UI | ✅ ваш фронт, поиск/оценка/разметка — на живом API |
| D4 README | ✅ этот файл |
| CLAP full-режим | 🔌 `SFX_USE_CLAP=1` (нужны torch + ~4 GB) |

## Быстрый старт

**Windows:** двойной клик `install.bat` (один раз), затем `start-backend.bat`
и `start-frontend.bat`. **Mac/Linux:** `./install.sh`, `./start-backend.sh`,
`./start-frontend.sh`. Либо вручную:

```bash
# 1. Бэкенд
pip install -r backend/requirements.txt
python scripts/generate_demo_sounds.py   # 40 .wav + tags.csv (если нет своих)
cd backend && uvicorn app.main:app --port 8000

# 2. Фронтенд (вторым терминалом, из корня)
cd frontend && npm install && npm run dev  # UI: http://localhost:5173
```

В UI: «Каталог» → «Импорт файлов» → «Папка на сервере» → `../data/demo_sfx`
→ «Сканировать». Дальше — поиск словами или референсом, разметка, оценка.

Своя папка звуков: тот же путь, вписать свой путь (на машине бэкенда).

Своя папка подробно:
- демо-звуки необязательны: `data/demo_sfx` можно удалить; каждый скан
  заменяет индекс целиком, а при старте бэкенд сам выкидывает записи
  удалённых файлов (или принудительно `POST /api/index/clear`)
- категории = имена подпапок, теги = слова из имён файлов
- точная разметка: `python scripts/make_tags_template.py "D:\MySFX"` создаст
  `tags.csv` в каждой подпапке — допишите описания в Excel и сканируйте
- поиск билингвальный: русские запросы находят английские имена файлов

## Структура

```
sfx-smart-search/
├── frontend/                 # ВАШ код (TS): App, api/client, store, components/*,
│                             # electron/main.js + preload.js, IMPORT_GUIDE.md
├── backend/app/
│   ├── main.py               # FastAPI: контракт команды (/api/files, PUT tags, ...)
│   ├── store.py              # JSON/numpy store + shape под client.ts
│   ├── embeddings.py         # CLAP (full) / heuristic EN+RU (demo)
│   └── audio_features.py     # Librosa: pitch, energy, noise, reverb, MFCC
├── scripts/                  # generate_demo_sounds, demo_queries (24, EN+RU),
│                             # evaluate_recall
├── data/                     # demo_sfx (40 .wav + tags.csv), index.json
├── docs/original/            # ваши исходники как были (main.py, README, GUIDE)
└── .jsx-backup/              # мой временный JSX-каркас (до слияния)
```

## API-контракт (ваш, реализован полностью)

| Метод | Путь | Ответ |
|---|---|---|
| `POST` | `/api/scan?directory=` | `{"status":"ok","added":N}` (синхронно) |
| `POST` | `/api/scan {"folder"}` | `{"accepted":true}` + прогресс в `/api/scan/status` |
| `POST` | `/api/index/clear` | очистить индекс (напр. после удаления демо-папки) |
| `POST` | `/api/search {"query","n_results"}` | `{results:[{id,score,file:{...}}]}` |
| `POST` | `/api/search/audio` (multipart) | `{results:[...]}` |
| `GET` | `/api/files` | `{files:[...], count}` |
| `PUT` | `/api/files/{id}/tags` | `{"status":"ok"}` |
| `PUT` | `/api/files/{id}/description` | `{"status":"ok"}` |
| `GET` | `/api/files/{id}/audio` | стрим для WaveSurfer |
| `GET` | `/api/health` | индикатор в футере |
| `POST` | `/api/evaluate` | Recall@5/Recall@10 (скрипт/интеграция) |

`file`: `{id, filename, duration, sample_rate, channels, category, tags,
description, spectral_features:{pitch(низкий/средний/высокий),
energy(low/medium/high), has_noise, has_reverb}}`.

## Метрики (R2 / HADI-3)

```bash
python scripts/evaluate_recall.py
```

- **Recall@5 = 1.0, Recall@10 = 1.0** (24 запроса: 20 EN + 4 RU) → гейт пройден
- Поиск **< 1 мс**, индексация 40 файлов ~1.5 с на CPU;
  повторный скан инкрементальный (без изменений — мгновенно, теги целы)
- Demo-режим билингвальный: понимает «metallic hit» и «тяжёлый
  металлический удар» (синонимы + толерантность к склонениям)

## Режимы эмбеддера

- **Demo (по умолчанию):** текст — синонимы+теги (EN+RU), сходство аудио —
  косинус по MFCC. В футере «API: демо-режим» только если бэкенд недоступен;
  при запущенном бэкенде — «API: подключён».
- **Full CLAP (рекомендуется для реальных библиотек):** `install-ml.bat`
  (CPU torch ~200 МБ + laion-clap), затем `start-backend-clap.bat`.
  Первый запуск докачивает веса `630k-audioset-best.pt` (~1.9 ГБ) и
  токенизатор RoBERTa. Требования: ~4 ГБ свободной RAM, ~3 ГБ на диске
  (torch + веса). Индексация на CPU: ~1–2 с/файл (96 файлов ≈ 2–4 мин
  один раз, с прогресс-баром); повторные сканы инкрементальные.
  Поиск гибридный (CLAP + keywords через RRF): английская семантика +
  русские теги/синонимы. CLAP понимает английский — для лучшего качества
  запросы на английском или теги на русском. Тот же API и скрипты оценки.
- **Авто-тегирование (только CLAP-режим):** при скане каждый файл сравнивается
  со словарём из ~60 меток (речь/мемы, музыка, эффекты — см.
  `backend/app/autotag.py`) и топ-совпадения пишутся в `auto_tags` с оценками,
  а также добавляются в обычные теги (русская метка + короткая английская),
  чтобы их сразу находил поиск. `tags.csv` по-прежнему главнее: где он есть,
  авто-теги только показываются в «Разметке» как гипотезы. Честные границы:
  речь размечается по звучанию (мужская/женская, крик, смех), а не по смыслу
  слов — расшифровка содержания мемных фраз требует ASR (Whisper), это
  следующий шаг. Настройка: `SFX_AUTOTAG_TOPK` (по умолчанию 3),
  `SFX_AUTOTAG_THRESHOLD` (по умолчанию 0.12), свой словарь —
  `SFX_LABELS_PATH` (JSON вида `[{"en": "...", "ru": "...", "group": "..."}]`).

## Связь с рисками из презы (слайд 17)

1. **CLAP/сленг** → fallback-фильтры, ручные теги (теперь персистентные),
   RU-синонимы в demo-режиме.
2. **Слабый ПК** → demo на CPU; скан с прогрессом; CLAP опционален.
3. **«Чёрный ящик»** → score % у результатов, audio-to-audio референсом,
   экран «Оценка» на живых данных.

## Следующие шаги

1. CustDev-интервью (R1) — скрипт и план в презе (слайды 2–3).
2. Набрать 200 реальных .wav → `evaluate_recall.py` → сравнить demo vs CLAP.
3. Electron: `npm install -D electron`, `npx electron electron/main.js`
   (нужен `VITE_API_URL=http://localhost:8000` в prod-сборке; добавить icon.png).
4. Опционально: почистить неиспользуемые deps шаблона, подключить ChromaDB.
