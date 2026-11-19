# Frontend — React + TypeScript + Tailwind (код команды, подключён к API)

## Запуск

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (нужен запущенный бэкенд :8000)
npm run typecheck  # проверка типов (tsc --noEmit)
npm run build      # production-сборка в dist/
```

`/api/*` проксируется на FastAPI (см. `vite.config.js`), в браузере
используются относительные URL. Для Electron/prod укажите адрес API явно:

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

## Структура (ваша)

```
frontend/
├── src/
│   ├── App.tsx               # вкладки + футер + загрузка файлов с API
│   ├── main.tsx
│   ├── index.css             # Tailwind v4 (@import "tailwindcss")
│   ├── api/client.ts         # REST-клиент + маппер toAudioFile
│   ├── store/useStore.ts     # Zustand-стор
│   ├── data/mockData.ts      # моки (fallback, если API недоступен)
│   ├── utils/audioAnalysis.ts# анализ через Web Audio API (импорт)
│   └── components/           # Header, AudioCatalog, SearchView,
│                             # AnnotationMode, EvaluationMode,
│                             # AudioPlayer (WaveSurfer v8), ImportModal
├── electron/
│   ├── main.js               # ваш (dev грузит http://localhost:5173)
│   └── preload.js            # создан при слиянии (был утерян)
├── IMPORT_GUIDE.md           # ваш гайд по импорту
└── vite.config.js            # + прокси /api -> :8000, порт 5173
```

## Что изменено при слиянии (ваш код — основа)

Подключение к живому API (раньше поиск/оценка работали на моках):

- `api/client.ts` — относительные URL по умолчанию (`?? ''`, было
  `|| 'http://localhost:8000'`); `healthCheck` → `GET /api/health`;
  добавлен маппер `toAudioFile` (snake_case → camelCase).
- `App.tsx` — при старте тянет `getAllFiles()` и проставляет реальные
  URL в плеер; при недоступности API — моки как раньше.
- `SearchView.tsx` — реальный `searchByText` / `searchByAudio`, fallback
  на моки; чипы-примеры стали кликабельными.
- `EvaluationMode.tsx` — «Выполнить поиск» идёт в API, fallback на моки.
- `AnnotationMode.tsx` — теги/описания сохраняются в API (`PUT`), локально
  работают как раньше.
- `AudioCatalog.tsx` — категории строятся по реальным файлам.
- `ImportModal.tsx` — **фикс краша**: хук `useStore()` перенесён выше
  раннего `return null` (иначе React ронял модалку при закрытии);
  добавлена секция «Папка на сервере» → `scanDirectory` + обновление каталога.
- `index.html` — **фикс**: `/src/main.jsx` → `/src/main.tsx` (со старым
  путём приложение не стартовало); убран шаблонный мусор, `lang="ru"`.
- `vite.config.js` — порт 5173 (как в README и Electron), прокси `/api`,
  `allowedHosts: true` для песочницы.
- `electron/preload.js` — создан (в `main.js` на него ссылка, файла не было).

Без изменений: `Header`, `AudioPlayer`, `mockData`, `useStore`,
`audioAnalysis`, `main.tsx`, `index.css`, `tsconfig`, `package.json`
(кроме имени), `electron/main.js`.

## Заметки

- `package.json` оставлен как есть (включая неиспользуемые пока deps
  шаблона: dnd-kit, supabase, framer-motion и др.) — можно почистить позже.
- `electron/icon.png` отсутствует в проекте (некритично, Electron
  стартует и без иконки) — добавьте файл, когда будет логотип.
- Импорт файлов в браузере (Web Audio API) сохранён как был; серверное
  сканирование — новая секция в том же окне импорта.
