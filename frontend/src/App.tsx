import { useEffect, useState } from 'react';
import { useStore } from './store/useStore';
import { mockAudioFiles } from './data/mockData';
import { api, toAudioFile } from './api/client';
import Header from './components/Header';
import AudioCatalog from './components/AudioCatalog';
import SearchView from './components/SearchView';
import AnnotationMode from './components/AnnotationMode';
import EvaluationMode from './components/EvaluationMode';

export default function App() {
  const { setFiles, currentView } = useStore();
  const [apiInfo, setApiInfo] = useState<{ ok: boolean; mode: string }>({ ok: false, mode: '' });

  useEffect(() => {
    let cancelled = false;
    // Пробуем загрузить реальные файлы с бэкенда, иначе — моки
    (async () => {
      try {
        const remote = await api.getAllFiles();
        if (cancelled) return;
        if (remote.length > 0) {
          const mapped = remote.map(toAudioFile);
          const st = useStore.getState();
          mapped.forEach(f => st.setAudioUrl(f.id, api.getAudioUrl(f.id)));
          st.setFiles(mapped);
        } else {
          setFiles(mockAudioFiles);
        }
      } catch {
        if (!cancelled) setFiles(mockAudioFiles);
      }
    })();
    // Проверяем доступность бэкенда и его режим
    const poll = () => {
      api.health()
        .then(h => { if (!cancelled) setApiInfo({ ok: true, mode: h.embedder }); })
        .catch(() => { if (!cancelled) setApiInfo({ ok: false, mode: '' }); });
    };
    poll();
    const interval = setInterval(poll, 10000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [setFiles]);

  const renderView = () => {
    switch (currentView) {
      case 'catalog':
        return <AudioCatalog />;
      case 'search':
        return <SearchView />;
      case 'annotate':
        return <AnnotationMode />;
      case 'evaluate':
        return <EvaluationMode />;
      default:
        return <AudioCatalog />;
    }
  };

  const modeLabel = !apiInfo.ok
    ? 'демо-режим'
    : apiInfo.mode === 'laion-clap'
      ? 'подключён · CLAP'
      : 'подключён · keywords';

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      <Header />
      <main className="flex-1 p-6 overflow-hidden">
        {renderView()}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-3 flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <span>SFX Smart Search v0.1</span>
          <span className="text-gray-700">|</span>
          <span>CLAP + Librosa + ChromaDB</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <span className={`w-2 h-2 rounded-full ${apiInfo.ok ? 'bg-green-500' : 'bg-yellow-500'} ${apiInfo.ok ? '' : 'animate-pulse'}`}></span>
            API: {modeLabel}
          </span>
          <span>Electron-ready</span>
        </div>
      </footer>
    </div>
  );
}
