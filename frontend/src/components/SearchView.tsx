import { useState } from 'react';
import { Search, Upload, X, Mic, Loader2, ArrowUpDown, SearchX } from 'lucide-react';
import { useStore } from '../store/useStore';
import { api, toAudioFile } from '../api/client';
import AudioPlayer from './AudioPlayer';

const EXAMPLES = ['гулкий шаг по дереву', 'кинематографичный взрыв', 'мягкий клик интерфейса'];

export default function SearchView() {
  const { searchQuery, setSearchQuery, searchResults, setSearchResults, isSearching, setIsSearching, selectedFile, setSelectedFile, setCurrentView } = useStore();
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [sortBy, setSortBy] = useState<'score' | 'duration' | 'name'>('score');
  const [hasSearched, setHasSearched] = useState(false);
  const [searchError, setSearchError] = useState('');

  const handleSearch = async (q?: string) => {
    const text = (q ?? searchQuery).trim();
    if (!text && !referenceFile) return;
    if (q !== undefined) setSearchQuery(q);
    setIsSearching(true);
    setSearchError('');
    try {
      // Референс + описание складываются: файл задаёт «похожесть»,
      // текст уточняет её. Если заполнено только одно — работает одно.
      const res = referenceFile
        ? await api.searchByAudio(referenceFile, 10, text)
        : await api.searchByText(text, 10);
      setSearchResults(res.map(r => ({
        file: toAudioFile(r.file),
        // Показываем калиброванный confidence с бэкенда, если он есть:
        // сырой score (косинус/RRF) для пользователя не шкала «похожести».
        score: Math.max(0, Math.min(1,
          typeof r.confidence === 'number' ? r.confidence / 100 : r.score)),
      })));
      // Проставляем реальные URL для плеера
      const st = useStore.getState();
      res.forEach(r => st.setAudioUrl(r.file.id, api.getAudioUrl(r.file.id)));
    } catch (e) {
      // Никаких подставных результатов: если бэкенд недоступен —
      // показываем пусто и объясняем причину.
      setSearchResults([]);
      setSearchError(
        e instanceof Error && e.message.includes('Library is empty')
          ? 'Индекс пуст — сначала просканируйте папку в разделе «Каталог».'
          : 'Бэкенд недоступен. Он запущен? Проверьте http://localhost:8000/api/health',
      );
    } finally {
      setIsSearching(false);
      setHasSearched(true);
    }
    setCurrentView('search');
  };

  const sorted = [...searchResults].sort((a, b) => { if (sortBy === 'score') return b.score - a.score; if (sortBy === 'duration') return a.file.duration - b.file.duration; return a.file.filename.localeCompare(b.file.filename); });

  return (
    <div className="flex flex-col h-full">
      <div className="w-full max-w-3xl mx-auto">
        <div className="relative flex items-center gap-2">
          <div className="flex-1 relative">
            <Search size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
            <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Опишите нужный звук... (напр. «гулкий шаг по деревянному полу»)"
              className="w-full pl-12 pr-4 py-3.5 bg-gray-800 border border-gray-600 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-violet-500 text-sm" />
          </div>
          <button onClick={() => handleSearch()} className="px-6 py-3.5 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-medium text-sm shadow-lg shadow-violet-600/20">Найти</button>
        </div>
        <div className="flex items-center gap-3 mt-3">
          <label className="flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-300 hover:text-white hover:border-violet-500 text-xs cursor-pointer">
            <Upload size={14} />Загрузить референс .wav
            <input type="file" accept=".wav,.mp3,.ogg,.flac" onChange={(e) => e.target.files?.[0] && setReferenceFile(e.target.files[0])} className="hidden" />
          </label>
          {referenceFile && (<div className="flex items-center gap-2 px-3 py-1.5 bg-violet-900/50 border border-violet-600 rounded-lg text-xs text-violet-300"><Mic size={12} /><span className="max-w-[200px] truncate">{referenceFile.name}</span><button onClick={() => setReferenceFile(null)}><X size={12} /></button></div>)}
          {referenceFile && searchQuery.trim() && (<span className="text-[11px] text-gray-500">референс + описание: найдёт похожее и сузит по тексту</span>)}
        </div>
      </div>

      {isSearching ? (
        <div className="flex-1 flex items-center justify-center"><div className="text-center"><Loader2 size={40} className="text-violet-400 animate-spin mx-auto" /><p className="text-gray-400 text-sm mt-4">Поиск через CLAP embeddings...</p></div></div>
      ) : !hasSearched ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-violet-600/20 to-cyan-600/20 border border-violet-600/30 flex items-center justify-center"><Search size={28} className="text-violet-400" /></div>
            <h3 className="text-white text-lg font-medium mb-2">Семантический поиск звуков</h3>
            <p className="text-gray-400 text-sm mb-4">Опишите нужный звук или загрузите референсный .wav</p>
            <div className="flex flex-wrap gap-2 justify-center">{EXAMPLES.map(e => (<button key={e} onClick={() => handleSearch(e)} className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white hover:border-violet-500 text-xs transition-all">«{e}»</button>))}</div>
          </div>
        </div>
      ) : searchResults.length > 0 ? (
        <div className="mt-6 flex gap-4 flex-1 min-h-0">
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-white text-sm font-medium">Найдено: {searchResults.length} результатов</h3>
              <div className="flex items-center gap-2 text-xs text-gray-400"><ArrowUpDown size={12} /><button onClick={() => setSortBy('score')} className={`hover:text-white ${sortBy === 'score' ? 'text-violet-400' : ''}`}>Релевантность</button><button onClick={() => setSortBy('duration')} className={`hover:text-white ${sortBy === 'duration' ? 'text-violet-400' : ''}`}>Длительность</button></div>
            </div>
            <div className="flex-1 overflow-auto rounded-xl border border-gray-700">
              {sorted.map((r, i) => (
                <div key={r.file.id} onClick={() => setSelectedFile(r.file)} className={`p-4 border-b border-gray-800 cursor-pointer transition-colors ${selectedFile?.id === r.file.id ? 'bg-violet-900/20' : 'hover:bg-gray-800/50'}`}>
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gray-700 flex items-center justify-center text-gray-400 text-xs font-mono">#{i + 1}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2"><p className="text-white text-sm font-medium truncate">{r.file.filename}</p><span className="text-[10px] px-1.5 py-0.5 bg-gray-700 text-gray-300 rounded">{r.file.category}</span></div>
                      <p className="text-gray-400 text-xs mt-1">{r.file.description}</p>
                      <div className="flex items-center gap-3 mt-2"><span className="text-gray-500 text-[10px]">{r.file.duration.toFixed(1)}s</span></div>
                    </div>
                    <div className="flex-shrink-0 text-right"><div className="text-lg font-bold text-violet-400">{(r.score * 100).toFixed(0)}%</div><div className="text-[10px] text-gray-500">match</div></div>
                  </div>
                  <div className="mt-2 ml-11"><div className="h-1 bg-gray-700 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-violet-600 to-cyan-500 rounded-full" style={{ width: `${r.score * 100}%` }} /></div></div>
                </div>
              ))}
            </div>
          </div>
          <div className="w-80 flex-shrink-0"><AudioPlayer file={selectedFile} /></div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-800 border border-gray-700 flex items-center justify-center"><SearchX size={28} className="text-gray-500" /></div>
            <h3 className="text-white text-lg font-medium mb-2">
              {searchError ? 'Поиск недоступен' : 'Ничего не найдено'}
            </h3>
            {searchError ? (
              <p className="text-amber-300 text-sm mb-2">{searchError}</p>
            ) : (
              <>
                <p className="text-gray-400 text-sm mb-2">Попробуйте переформулировать запрос или найти по референсному аудио.</p>
                <p className="text-gray-500 text-xs">CLAP понимает английский; русские запросы идут через словарь синонимов.</p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
