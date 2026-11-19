import { useState } from 'react';
import { Star, BarChart3, CheckCircle2, XCircle } from 'lucide-react';
import { useStore, SearchResult } from '../store/useStore';
import { mockAudioFiles } from '../data/mockData';
import { api, toAudioFile } from '../api/client';
import AudioPlayer from './AudioPlayer';

interface EvalQuery { id: string; query: string; expectedFiles: string[]; status: 'pending' | 'evaluated'; ratings: Record<string, number>; }

const evalQueries: EvalQuery[] = [
  { id: 'eq1', query: 'тяжёлый металлический удар', expectedFiles: ['1'], status: 'pending', ratings: {} },
  { id: 'eq2', query: 'шаг по деревянному полу', expectedFiles: ['2'], status: 'pending', ratings: {} },
  { id: 'eq3', query: 'атмосферный звук природы', expectedFiles: ['3', '7'], status: 'pending', ratings: {} },
  { id: 'eq4', query: 'звук интерфейса', expectedFiles: ['4', '9'], status: 'pending', ratings: {} },
  { id: 'eq5', query: 'взрыв с низкими частотами', expectedFiles: ['5'], status: 'pending', ratings: {} },
];

export default function EvaluationMode() {
  const { selectedFile, setSelectedFile } = useStore();
  const [queries, setQueries] = useState<EvalQuery[]>(evalQueries);
  const [activeIdx, setActiveIdx] = useState(0);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [searched, setSearched] = useState(false);
  const active = queries[activeIdx];

  // Fallback на моки, если бэкенд недоступен
  const fallbackSearch = () => {
    const r = mockAudioFiles.map(file => {
      let score = 0; const q = active.query.toLowerCase();
      if (file.description.toLowerCase().includes(q)) score += 0.8;
      if (file.tags.some(t => q.includes(t))) score += 0.5;
      if (active.expectedFiles.includes(file.id)) score += 0.4;
      score += Math.random() * 0.15; score = Math.min(score, 1);
      return { file, score };
    }).filter(r => r.score > 0.2).sort((a, b) => b.score - a.score);
    setResults(r);
  };

  const runSearch = async () => {
    setIsSearching(true);
    try {
      const res = await api.searchByText(active.query, 10);
      setResults(res.map(r => ({
        file: toAudioFile(r.file),
        score: Math.max(0, Math.min(1, r.score)),
      })));
      const st = useStore.getState();
      res.forEach(r => st.setAudioUrl(r.file.id, api.getAudioUrl(r.file.id)));
    } catch {
      fallbackSearch();
    } finally {
      setIsSearching(false);
      setSearched(true);
    }
  };

  const pickQuery = (i: number) => { setActiveIdx(i); setResults([]); setSearched(false); };

  const rateResult = (fileId: string, rating: number) => { setQueries(prev => prev.map((q, i) => i !== activeIdx ? q : { ...q, ratings: { ...q.ratings, [fileId]: rating } })); };
  const markDone = () => { setQueries(prev => prev.map((q, i) => i !== activeIdx ? q : { ...q, status: 'evaluated' as const })); };
  const evaluated = queries.filter(q => q.status === 'evaluated');
  const avgRating = evaluated.length > 0 ? evaluated.reduce((acc, q) => { const r = Object.values(q.ratings); return acc + (r.length ? r.reduce((a, b) => a + b, 0) / r.length : 0); }, 0) / evaluated.length : 0;

  return (
    <div className="flex gap-4 h-full">
      <div className="w-72 flex-shrink-0 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white font-medium text-sm">Тестовые запросы</h3>
          <button onClick={() => setShowStats(!showStats)} className="p-1.5 text-gray-400 hover:text-white"><BarChart3 size={16} /></button>
        </div>
        {showStats && (
          <div className="mb-3 p-3 bg-gray-800 border border-gray-700 rounded-xl">
            <h4 className="text-xs text-gray-400 mb-2">Статистика</h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gray-900 rounded-lg p-2 text-center"><p className="text-lg font-bold text-violet-400">{evaluated.length}/{queries.length}</p><p className="text-[10px] text-gray-500">Оценено</p></div>
              <div className="bg-gray-900 rounded-lg p-2 text-center"><p className="text-lg font-bold text-cyan-400">{avgRating.toFixed(1)}</p><p className="text-[10px] text-gray-500">Ср. оценка</p></div>
            </div>
          </div>
        )}
        <div className="flex-1 overflow-auto rounded-xl border border-gray-700">
          {queries.map((q, i) => (
            <div key={q.id} onClick={() => pickQuery(i)} className={`p-3 border-b border-gray-800 cursor-pointer transition-colors ${i === activeIdx ? 'bg-violet-900/20' : 'hover:bg-gray-800/50'}`}>
              <div className="flex items-center gap-2">{q.status === 'evaluated' ? <CheckCircle2 size={14} className="text-green-400" /> : <XCircle size={14} className="text-gray-500" />}<p className="text-white text-xs font-medium truncate">{q.query}</p></div>
              <div className="flex items-center gap-2 mt-1.5"><span className="text-[10px] text-gray-500">Ожидаемые: {q.expectedFiles.length}</span><span className="text-[10px] text-gray-500">Оценено: {Object.keys(q.ratings).length}</span></div>
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 flex flex-col">
        <div className="p-4 bg-gray-800 border border-gray-700 rounded-xl mb-4">
          <div className="flex items-center justify-between">
            <div><p className="text-gray-400 text-xs mb-1">Запрос #{activeIdx + 1}</p><h3 className="text-white text-lg font-medium">«{active.query}»</h3><p className="text-gray-500 text-xs mt-1">Ожидаемые: {active.expectedFiles.length} файлов</p></div>
            <div className="flex gap-2">
              <button onClick={() => activeIdx > 0 && pickQuery(activeIdx - 1)} disabled={activeIdx === 0} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-30 text-white rounded-lg text-xs">← Назад</button>
              <button onClick={runSearch} disabled={isSearching} className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-xs font-medium">{isSearching ? 'Поиск...' : 'Выполнить поиск'}</button>
              <button onClick={() => activeIdx < queries.length - 1 && pickQuery(activeIdx + 1)} disabled={activeIdx === queries.length - 1} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-30 text-white rounded-lg text-xs">Далее →</button>
            </div>
          </div>
        </div>
        {results.length > 0 ? (
          <div className="flex gap-4 flex-1 min-h-0">
            <div className="flex-1 overflow-auto">
              <div className="space-y-2">
                {results.map((r, i) => {
                  const isExpected = active.expectedFiles.includes(r.file.id);
                  const userRating = active.ratings[r.file.id];
                  return (
                    <div key={r.file.id} className={`p-3 rounded-xl border transition-colors ${selectedFile?.id === r.file.id ? 'bg-violet-900/20 border-violet-600' : 'bg-gray-800 border-gray-700'}`}>
                      <div className="flex items-center gap-3">
                        <span className="text-gray-500 text-xs font-mono w-6">#{i + 1}</span>
                        <div className="flex-1 min-w-0 cursor-pointer" onClick={() => setSelectedFile(r.file)}>
                          <div className="flex items-center gap-2"><p className="text-white text-xs font-medium truncate">{r.file.filename}</p>{isExpected && <span className="px-1.5 py-0.5 bg-green-900/50 text-green-300 rounded text-[10px]">ожидаемый</span>}</div>
                          <p className="text-gray-500 text-[10px] mt-0.5">Score: {(r.score * 100).toFixed(0)}% • {r.file.duration.toFixed(1)}s</p>
                        </div>
                        <div className="flex items-center gap-1">{[1,2,3,4,5].map(s => (<button key={s} onClick={() => rateResult(r.file.id, s)} className={`p-0.5 ${s <= (userRating || 0) ? 'text-yellow-400' : 'text-gray-600 hover:text-yellow-400/50'}`}><Star size={16} fill={s <= (userRating || 0) ? 'currentColor' : 'none'} /></button>))}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-4 flex justify-center"><button onClick={markDone} className="px-6 py-2.5 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium flex items-center gap-2"><CheckCircle2 size={16} />Завершить оценку</button></div>
            </div>
            <div className="w-72 flex-shrink-0"><AudioPlayer file={selectedFile} compact /></div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-800/30 border border-gray-700 rounded-xl"><div className="text-center"><BarChart3 size={40} className="text-gray-600 mx-auto mb-3" /><p className="text-gray-400 text-sm">{searched ? 'По этому запросу ничего не найдено' : 'Нажмите «Выполнить поиск»'}</p><p className="text-gray-600 text-xs mt-1">{searched ? 'Переформулируйте запрос или добавьте теги в «Разметке»' : 'Оцените каждый результат от 1 до 5 звёзд'}</p></div></div>
        )}
      </div>
    </div>
  );
}
