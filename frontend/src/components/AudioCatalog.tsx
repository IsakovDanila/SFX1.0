import { useState } from 'react';
import { Play, Pause, ChevronDown, ChevronUp, Upload } from 'lucide-react';
import { useStore, AudioFile } from '../store/useStore';
import AudioPlayer from './AudioPlayer';
import ImportModal from './ImportModal';

export default function AudioCatalog() {
  const { files, selectedFile, setSelectedFile, showImportModal, setShowImportModal } = useStore();
  const [cat, setCat] = useState('Все');
  const [sortBy, setSortBy] = useState<'name' | 'duration' | 'date'>('name');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  // Категории строим по реальным файлам (моки — статический список)
  const cats = ['Все', ...Array.from(new Set(files.map(f => f.category || 'Без категории')))];
  const filtered = files.filter(f => cat === 'Все' || (f.category || 'Без категории') === cat).sort((a, b) => {
    let c = 0; if (sortBy === 'name') c = a.filename.localeCompare(b.filename); else if (sortBy === 'duration') c = a.duration - b.duration; else c = a.createdAt.localeCompare(b.createdAt);
    return sortDir === 'asc' ? c : -c;
  });
  const toggleSort = (f: typeof sortBy) => { if (sortBy === f) setSortDir(d => d === 'asc' ? 'desc' : 'asc'); else { setSortBy(f); setSortDir('asc'); } };
  const fmtDur = (d: number) => { const m = Math.floor(d / 60), s = Math.floor(d % 60), ms = Math.floor((d % 1) * 100); return m > 0 ? `${m}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}` : `${s}.${ms.toString().padStart(2, '0')}s`; };

  return (
    <div className="flex gap-4 h-full">
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          {cats.map(c => (<button key={c} onClick={() => setCat(c)} className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${cat === c ? 'bg-violet-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'}`}>{c}</button>))}
          <button onClick={() => setShowImportModal(true)} className="ml-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white flex items-center gap-1.5 transition-all shadow-lg shadow-cyan-600/20">
            <Upload size={14} />Импорт файлов
          </button>
          <div className="ml-auto flex items-center gap-2 text-xs text-gray-400">
            <button onClick={() => toggleSort('name')} className={`hover:text-white ${sortBy === 'name' ? 'text-violet-400' : ''}`}>Имя {sortBy === 'name' && (sortDir === 'asc' ? <ChevronUp size={12} className="inline" /> : <ChevronDown size={12} className="inline" />)}</button>
            <button onClick={() => toggleSort('duration')} className={`hover:text-white ${sortBy === 'duration' ? 'text-violet-400' : ''}`}>Длительность {sortBy === 'duration' && (sortDir === 'asc' ? <ChevronUp size={12} className="inline" /> : <ChevronDown size={12} className="inline" />)}</button>
          </div>
        </div>
        <div className="flex-1 overflow-auto rounded-xl border border-gray-700">
          <table className="w-full text-sm">
            <thead className="bg-gray-800/80 sticky top-0"><tr className="text-gray-400 text-xs"><th className="text-left px-4 py-3 font-medium">Файл</th><th className="text-left px-4 py-3 font-medium">Категория</th><th className="text-left px-4 py-3 font-medium">Длит.</th><th className="text-left px-4 py-3 font-medium">Теги</th><th className="text-left px-4 py-3 font-medium">Рейтинг</th></tr></thead>
            <tbody>{filtered.map(f => (<tr key={f.id} onClick={() => setSelectedFile(f)} className={`border-t border-gray-800 cursor-pointer transition-colors ${selectedFile?.id === f.id ? 'bg-violet-900/30' : 'hover:bg-gray-800/50'}`}>
              <td className="px-4 py-3"><div className="flex items-center gap-2"><div className={`w-8 h-8 rounded-lg flex items-center justify-center ${selectedFile?.id === f.id ? 'bg-violet-600' : 'bg-gray-700'}`}>{selectedFile?.id === f.id ? <Pause size={12} className="text-white" /> : <Play size={12} className="text-gray-300" />}</div><div><p className="text-white text-xs font-medium truncate max-w-[200px]">{f.filename}</p><p className="text-gray-500 text-[10px]">{f.sampleRate}Hz • {f.channels}ch</p></div></div></td>
              <td className="px-4 py-3"><span className="px-2 py-0.5 bg-gray-700 rounded text-gray-300 text-[10px]">{f.category}</span></td>
              <td className="px-4 py-3 text-gray-400 text-xs font-mono">{fmtDur(f.duration)}</td>
              <td className="px-4 py-3"><div className="flex gap-1 flex-wrap max-w-[150px]">{f.tags.slice(0, 3).map(t => (<span key={t} className="px-1.5 py-0.5 bg-violet-900/50 text-violet-300 rounded text-[10px]">{t}</span>))}{f.tags.length > 3 && <span className="text-gray-500 text-[10px]">+{f.tags.length - 3}</span>}</div></td>
              <td className="px-4 py-3"><div className="flex gap-0.5">{[1,2,3,4,5].map(s => (<span key={s} className={`text-xs ${s <= (f.rating || 0) ? 'text-yellow-400' : 'text-gray-600'}`}>★</span>))}</div></td>
            </tr>))}</tbody>
          </table>
        </div>
        <div className="mt-3 text-xs text-gray-500">Показано {filtered.length} из {files.length} файлов</div>
      </div>

      {/* Модальное окно импорта */}
      <ImportModal isOpen={showImportModal} onClose={() => setShowImportModal(false)} />
      <div className="w-96 flex-shrink-0">
        <AudioPlayer file={selectedFile} />
        {selectedFile && (
          <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-xl">
            <h4 className="text-white text-sm font-medium mb-2">Описание</h4>
            <p className="text-gray-300 text-xs leading-relaxed">{selectedFile.description}</p>
            <h4 className="text-white text-sm font-medium mt-4 mb-2">Спектральные характеристики</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-900 rounded-lg p-2"><span className="text-gray-500">Тон:</span><span className="text-gray-300 ml-1">{selectedFile.spectralFeatures.pitch}</span></div>
              <div className="bg-gray-900 rounded-lg p-2"><span className="text-gray-500">Энергия:</span><span className="text-gray-300 ml-1">{selectedFile.spectralFeatures.energy}</span></div>
              <div className="bg-gray-900 rounded-lg p-2"><span className="text-gray-500">Шум:</span><span className="text-gray-300 ml-1">{selectedFile.spectralFeatures.hasNoise ? 'Да' : 'Нет'}</span></div>
              <div className="bg-gray-900 rounded-lg p-2"><span className="text-gray-500">Реверб:</span><span className="text-gray-300 ml-1">{selectedFile.spectralFeatures.hasReverb ? 'Да' : 'Нет'}</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
