import { useState } from 'react';
import { Tag, Plus, X, Search, AlertCircle, Sparkles } from 'lucide-react';
import { useStore } from '../store/useStore';
import { suggestedTags } from '../data/mockData';
import { api } from '../api/client';
import AudioPlayer from './AudioPlayer';

export default function AnnotationMode() {
  const { files, selectedFile, setSelectedFile, addTag, removeTag, updateFileDescription } = useStore();
  const [newTag, setNewTag] = useState('');
  const [tagFilter, setTagFilter] = useState('');
  const [description, setDescription] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleSelectFile = (file: typeof files[0]) => { setSelectedFile(file); setDescription(file.description); };
  const handleAddTag = (tag: string) => {
    if (selectedFile && tag.trim()) {
      const t = tag.trim().toLowerCase();
      addTag(selectedFile.id, t);
      setNewTag('');
      // Сохраняем на бэкенд (тихо игнорируем, если он недоступен)
      api.updateTags(selectedFile.id, [...new Set([...selectedFile.tags, t])]).catch(() => {});
    }
  };
  const handleRemoveTag = (tag: string) => {
    if (!selectedFile) return;
    removeTag(selectedFile.id, tag);
    api.updateTags(selectedFile.id, selectedFile.tags.filter(t => t !== tag)).catch(() => {});
  };
  const handleSaveDescription = () => {
    if (selectedFile) {
      updateFileDescription(selectedFile.id, description);
      api.updateDescription(selectedFile.id, description).catch(() => {});
    }
  };
  const filteredSuggestions = suggestedTags.filter(t => t.toLowerCase().includes(tagFilter.toLowerCase()) && !selectedFile?.tags.includes(t));
  const filesByTagCount = [...files].sort((a, b) => a.tags.length - b.tags.length);

  return (
    <div className="flex gap-4 h-full">
      <div className="w-80 flex-shrink-0 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white font-medium text-sm">Файлы для разметки</h3>
          <span className="text-xs text-gray-500">{files.filter(f => f.tags.length === 0).length} без тегов</span>
        </div>
        <div className="flex-1 overflow-auto rounded-xl border border-gray-700">
          {filesByTagCount.map(file => (
            <div key={file.id} onClick={() => handleSelectFile(file)} className={`p-3 border-b border-gray-800 cursor-pointer transition-colors ${selectedFile?.id === file.id ? 'bg-violet-900/30' : 'hover:bg-gray-800/50'}`}>
              <div className="flex items-center justify-between">
                <p className="text-white text-xs font-medium truncate max-w-[180px]">{file.filename}</p>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${file.tags.length === 0 ? 'bg-red-900/50 text-red-300' : 'bg-green-900/50 text-green-300'}`}>{file.tags.length} тегов</span>
              </div>
              <div className="flex gap-1 mt-1.5 flex-wrap">{file.tags.slice(0, 4).map(tag => (<span key={tag} className="px-1.5 py-0.5 bg-gray-700 text-gray-300 rounded text-[10px]">{tag}</span>))}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 flex flex-col">
        {selectedFile ? (
          <>
            <AudioPlayer file={selectedFile} compact />
            <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-xl">
              <div className="flex items-center gap-2 mb-3"><Tag size={16} className="text-violet-400" /><h4 className="text-white text-sm font-medium">Теги</h4></div>
              <div className="flex gap-2 flex-wrap mb-3">{selectedFile.tags.map(tag => (<span key={tag} className="flex items-center gap-1 px-2.5 py-1 bg-violet-900/50 border border-violet-700 text-violet-200 rounded-lg text-xs">{tag}<button onClick={() => handleRemoveTag(tag)} className="hover:text-red-300"><X size={12} /></button></span>))}{selectedFile.tags.length === 0 && <span className="text-gray-500 text-xs italic">Нет тегов</span>}</div>
              <div className="relative">
                <div className="flex gap-2">
                  <div className="flex-1 relative"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" /><input type="text" value={newTag} onChange={(e) => { setNewTag(e.target.value); setTagFilter(e.target.value); setShowSuggestions(true); }} onFocus={() => setShowSuggestions(true)} onKeyDown={(e) => e.key === 'Enter' && handleAddTag(newTag)} placeholder="Введите тег..." className="w-full pl-9 pr-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white text-xs placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-violet-500" /></div>
                  <button onClick={() => handleAddTag(newTag)} className="px-3 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-xs"><Plus size={14} /></button>
                </div>
                {showSuggestions && filteredSuggestions.length > 0 && (<div className="absolute top-full left-0 right-0 mt-1 bg-gray-900 border border-gray-600 rounded-lg shadow-xl max-h-40 overflow-auto z-10">{filteredSuggestions.slice(0, 10).map(tag => (<button key={tag} onClick={() => { handleAddTag(tag); setShowSuggestions(false); }} className="w-full text-left px-3 py-2 text-xs text-gray-300 hover:bg-gray-800 hover:text-white">{tag}</button>))}</div>)}
              </div>
            </div>
            {(selectedFile.autoTags?.length ?? 0) > 0 && (
              <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2"><Sparkles size={16} className="text-cyan-400" /><h4 className="text-white text-sm font-medium">Предложено моделью (CLAP)</h4></div>
                  <button onClick={() => selectedFile.autoTags.forEach(a => { if (!selectedFile.tags.includes(a.ru)) handleAddTag(a.ru); })} className="px-2.5 py-1 bg-cyan-900/50 hover:bg-cyan-800/50 border border-cyan-700 text-cyan-200 rounded-lg text-[11px]">Принять все</button>
                </div>
                <div className="flex gap-2 flex-wrap">
                  {selectedFile.autoTags.map(a => (
                    <span key={a.ru} className="flex items-center gap-1.5 px-2.5 py-1 bg-cyan-900/30 border border-cyan-800 text-cyan-200 rounded-lg text-xs">
                      {a.ru}<span className="text-cyan-500 text-[10px]">{(a.score * 100).toFixed(0)}%</span>
                      {selectedFile.tags.includes(a.ru)
                        ? <span className="text-green-400 text-[10px]">✓</span>
                        : <button onClick={() => handleAddTag(a.ru)} className="hover:text-white"><Plus size={12} /></button>}
                    </span>
                  ))}
                </div>
                <p className="text-gray-500 text-[10px] mt-2">Речь размечается по звучанию, а не по смыслу слов — содержание мемных фраз модель не расшифровывает.</p>
              </div>
            )}
            <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-xl flex-1">
              <h4 className="text-white text-sm font-medium mb-3">Описание звука</h4>
              <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Опишите звук на естественном языке..." className="w-full h-32 px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white text-xs placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-violet-500 resize-none" />
              <div className="flex items-center justify-between mt-2"><span className="text-gray-500 text-[10px]">Используется для семантического поиска через CLAP</span><button onClick={handleSaveDescription} className="px-4 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-xs">Сохранить</button></div>
            </div>
            <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-xl">
              <div className="flex items-center gap-2 mb-3"><AlertCircle size={14} className="text-cyan-400" /><h4 className="text-white text-sm font-medium">Авто-анализ (Librosa)</h4></div>
              <div className="grid grid-cols-4 gap-2">
                <div className="bg-gray-900 rounded-lg p-2.5 text-center"><p className="text-gray-500 text-[10px] mb-1">Тон</p><p className="text-cyan-300 text-xs font-medium">{selectedFile.spectralFeatures.pitch}</p></div>
                <div className="bg-gray-900 rounded-lg p-2.5 text-center"><p className="text-gray-500 text-[10px] mb-1">Энергия</p><p className="text-cyan-300 text-xs font-medium">{selectedFile.spectralFeatures.energy}</p></div>
                <div className="bg-gray-900 rounded-lg p-2.5 text-center"><p className="text-gray-500 text-[10px] mb-1">Шум</p><p className="text-cyan-300 text-xs font-medium">{selectedFile.spectralFeatures.hasNoise ? '✓' : '✗'}</p></div>
                <div className="bg-gray-900 rounded-lg p-2.5 text-center"><p className="text-gray-500 text-[10px] mb-1">Реверб</p><p className="text-cyan-300 text-xs font-medium">{selectedFile.spectralFeatures.hasReverb ? '✓' : '✗'}</p></div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-800/30 border border-gray-700 rounded-xl"><div className="text-center"><Tag size={40} className="text-gray-600 mx-auto mb-3" /><p className="text-gray-400 text-sm">Выберите файл для разметки</p><p className="text-gray-600 text-xs mt-1">Добавляйте теги и описания для улучшения поиска</p></div></div>
        )}
      </div>
    </div>
  );
}
