import { useState, useRef } from 'react';
import { Upload, X, FileAudio, FolderOpen, Loader2, Check, Server } from 'lucide-react';
import { createAudioFileFromImport } from '../utils/audioAnalysis';
import { useStore, AudioFile } from '../store/useStore';
import { categories } from '../data/mockData';
import { api, toAudioFile } from '../api/client';

interface ImportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ImportModal({ isOpen, onClose }: ImportModalProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processed, setProcessed] = useState<{ name: string; success: boolean }[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('Без категории');
  const [serverDir, setServerDir] = useState('');
  const [scanMsg, setScanMsg] = useState('');
  const [scanning, setScanning] = useState(false);
  const [scanProg, setScanProg] = useState<{ done: number; total: number; current: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // NOTE: хуки обязаны вызываться до раннего return (иначе React роняет
  // компонент при закрытии модалки: "Rendered fewer hooks than expected").
  const { files, setFiles, setAudioUrl } = useStore();

  if (!isOpen) return null;

  const handleFiles = async (fileList: FileList | File[]) => {
    const audioFiles = Array.from(fileList).filter(f =>
      f.type.startsWith('audio/') ||
      /\.(wav|mp3|ogg|flac|aiff|m4a)$/i.test(f.name)
    );

    if (audioFiles.length === 0) {
      alert('Выберите аудиофайлы (.wav, .mp3, .ogg, .flac)');
      return;
    }

    setIsProcessing(true);
    setProcessed([]);
    const newFiles: AudioFile[] = [];

    for (const file of audioFiles) {
      try {
        const { audioFile, objectUrl } = await createAudioFileFromImport(file, selectedCategory);
        newFiles.push(audioFile);
        setAudioUrl(audioFile.id, objectUrl);
        setProcessed(prev => [...prev, { name: file.name, success: true }]);
      } catch (err) {
        console.error(`Error processing ${file.name}:`, err);
        setProcessed(prev => [...prev, { name: file.name, success: false }]);
      }
    }

    // Добавляем новые файлы в store
    setFiles([...files, ...newFiles]);
    setIsProcessing(false);
  };

  // Индексация папки на стороне бэкенда: фоновый скан + прогресс-бар.
  // Повторный скан той же папки почти мгновенный (бэкенд переиспользует
  // файлы без изменений) и не затирает ручные теги/описания.
  const handleServerScan = async () => {
    if (!serverDir.trim() || scanning) return;
    setScanning(true);
    setScanMsg('');
    setScanProg({ done: 0, total: 0, current: '' });
    try {
      await api.scanBackground(serverDir.trim());
      for (;;) {
        await new Promise(r => setTimeout(r, 800));
        const st = await api.scanStatus();
        setScanProg({ done: st.done, total: st.total, current: st.current });
        if (st.status === 'done') break;
        if (st.status === 'error') throw new Error(st.result?.error || 'scan failed');
        if (st.status === 'idle') break;
      }
      const remote = await api.getAllFiles();
      const mapped = remote.map(toAudioFile);
      mapped.forEach(f => setAudioUrl(f.id, api.getAudioUrl(f.id)));
      setFiles(mapped);
      const st = await api.scanStatus();
      const blind = Number(st.result?.without_tags ?? 0);
      const welcome = `✓ Проиндексировано файлов: ${remote.length}. Каталог обновлён.`;
      // Файлы, о которых нет ни одного слова (имя вида 1-20133-A-39.wav и нет
      // tags.csv): их найдёт только поиск по звуку или ручная разметка.
      setScanMsg(blind > 0
        ? `${welcome} ⚠ У ${blind} из ${remote.length} файлов нет ни тегов, ни осмысленного имени — `
          + 'они не найдутся по словам. Добавьте tags.csv в папку или разметьте их во вкладке «Разметка».'
        : welcome);
    } catch (e) {
      setScanMsg(`✕ ${e instanceof Error ? e.message : 'бэкенд недоступен'}`);
    } finally {
      setScanning(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleClose = () => {
    setProcessed([]);
    setScanMsg('');
    setScanProg(null);
    onClose();
  };

  const scanPct = scanProg && scanProg.total > 0
    ? Math.round((100 * scanProg.done) / scanProg.total) : 0;
  const scanCurrent = scanProg?.current
    ? scanProg.current.split(/[/\\]/).pop() : '';

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-gray-700">
          <div>
            <h2 className="text-white text-lg font-bold">Импорт аудиофайлов</h2>
            <p className="text-gray-400 text-xs mt-1">Добавьте свои звуки в каталог</p>
          </div>
          <button onClick={handleClose} className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800">
            <X size={20} />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Выбор категории */}
          <div>
            <label className="text-gray-300 text-xs font-medium mb-2 block">Категория для новых файлов:</label>
            <div className="flex gap-2 flex-wrap">
              {categories.filter(c => c !== 'Все').map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    selectedCategory === cat
                      ? 'bg-violet-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  {cat}
                </button>
              ))}
              <button
                onClick={() => setSelectedCategory('Без категории')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedCategory === 'Без категории'
                    ? 'bg-violet-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                Без категории
              </button>
            </div>
          </div>

          {/* Drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${
              isDragging
                ? 'border-violet-500 bg-violet-900/20'
                : 'border-gray-600 hover:border-violet-500 hover:bg-gray-800/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="audio/*,.wav,.mp3,.ogg,.flac,.aiff,.m4a"
              onChange={(e) => e.target.files && handleFiles(e.target.files)}
              className="hidden"
            />

            {isProcessing ? (
              <div>
                <Loader2 size={48} className="text-violet-400 animate-spin mx-auto mb-3" />
                <p className="text-white text-sm font-medium">Анализ аудио...</p>
                <p className="text-gray-400 text-xs mt-1">Определяем характеристики через Web Audio API</p>
              </div>
            ) : (
              <div>
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-violet-600/20 to-cyan-600/20 border border-violet-600/30 flex items-center justify-center">
                  <Upload size={28} className="text-violet-400" />
                </div>
                <p className="text-white text-sm font-medium mb-1">
                  Перетащите аудиофайлы сюда
                </p>
                <p className="text-gray-400 text-xs mb-3">
                  или нажмите для выбора
                </p>
                <div className="flex items-center justify-center gap-4 text-[10px] text-gray-500">
                  <span>WAV</span>
                  <span>•</span>
                  <span>MP3</span>
                  <span>•</span>
                  <span>OGG</span>
                  <span>•</span>
                  <span>FLAC</span>
                  <span>•</span>
                  <span>AIFF</span>
                </div>
              </div>
            )}
          </div>

          {/* Список обработанных файлов */}
          {processed.length > 0 && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <h4 className="text-white text-sm font-medium mb-3 flex items-center gap-2">
                <Check size={16} className="text-green-400" />
                Обработано: {processed.filter(p => p.success).length} / {processed.length}
              </h4>
              <div className="space-y-1.5 max-h-48 overflow-auto">
                {processed.map((item, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <FileAudio size={14} className={item.success ? 'text-green-400' : 'text-red-400'} />
                    <span className={`truncate ${item.success ? 'text-gray-300' : 'text-red-300'}`}>
                      {item.name}
                    </span>
                    {item.success ? (
                      <Check size={12} className="text-green-400 ml-auto flex-shrink-0" />
                    ) : (
                      <X size={12} className="text-red-400 ml-auto flex-shrink-0" />
                    )}
                  </div>
                ))}
              </div>

              {processed.some(p => p.success) && (
                <button
                  onClick={handleClose}
                  className="mt-4 w-full px-4 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Готово — перейти в каталог
                </button>
              )}
            </div>
          )}

          {/* Индексация папки сервера */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
            <h4 className="text-white text-sm font-medium mb-1 flex items-center gap-2">
              <Server size={16} className="text-cyan-400" />
              Папка на сервере (для поиска)
            </h4>
            <p className="text-gray-500 text-[11px] mb-3">
              Проиндексировать директорию на машине бэкенда — файлы попадут в поиск и сохранятся после перезагрузки.
            </p>
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <FolderOpen size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                  type="text"
                  value={serverDir}
                  onChange={(e) => setServerDir(e.target.value)}
                  placeholder="Путь к папке: D:\звуки"
                  className="w-full pl-9 pr-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white text-xs placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
              </div>
              <button
                onClick={handleServerScan}
                disabled={scanning || isProcessing || !serverDir.trim()}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-white rounded-lg text-xs font-medium whitespace-nowrap"
              >
                {scanning ? 'Сканирую…' : 'Сканировать'}
              </button>
            </div>
            {scanning && scanProg && (
              <div className="mt-3">
                <div className="h-2 bg-gray-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-600 to-violet-500 rounded-full transition-all"
                    style={{ width: `${scanPct}%` }}
                  />
                </div>
                <p className="text-gray-400 text-[11px] mt-1.5 truncate">
                  {scanProg.total > 0
                    ? `${scanProg.done}/${scanProg.total} — ${scanCurrent}`
                    : 'Подготовка…'}
                </p>
              </div>
            )}
            {scanMsg && <p className="text-xs mt-2 text-gray-300">{scanMsg}</p>}
          </div>

          {/* Подсказка */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
            <p className="text-gray-400 text-xs leading-relaxed">
              <span className="text-cyan-400 font-medium">💡 Как это работает:</span> Аудиофайлы анализируются прямо в браузере через Web Audio API.
              Определяются длительность, частота дискретизации, высота тона, энергия и другие спектральные характеристики.
              Для полноценного семантического поиска подключите бэкенд с CLAP + ChromaDB.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
