import { useEffect, useRef, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX } from 'lucide-react';
import WaveSurfer from 'wavesurfer.js';
import { AudioFile, useStore } from '../store/useStore';

interface AudioPlayerProps { file: AudioFile | null; compact?: boolean; }

export default function AudioPlayer({ file, compact = false }: AudioPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const audioUrls = useStore(s => s.audioUrls);

  useEffect(() => {
    if (!containerRef.current || !file) return;
    if (wavesurferRef.current) wavesurferRef.current.destroy();
    
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#8b5cf6',
      progressColor: '#c4b5fd',
      cursorColor: '#f59e0b',
      barWidth: 2,
      barRadius: 2,
      barGap: 1,
      height: compact ? 48 : 80,
      normalize: true,
    });
    wavesurferRef.current = ws;

    // Если есть реальный URL файла — используем его, иначе генерируем синтетический
    const audioUrl = audioUrls[file.id];
    if (audioUrl) {
      ws.load(audioUrl);
    } else {
      ws.load(generateFakeWaveformData(file));
    }

    ws.on('ready', () => setDuration(file.duration));
    ws.on('audioprocess', () => setCurrentTime(ws.getCurrentTime()));
    ws.on('seeking', () => setCurrentTime(ws.getCurrentTime()));
    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    ws.on('finish', () => setIsPlaying(false));

    return () => { ws.destroy(); };
  }, [file, audioUrls]);

  const togglePlay = () => wavesurferRef.current?.playPause();
  const toggleMute = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.setMuted(!isMuted);
      setIsMuted(!isMuted);
    }
  };
  const skip = (s: number) => {
    if (wavesurferRef.current && duration > 0) {
      const t = Math.max(0, Math.min(wavesurferRef.current.getCurrentTime() + s, duration));
      wavesurferRef.current.seekTo(t / duration);
    }
  };
  const fmt = (t: number) => `${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, '0')}`;

  if (!file) {
    return (
      <div className={`${compact ? 'p-3' : 'p-6'} bg-gray-800/50 border border-gray-700 rounded-xl flex items-center justify-center`}>
        <p className="text-gray-500 text-sm">Выберите файл для воспроизведения</p>
      </div>
    );
  }

  const hasRealAudio = !!audioUrls[file.id];

  return (
    <div className={`${compact ? 'p-3' : 'p-4'} bg-gray-800 border border-gray-700 rounded-xl`}>
      {!compact && (
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-white text-sm font-medium truncate max-w-[300px]">{file.filename}</h3>
            <p className="text-gray-400 text-xs mt-0.5">
              {file.sampleRate}Hz • {file.channels === 2 ? 'Стерео' : 'Моно'} • {file.category}
              {hasRealAudio && <span className="ml-2 text-green-400">● реальный файл</span>}
            </p>
          </div>
          <div className="text-gray-400 text-xs font-mono">{fmt(currentTime)} / {fmt(duration)}</div>
        </div>
      )}
      <div className="flex items-center gap-3">
        <button onClick={() => skip(-2)} className="p-1.5 text-gray-400 hover:text-white"><SkipBack size={16} /></button>
        <button onClick={togglePlay} className="p-2.5 bg-violet-600 hover:bg-violet-500 rounded-full text-white shadow-lg shadow-violet-600/30">
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button onClick={() => skip(2)} className="p-1.5 text-gray-400 hover:text-white"><SkipForward size={16} /></button>
        <div ref={containerRef} className="flex-1 min-w-0 rounded-lg overflow-hidden" />
        <button onClick={toggleMute} className="p-1.5 text-gray-400 hover:text-white">
          {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
      </div>
    </div>
  );
}

// Генерация синтетического WAV для демо-файлов (без реального аудио)
function generateFakeWaveformData(file: AudioFile): string {
  const sr = file.sampleRate, dur = file.duration, n = Math.floor(sr * dur);
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE'); ws(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, 'data'); v.setUint32(40, n * 2, true);
  const amp = file.spectralFeatures.energy === 'high' ? 0.8 : file.spectralFeatures.energy === 'medium' ? 0.5 : 0.3;
  const freq = file.spectralFeatures.pitch === 'high' ? 800 : file.spectralFeatures.pitch === 'low' ? 150 : 400;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const env = Math.exp(-t * 3) * amp;
    const val = Math.max(-1, Math.min(1, Math.sin(2 * Math.PI * freq * t) * env + (Math.random() - 0.5) * env * 0.3));
    v.setInt16(44 + i * 2, val * 32767, true);
  }
  return URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
}
