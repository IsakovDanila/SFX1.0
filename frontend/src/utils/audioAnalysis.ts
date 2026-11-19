/**
 * Утилита для анализа аудиофайлов через Web Audio API
 * Работает прямо в браузере без бэкенда
 */

import { AudioFile } from '../store/useStore';

export interface AudioAnalysis {
  duration: number;
  sampleRate: number;
  channels: number;
  pitch: 'низкий' | 'средний' | 'высокий';
  energy: 'low' | 'medium' | 'high';
  hasNoise: boolean;
  hasReverb: boolean;
  peakData: number[]; // для waveform
}

/**
 * Анализ аудиофайла через Web Audio API
 */
export async function analyzeAudioFile(file: File): Promise<AudioAnalysis> {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  
  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    
    const duration = audioBuffer.duration;
    const sampleRate = audioBuffer.sampleRate;
    const channels = audioBuffer.numberOfChannels;
    
    // Получаем данные первого канала для анализа и waveform
    const channelData = audioBuffer.getChannelData(0);
    
    // === Спектральный анализ (упрощённый) ===
    
    // Энергия (RMS)
    let sumSquares = 0;
    for (let i = 0; i < channelData.length; i++) {
      sumSquares += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sumSquares / channelData.length);
    const energy: 'low' | 'medium' | 'high' = rms < 0.05 ? 'low' : rms < 0.15 ? 'medium' : 'high';
    
    // Высота тона (через zero-crossing rate как approximation)
    let zeroCrossings = 0;
    for (let i = 1; i < channelData.length; i++) {
      if ((channelData[i] >= 0 && channelData[i - 1] < 0) || 
          (channelData[i] < 0 && channelData[i - 1] >= 0)) {
        zeroCrossings++;
      }
    }
    const zcr = zeroCrossings / duration;
    const pitch: 'низкий' | 'средний' | 'высокий' = zcr < 200 ? 'низкий' : zcr < 1500 ? 'средний' : 'высокий';
    
    // Шум (spectral flatness approximation через variance)
    let mean = 0;
    for (let i = 0; i < channelData.length; i++) mean += Math.abs(channelData[i]);
    mean /= channelData.length;
    
    let variance = 0;
    for (let i = 0; i < channelData.length; i++) {
      variance += (Math.abs(channelData[i]) - mean) ** 2;
    }
    variance /= channelData.length;
    const hasNoise = variance < 0.01; // высокий variance = более "структурированный" звук
    
    // Реверб (через анализ затухания)
    const lastQuarter = channelData.slice(Math.floor(channelData.length * 0.75));
    let lastQuarterRms = 0;
    for (let i = 0; i < lastQuarter.length; i++) {
      lastQuarterRms += lastQuarter[i] * lastQuarter[i];
    }
    lastQuarterRms = Math.sqrt(lastQuarterRms / lastQuarter.length);
    const hasReverb = lastQuarterRms > rms * 0.1 && duration > 1.0;
    
    // === Генерация peak data для waveform ===
    const peakCount = 200; // количество пиков для визуализации
    const samplesPerPeak = Math.floor(channelData.length / peakCount);
    const peakData: number[] = [];
    
    for (let i = 0; i < peakCount; i++) {
      let max = 0;
      const start = i * samplesPerPeak;
      const end = Math.min(start + samplesPerPeak, channelData.length);
      for (let j = start; j < end; j++) {
        const abs = Math.abs(channelData[j]);
        if (abs > max) max = abs;
      }
      peakData.push(max);
    }
    
    return {
      duration,
      sampleRate,
      channels,
      pitch,
      energy,
      hasNoise: !hasNoise, // инвертируем: высокий variance = меньше шума
      hasReverb,
      peakData,
    };
  } finally {
    audioCtx.close();
  }
}

/**
 * Создать AudioFile из File + анализ
 */
export async function createAudioFileFromImport(
  file: File,
  category: string = 'Без категории'
): Promise<{ audioFile: AudioFile; objectUrl: string }> {
  const analysis = await analyzeAudioFile(file);
  
  const audioFile: AudioFile = {
    id: crypto.randomUUID(),
    filename: file.name,
    duration: analysis.duration,
    sampleRate: analysis.sampleRate,
    channels: analysis.channels,
    tags: [],
    autoTags: [],
    category,
    description: '',
    spectralFeatures: {
      pitch: analysis.pitch,
      energy: analysis.energy,
      hasNoise: analysis.hasNoise,
      hasReverb: analysis.hasReverb,
    },
    rating: 0,
    createdAt: new Date().toISOString().split('T')[0],
  };
  
  const objectUrl = URL.createObjectURL(file);
  
  return { audioFile, objectUrl };
}
