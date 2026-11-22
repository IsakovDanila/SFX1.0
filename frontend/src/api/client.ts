/**
 * API клиент для связи фронтенда с FastAPI бэкендом
 *
 * По умолчанию используются ОТНОСИТЕЛЬНЫЕ URL ('/api/...'), которые в dev
 * идут через прокси Vite -> FastAPI :8000 (см. vite.config.js).
 * Для Electron/prod-сборки задайте абсолютный адрес через .env:
 *   VITE_API_URL=http://localhost:8000
 */
import type { AudioFile } from '../store/useStore';

const API_URL = import.meta.env.VITE_API_URL ?? '';

/** Файл в формате бэкенда (snake_case) */
export interface ApiFile {
  id: string;
  filename: string;
  duration: number;
  sample_rate: number;
  channels: number;
  category: string;
  /** русское название категории с бэкенда (labels_ru) */
  category_ru?: string;
  tags: string[];
  auto_tags?: { ru: string; en: string; group: string; score: number }[];
  description: string;
  spectral_features: {
    pitch: string;
    energy: 'low' | 'medium' | 'high';
    has_noise: boolean;
    has_reverb: boolean;
  };
}

/** Состояние CLAP-режима с бэкенда (GET /api/clap/status) */
export interface ClapStatus {
  state: 'disabled' | 'loading' | 'ready' | 'failed';
  step: string;
  error: string;
  hint: string;
  device: string;
  ckpt: string;
  model_id: string;
  elapsed_s: number;
  ready: boolean;
  active_embedder?: string;
  indexed?: number;
}

export interface SearchResult {
  id: string;
  score: number;
  /** Калиброванный процент совпадения с бэкенда (5..99). */
  confidence?: number;
  file: ApiFile;
}

/** snake_case (API) -> camelCase (store) */
export function toAudioFile(f: ApiFile): AudioFile {
  return {
    id: f.id,
    filename: f.filename,
    duration: f.duration ?? 0,
    sampleRate: f.sample_rate ?? 44100,
    channels: f.channels ?? 1,
    tags: f.tags ?? [],
    autoTags: f.auto_tags ?? [],
    // В интерфейсе показываем русское название, ключ датасета держим отдельно
    category: f.category_ru || f.category || 'Без категории',
    categoryKey: f.category || '',
    description: f.description ?? '',
    spectralFeatures: {
      pitch: f.spectral_features?.pitch ?? 'средний',
      energy: f.spectral_features?.energy ?? 'medium',
      hasNoise: !!f.spectral_features?.has_noise,
      hasReverb: !!f.spectral_features?.has_reverb,
    },
    createdAt: new Date().toISOString().split('T')[0],
  };
}

export const api = {
  /**
   * Семантический поиск по текстовому описанию
   */
  async searchByText(query: string, nResults = 10): Promise<SearchResult[]> {
    const response = await fetch(`${API_URL}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, n_results: nResults }),
    });

    if (!response.ok) throw new Error(`Search failed: ${response.statusText}`);

    const data = await response.json();
    return data.results;
  },

  /**
   * Поиск по референсному аудиофайлу.
   * Если передан query — описание работает как уточнение к похожести
   * («похоже на этот взрыв, но деревянный»).
   */
  async searchByAudio(file: File, nResults = 10, query = ''): Promise<SearchResult[]> {
    const formData = new FormData();
    formData.append('file', file);

    const qs = new URLSearchParams({ n_results: String(nResults) });
    if (query.trim()) qs.set('query', query.trim());

    const response = await fetch(`${API_URL}/api/search/audio?${qs.toString()}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`Audio search failed: ${response.statusText}`);

    const data = await response.json();
    return data.results;
  },

  /**
   * Сканирование директории с аудиофайлами (синхронное, возвращает {added})
   */
  async scanDirectory(directory: string): Promise<{ added: number }> {
    const response = await fetch(`${API_URL}/api/scan?directory=${encodeURIComponent(directory)}`, {
      method: 'POST',
    });

    if (!response.ok) throw new Error(`Scan failed: ${response.statusText}`);

    return response.json();
  },

  /**
   * Фоновое сканирование (не блокирует UI; прогресс через scanStatus)
   */
  async scanBackground(folder: string): Promise<{ accepted: boolean; folder: string }> {
    const response = await fetch(`${API_URL}/api/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder }),
    });

    if (!response.ok) throw new Error(`Scan failed: ${response.statusText}`);

    return response.json();
  },

  /**
   * Прогресс фонового сканирования
   */
  async scanStatus(): Promise<{
    status: string; total: number; done: number; current: string; result?: any;
  }> {
    const response = await fetch(`${API_URL}/api/scan/status`);

    if (!response.ok) throw new Error(`Scan status failed: ${response.statusText}`);

    return response.json();
  },

  /**
   * Получить все файлы
   */
  async getAllFiles(): Promise<ApiFile[]> {
    const response = await fetch(`${API_URL}/api/files`);

    if (!response.ok) throw new Error(`Get files failed: ${response.statusText}`);

    const data = await response.json();
    return data.files;
  },

  /**
   * Обновить теги файла
   */
  async updateTags(fileId: string, tags: string[]): Promise<void> {
    const response = await fetch(`${API_URL}/api/files/${fileId}/tags`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags }),
    });

    if (!response.ok) throw new Error(`Update tags failed: ${response.statusText}`);
  },

  /**
   * Обновить описание файла
   */
  async updateDescription(fileId: string, description: string): Promise<void> {
    const response = await fetch(`${API_URL}/api/files/${fileId}/description`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description }),
    });

    if (!response.ok) throw new Error(`Update description failed: ${response.statusText}`);
  },

  /**
   * Получить URL аудиофайла для плеера
   */
  getAudioUrl(fileId: string): string {
    return `${API_URL}/api/files/${fileId}/audio`;
  },

  /**
   * Состояние бэкенда (режим эмбеддера, число файлов)
   */
  async health(): Promise<{ status: string; embedder: string; indexed: number }> {
    const response = await fetch(`${API_URL}/api/health`);
    if (!response.ok) throw new Error('health failed');
    return response.json();
  },

  /**
   * Состояние CLAP-режима (плашка в UI)
   */
  async clapStatus(): Promise<ClapStatus> {
    const response = await fetch(`${API_URL}/api/clap/status`);
    if (!response.ok) throw new Error('clap status failed');
    return response.json();
  },

  /**
   * Повторить загрузку CLAP (например, после скачивания весов)
   */
  async clapRetry(): Promise<ClapStatus & { status: string }> {
    const response = await fetch(`${API_URL}/api/clap/retry`, { method: 'POST' });
    if (!response.ok) throw new Error('clap retry failed');
    return response.json();
  },

  /**
   * Проверка доступности бэкенда
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_URL}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  },
};
