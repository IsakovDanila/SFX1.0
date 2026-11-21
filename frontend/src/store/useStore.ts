import { create } from 'zustand';

export interface AudioFile {
  id: string;
  filename: string;
  duration: number;
  sampleRate: number;
  channels: number;
  tags: string[];
  autoTags: { ru: string; en: string; group: string; score: number }[];
  category: string;
  /** исходный (английский) ключ категории, если категория локализована */
  categoryKey?: string;
  description: string;
  spectralFeatures: {
    pitch: string;
    energy: 'low' | 'medium' | 'high';
    hasNoise: boolean;
    hasReverb: boolean;
    tempo?: number;
  };
  embedding?: number[];
  rating?: number;
  createdAt: string;
  waveformColor?: string;
}

export interface SearchResult {
  file: AudioFile;
  score: number;
}

export type ViewMode = 'catalog' | 'search';

interface AppState {
  files: AudioFile[];
  selectedFile: AudioFile | null;
  searchQuery: string;
  searchResults: SearchResult[];
  isSearching: boolean;
  currentView: ViewMode;
  isPlaying: boolean;
  currentTime: number;
  audioUrls: Record<string, string>; // fileId -> objectUrl для реальных файлов
  showImportModal: boolean;
  setFiles: (files: AudioFile[]) => void;
  setSelectedFile: (file: AudioFile | null) => void;
  setSearchQuery: (query: string) => void;
  setSearchResults: (results: SearchResult[]) => void;
  setIsSearching: (isSearching: boolean) => void;
  setCurrentView: (view: ViewMode) => void;
  setIsPlaying: (playing: boolean) => void;
  setCurrentTime: (time: number) => void;
  addTag: (fileId: string, tag: string) => void;
  removeTag: (fileId: string, tag: string) => void;
  rateFile: (fileId: string, rating: number) => void;
  updateFileDescription: (fileId: string, description: string) => void;
  setShowImportModal: (show: boolean) => void;
  setAudioUrl: (fileId: string, url: string) => void;
}

export const useStore = create<AppState>((set) => ({
  files: [],
  selectedFile: null,
  searchQuery: '',
  searchResults: [],
  isSearching: false,
  currentView: 'catalog',
  isPlaying: false,
  currentTime: 0,
  audioUrls: {},
  showImportModal: false,
  setShowImportModal: (show) => set({ showImportModal: show }),
  setAudioUrl: (fileId, url) => set((state) => ({ audioUrls: { ...state.audioUrls, [fileId]: url } })),
  setFiles: (files) => set({ files }),
  setSelectedFile: (file) => set({ selectedFile: file }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  setSearchResults: (results) => set({ searchResults: results }),
  setIsSearching: (isSearching) => set({ isSearching }),
  setCurrentView: (view) => set({ currentView: view }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setCurrentTime: (time) => set({ currentTime: time }),
  addTag: (fileId, tag) => set((state) => ({
    files: state.files.map(f => f.id === fileId ? { ...f, tags: [...f.tags.filter(t => t !== tag), tag] } : f),
    selectedFile: state.selectedFile?.id === fileId ? { ...state.selectedFile, tags: [...state.selectedFile.tags.filter(t => t !== tag), tag] } : state.selectedFile
  })),
  removeTag: (fileId, tag) => set((state) => ({
    files: state.files.map(f => f.id === fileId ? { ...f, tags: f.tags.filter(t => t !== tag) } : f),
    selectedFile: state.selectedFile?.id === fileId ? { ...state.selectedFile, tags: state.selectedFile.tags.filter(t => t !== tag) } : state.selectedFile
  })),
  rateFile: (fileId, rating) => set((state) => ({
    files: state.files.map(f => f.id === fileId ? { ...f, rating } : f),
    selectedFile: state.selectedFile?.id === fileId ? { ...state.selectedFile, rating } : state.selectedFile
  })),
  updateFileDescription: (fileId, description) => set((state) => ({
    files: state.files.map(f => f.id === fileId ? { ...f, description } : f),
    selectedFile: state.selectedFile?.id === fileId ? { ...state.selectedFile, description } : state.selectedFile
  })),
}));
