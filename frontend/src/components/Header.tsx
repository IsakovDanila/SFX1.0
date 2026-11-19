import { Search, Music, Tags, Star, FolderOpen } from 'lucide-react';
import { useStore, ViewMode } from '../store/useStore';

const navItems: { view: ViewMode; label: string; icon: React.ReactNode }[] = [
  { view: 'catalog', label: 'Каталог', icon: <FolderOpen size={18} /> },
  { view: 'search', label: 'Поиск', icon: <Search size={18} /> },
  { view: 'annotate', label: 'Разметка', icon: <Tags size={18} /> },
  { view: 'evaluate', label: 'Оценка', icon: <Star size={18} /> },
];

export default function Header() {
  const { currentView, setCurrentView } = useStore();
  return (
    <header className="bg-gray-900 border-b border-gray-700 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
          <Music size={20} className="text-white" />
        </div>
        <div>
          <h1 className="text-white font-bold text-lg leading-tight">SFX Smart Search</h1>
          <p className="text-gray-400 text-xs">Умный каталог аудиоэффектов</p>
        </div>
      </div>
      <nav className="flex gap-1">
        {navItems.map(item => (
          <button key={item.view} onClick={() => setCurrentView(item.view)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${currentView === item.view ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/30' : 'text-gray-400 hover:text-white hover:bg-gray-800'}`}>
            {item.icon}
            <span className="hidden sm:inline">{item.label}</span>
          </button>
        ))}
      </nav>
    </header>
  );
}
