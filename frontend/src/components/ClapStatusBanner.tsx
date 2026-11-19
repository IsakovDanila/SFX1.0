/**
 * Плашка состояния CLAP-режима.
 *
 * Зачем: загрузка модели занимает минуты и качает ~1.9 ГБ. Без плашки
 * пользователь видит «демо-режим» и не понимает, идёт загрузка, всё сломалось
 * или CLAP просто не установлен. Здесь три честных состояния:
 *   loading — «загружаю модель, N с» + шаг;
 *   failed  — что случилось и что нажать (повторить / как поставить);
 *   ready   — короткая зелёная отметка (гаснет сама).
 */
import { useEffect, useState } from 'react';
import { Cpu, Loader2, AlertTriangle, CheckCircle2, RefreshCw } from 'lucide-react';
import { api, ClapStatus } from '../api/client';

export default function ClapStatusBanner() {
  const [st, setSt] = useState<ClapStatus | null>(null);
  const [hidden, setHidden] = useState(false);
  const [retrying, setRetrying] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.clapStatus();
        if (!cancelled) setSt(s);
      } catch {
        if (!cancelled) setSt(null);
      }
    };
    poll();
    // опрашиваем часто, пока грузится; редко — в остальных состояниях
    const interval = setInterval(poll, st?.state === 'loading' ? 2000 : 15000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [st?.state]);

  // «ready» показываем 8 секунд и убираем — не мешает работать
  useEffect(() => {
    if (st?.state !== 'ready') return;
    const t = setTimeout(() => setHidden(true), 8000);
    return () => clearTimeout(t);
  }, [st?.state]);

  if (!st || st.state === 'disabled' || hidden) return null;

  const retry = async () => {
    setRetrying(true);
    try {
      const s = await api.clapRetry();
      setSt(s as ClapStatus);
      setHidden(false);
    } finally {
      setRetrying(false);
    }
  };

  if (st.state === 'loading') {
    return (
      <div className="bg-cyan-950/60 border-b border-cyan-800 px-6 py-2 flex items-center gap-3 text-xs text-cyan-200">
        <Loader2 size={14} className="animate-spin flex-shrink-0" />
        <span>
          <b>Загружаю CLAP — {st.elapsed_s.toFixed(0)} с.</b>{' '}
          {st.step || 'подготовка'}
          <span className="text-cyan-400/80">
            {' '}— поиск уже работает по словам и русским тегам; семантика включится
            автоматически, когда модель будет готова.
          </span>
        </span>
      </div>
    );
  }

  if (st.state === 'failed') {
    return (
      <div className="bg-amber-950/60 border-b border-amber-800 px-6 py-2.5 text-xs text-amber-100">
        <div className="flex items-start gap-3">
          <AlertTriangle size={14} className="mt-0.5 flex-shrink-0 text-amber-400" />
          <div className="flex-1">
            <p>
              <b>CLAP не загрузился</b>
              {st.error ? <>: {st.error}</> : null}. Работает поиск по словам и
              русским тегам — он не требует модели.
            </p>
            {st.hint && (
              <p className="text-amber-300/80 mt-1 font-mono text-[11px] leading-relaxed">
                {st.hint}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={retry}
              disabled={retrying}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-800/60 hover:bg-amber-700/70 disabled:opacity-50 rounded-lg text-amber-50 font-medium"
            >
              <RefreshCw size={12} className={retrying ? 'animate-spin' : ''} />
              Повторить
            </button>
            <button onClick={() => setHidden(true)} className="text-amber-400/70 hover:text-amber-200 px-1">
              скрыть
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ready
  return (
    <div className="bg-emerald-950/60 border-b border-emerald-800 px-6 py-2 flex items-center gap-3 text-xs text-emerald-200">
      <CheckCircle2 size={14} className="flex-shrink-0" />
      <span>
        <b>CLAP загружен</b> за {st.elapsed_s.toFixed(0)} с
        {st.device ? ` · ${st.device}` : ''}. Семантический поиск включён,
        русские теги работают как fallback.
      </span>
      <button onClick={() => setHidden(true)} className="ml-auto text-emerald-400/70 hover:text-emerald-200">
        скрыть
      </button>
    </div>
  );
}
