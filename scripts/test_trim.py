"""Регрессия: запас при обрезке тишины не должен ломать кратность окна CLAP.

Дефект. _trim_silence добавляет запас по 50 мс с краёв. Для файла, где
звук занимает всё время, выходило 240001 сэмпл вместо 240000. CLAP
считает n_repeat = int(480000 / len(y)):
    480000 / 240000 = 2.0  -> 2 повтора, нулей 0%
    480000 / 240001 = 1.99 -> 1 повтор, нулей 50% окна
Один лишний сэмпл удваивал долю тишины в окне модели.

Run:  python3 scripts/test_trim.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from app import embeddings as E          # noqa: E402

SR = 48000
WIN = 480000                            # окно CLAP: 10 с @ 48 кГц


def make_embedder():
    e = E.CLAPEmbedder.__new__(E.CLAPEmbedder)
    e.CLAP_SR, e.MAX_SECONDS, e.SILENCE_DB = SR, 10.0, -45.0
    return e


def tone_with_silence(seconds, lead=0.0, tail=0.0):
    """Шум + тишина по краям; звук доходит до самого конца файла."""
    n = int(seconds * SR)
    y = np.zeros(n, dtype=np.float32)
    start = int(lead * SR)
    end = n - int(tail * SR)
    rng = np.random.default_rng(0)
    y[start:end] = rng.standard_normal(end - start).astype(np.float32) * 0.1
    return y


e = make_embedder()

# --- 1. ровно 5 с звука от края до края: нулевой добивки быть не должно ---
y = tone_with_silence(5.0)
trimmed = e._trim_silence(y)
reps = int(WIN / len(trimmed))
pad = max(0, WIN - reps * len(trimmed))
assert len(trimmed) == 240000, (
    f"после обрезки {len(trimmed)} сэмплов вместо 240000 — CLAP получит "
    f"{reps} повтор(ов) и {100 * pad / WIN:.0f}% нулей")
assert reps == 2 and pad == 0, (reps, pad)
print(f"1. 5 с до краёв -> {len(trimmed)} сэмплов, повторов {reps}, нули 0% OK")


# --- 2. тишина по краям срезается, звук не теряется ---------------------
y = tone_with_silence(5.0, lead=1.0, tail=1.0)
trimmed = e._trim_silence(y)
assert 3.0 * SR <= len(trimmed) <= 3.2 * SR, len(trimmed) / SR
print(f"2. тишина по краям срезана: {len(trimmed) / SR:.2f} с OK")


# --- 3. короткий файл: подрезка не должна съедать сам звук --------------
y = tone_with_silence(0.8)
trimmed = e._trim_silence(y)
assert 0.7 * SR <= len(trimmed) <= 0.9 * SR, len(trimmed) / SR
print(f"3. короткий файл 0.8 с сохранён: {len(trimmed) / SR:.2f} с OK")


# --- 4. пустой и очень короткий вход не падает -------------------------
assert e._trim_silence(np.zeros(0, dtype=np.float32)).size == 0
assert e._trim_silence(np.zeros(100, dtype=np.float32)).size == 100
print("4. пустой/короткий вход обрабатывается без ошибок OK")

print("\nВСЕ ТЕСТЫ ОБРЕЗКИ ПРОЙДЕНЫ")
