"""Регрессия: проценты не должны проваливаться на верных файлах.

Случаи взяты из реальных скриншотов пользователя:

A. «sound bells» на библиотеке ESC-50. Оба файла 1-13572-A-46 и
   1-13571-A-46 — church_bells (из tags.csv), косинусы одинаковые.
   Было: 26% и 26% (ничья обнуляла margin, а «sound» считался промахом).
   Стало: 85% у обоих.
B. «sound of broken glass»: в запросе 4 токена, попадание одно.
   Было: 42% верному файлу.
C. Мусорный запрос: все косинусы вплотную — низкий процент, не бонус.
D. Русский запрос с служебными словами: «мне нужен звук шагов».

Run:  python3 scripts/test_confidence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import embeddings as E
from app.store import JsonStore


class StubCLAP(E.CLAPEmbedder):
    def __init__(self, qv):
        self._qv = np.asarray(qv, dtype=float)

    def text_vector(self, query):
        return self._qv


def item(i, cos, fn, cat, tags):
    """Вектор с заданным косинусом к [1,0,0,0]."""
    vec = [cos, (1 - cos ** 2) ** 0.5, 0, 0]
    return {"id": f"id{i}", "filename": fn, "category": cat, "tags": tags,
            "description": "d", "features": {}, "duration": 1.0,
            "sample_rate": 48000, "channels": 1, "vector": vec,
            "path": f"/tmp/{fn}"}


def store(items):
    st = JsonStore(path="/tmp/conf_idx.json", embedder=StubCLAP([1, 0, 0, 0]))
    st.items = items
    st._by_id = {it["id"]: it for it in st.items}
    return st


def confs(r):
    return [h["confidence"] for h in r["results"]]


# --- A. ничья двух верных файлов одной категории ---------------------
st = store([
    item(0, 0.26, "1-13572-A-46.wav", "church_bells", ["церковные колокола"]),
    item(1, 0.26, "1-13571-A-46.wav", "church_bells", ["церковные колокола"]),
    item(2, 0.15, "1-13613-A-37.wav", "clock_alarm", []),
    item(3, 0.11, "1-137-A-32.wav", "keyboard_typing", []),
])
r = st.search_text("sound bells", top_k=4)
c = confs(r)
assert c[0] >= 80 and c[1] >= 80, c
assert c[0] == c[1], f"ничья должна дать одинаковый процент: {c}"
assert c[2] <= 35 and c[3] <= 25, c
print("A. ничья church_bells OK:", c, "(было 26/26/15/11)")

# --- B. служебные слова не занижают попадание -------------------------
kw_before = E.keyword_score(E.tokenize("sound of a bells"),
                            {"category": "church_bells", "tags": [], "filename": "x.wav"})
assert kw_before >= 0.85, kw_before
print("B. 'sound of a bells' -> kw =", kw_before, "(было 0.75 при попадании 1/4)")

# --- C. мусорный запрос не получает бонус -----------------------------
st = store([item(i, 0.24 - i * 0.001, f"junk{i}.wav", "train", []) for i in range(5)])
r = st.search_text("zzz-no-such-word", top_k=5)
if r["results"]:
    c = confs(r)
    assert max(c) <= 25, c
    print("C. мусорный запрос OK:", c)
else:
    print("C. мусорный запрос -> честная пустота OK")

# --- D. русский запрос со служебными словами --------------------------
kw_d = E.keyword_score(E.tokenize("мне нужен звук шагов"),
                        {"category": "footsteps", "tags": ["шаги"], "filename": "1.wav"})
assert kw_d >= 0.85, kw_d
print("D. 'мне нужен звук шагов' -> kw =", kw_d, "(было 0.0)")

# --- E. проценты монотонны вниз по списку -----------------------------
st = store([
    item(0, 0.40, "a.wav", "church_bells", ["церковные колокола"]),
    item(1, 0.33, "b.wav", "clapping", []),
    item(2, 0.30, "c.wav", "dog", []),
    item(3, 0.28, "d.wav", "rain", []),
])
r = st.search_text("bells", top_k=4)
c = confs(r)
assert c == sorted(c, reverse=True), c
assert all(5 <= x <= 99 for x in c), c
print("E. монотонность и диапазон OK:", c)

print("\nВСЕ ТЕСТЫ ПРОЦЕНТОВ ПРОЙДЕНЫ")
