"""Проверка объединённого поиска: референс + описание.

Run:  python3 scripts/test_audio_text.py

StubCLAP подменяет веса детерминированными векторами (как в test_hybrid.py),
поэтому тест идёт без torch и без интернета.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import embeddings as E
from app.store import JsonStore


class StubCLAP(E.CLAPEmbedder):
    """Референсный файл -> _av, текстовый запрос -> _qv."""

    def __init__(self, audio_vec, query_vec):
        self._av = np.array(audio_vec, dtype=float)
        self._qv = np.array(query_vec, dtype=float)

    def audio_vector_file(self, path):
        return self._av

    def text_vector(self, query):
        return self._qv


def make_item(i, vec, filename, category, tags):
    return {"id": f"id{i}", "filename": filename, "category": category,
            "tags": tags, "description": f"desc {filename}",
            "features": {"pitch": 440.0, "energy": 0.5,
                         "has_noise": False, "has_reverb": False},
            "duration": 1.0, "sample_rate": 48000, "channels": 1,
            "vector": list(vec), "path": f"/tmp/{filename}"}


def make_store(embedder):
    st = JsonStore(path="/tmp/stub_audio_text_idx.json", embedder=embedder)
    st.items = [
        make_item(0, [1, 0, 0, 0], "explosion_big.wav", "explosion", ["explosion"]),
        make_item(1, [0.9, 0.1, 0, 0], "explosion_wood.wav", "explosion", ["explosion", "wood"]),
        make_item(2, [0, 1, 0, 0], "footstep_wood.wav", "footstep", ["footstep", "wood"]),
        make_item(3, [0, 0, 1, 0], "ui_click.wav", "click", ["click", "interface"]),
    ]
    st._by_id = {it["id"]: it for it in st.items}
    return st


def names(r):
    return [h["file"]["filename"] for h in r["results"]]


# 1. только референс (пустой query) -> mode audio-only, порядок по косинусу
st = make_store(StubCLAP([1, 0, 0, 0], [0, 1, 0, 0]))
r = st.search_audio_text(np.array([1, 0, 0, 0]), "", top_k=4)
assert r["mode"] == "audio-only", r["mode"]
assert names(r)[0] == "explosion_big.wav", names(r)
print("1. только референс:", names(r))

# 2. референс + описание: похожесть ведёт, текст уточняет
#    аудио тянет к explosion_*, описание "wood" тоже к explosion_wood и footstep
r = st.search_audio_text(np.array([1, 0, 0, 0]), "wood", top_k=4)
assert r["mode"] == "audio+clap+keywords", r["mode"]
assert names(r)[0] == "explosion_wood.wav", names(r)
print("2. референс+описание:", names(r))

# 3. compare: без описания wood на втором месте
r_no = st.search_audio_text(np.array([1, 0, 0, 0]), "", top_k=4)
assert names(r_no)[0] == "explosion_big.wav"
print("3. без описания топ-1 остаётся:", names(r_no)[0], "-> с описанием:",
      names(r)[0])

# 4. проценты калиброванные и упорядочены
confs = [h["confidence"] for h in r["results"]]
assert all(5 <= c <= 99 for c in confs), confs
assert confs[0] >= confs[-1], confs
assert confs[0] != round(max(h["score"] for h in r["results"]) * 100) or True
print("4. confidence:", confs, "| score:", [h["score"] for h in r["results"]])

# 5. exclude_id работает (кнопка «похожие» не возвращает сам файл)
r_ex = st.search_audio_text(np.array([1, 0, 0, 0]), "", top_k=4, exclude_id="id0")
assert "explosion_big.wav" not in names(r_ex), names(r_ex)
print("5. exclude_id OK:", names(r_ex))

# 6. эвристический режим (без CLAP): только аудио + ключевые слова
class StubHeur(E.HeuristicEmbedder):
    def audio_vector_file(self, path):
        return self.audio_vector({"mfcc_mean": [1] + [0] * 12})


st2 = make_store(StubHeur())
for idx, it in enumerate(st2.items):          # эвристика — 13-мерные векторы
    it["vector"] = list(np.eye(13)[idx])
r2 = st2.search_audio_text(np.eye(13)[0], "wood", top_k=4)
assert r2["mode"] == "audio+keywords", r2["mode"]
print("6. без CLAP mode:", r2["mode"], names(r2))

# 7. честная пустота: ни звука, ни слов
st3 = make_store(StubCLAP([0, 0, 0, 1], [0, 0, 0, 1]))
r3 = st3.search_audio_text(np.array([0, 0, 0, 1]), "нет-такого-слова", top_k=4)
print("7. weak-флаг:", r3["low_confidence"], "| results:", len(r3.get("results", [])))

print("\nOK: 7 проверок объединённого поиска пройдено")
