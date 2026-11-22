"""Регрессия на три решения, принятые по замерам:

1. Референс и текст НЕ смешиваются. Измерено: при неточном описании
   неверный файл поднимался выше правильного (вес текста 0.8 против 1.0
   у звука), плюс проценты шли не по порядку списка.
2. Проценты в выдаче идут по убыванию и согласованы с порядком.
3. index.json удаляется при остановке бэкенда (SFX_INDEX_KEEP=1 отключает).

Run:  python3 scripts/test_audio_only.py
"""
import ast
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from app import embeddings as E           # noqa: E402
from app.store import JsonStore           # noqa: E402

MAIN = os.path.join(HERE, "..", "backend", "app", "main.py")
VIEW = os.path.join(HERE, "..", "frontend", "src", "components", "SearchView.tsx")


class StubCLAP(E.CLAPEmbedder):
    def __init__(self):
        pass

    def text_vector(self, q):
        return np.array([0.0, 1.0, 0.0, 0.0])   # текст тянет к классу B

    def audio_vector_file(self, path):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def audio_audio_prob(self, cos):
        lo, hi = 0.30, 0.65
        return 1 / (1 + math.exp(-(12.0 / (hi - lo)) * (cos - (lo + hi) / 2)))


def mk(i, v, fn, cat, tags):
    return {"id": f"i{i}", "filename": fn, "category": cat, "tags": tags,
            "description": "d", "features": {}, "duration": 5.0,
            "sample_rate": 48000, "channels": 1,
            "vector": [float(x) for x in v], "path": f"/tmp/{fn}",
            "embedder": "laion-clap"}


def n(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


st = JsonStore(path="/tmp/audio_only_idx.json", embedder=StubCLAP())
st.items = [mk(0, n([1, 0, 0, 0]), "A_correct.wav", "glass_breaking", ["стекло"]),
            mk(1, n([.88, .45, 0, 0]), "A_second.wav", "glass_breaking", ["стекло"]),
            mk(2, n([0, 1, 0, 0]), "B_wrong.wav", "dog", ["собака"]),
            mk(3, n([0, 0, 1, 0]), "C.wav", "rain", ["дождь"])]
st._by_id = {x["id"]: x for x in st.items}


# --- 1. порядок и проценты: верный файл первый, числа по убыванию -----
res = st.search_vector(np.array([1.0, 0, 0, 0]), 4)
names = [h["file"]["filename"] for h in res]
confs = [h["confidence"] for h in res]
assert names[0] == "A_correct.wav", names
assert confs == sorted(confs, reverse=True), confs
assert confs[0] >= 90, confs
print("1. референс: верный файл первый, проценты по убыванию OK:", confs)


# --- 2. search_audio_text не выдаёт проценты против порядка ----------
r = st.search_audio_text(np.array([1.0, 0, 0, 0]), "dog", 4)
c = [h["confidence"] for h in r["results"]]
assert c == sorted(c, reverse=True), (
    f"проценты идут против порядка списка: {c}")
print("2. референс+текст: проценты монотонны OK:", c, f"(mode={r['mode']})")


# --- 3. исходники: текст не уходит в поиск по референсу --------------
main_src = open(MAIN, encoding="utf-8").read()
tree = ast.parse(main_src)
body = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_search_by_upload":
        body = ast.get_source_segment(main_src, node)
        break
assert body is not None, "_search_by_upload не найдена"
assert "search_audio_text" not in body, (
    "main.py всё ещё склеивает референс и текст через search_audio_text")
assert "search_vector" in body, body
print("3. бэкенд: /api/search/audio не использует query OK")

view_src = open(VIEW, encoding="utf-8").read()
assert "api.searchByAudio(referenceFile, 10)" in view_src, (
    "фронтенд всё ещё передаёт текст в поиск по референсу")
print("4. фронтенд: текст не передаётся при загруженном референсе OK")


# --- 4. очистка индекса при остановке --------------------------------
assert "_register_index_cleanup" in main_src, "нет очистки индекса при остановке"
assert "SFX_INDEX_KEEP" in main_src, "нельзя отключить очистку через SFX_INDEX_KEEP"
assert "atexit.register" in main_src, "нет atexit — Ctrl+C не удалит индекс"
print("5. index.json удаляется при остановке (SFX_INDEX_KEEP=1 отключает) OK")

print("\nВСЕ ТЕСТЫ РЕЖИМА «ТОЛЬКО РЕФЕРЕНС» ПРОЙДЕНЫ")
