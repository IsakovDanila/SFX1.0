"""Validate store.search_text hybrid RRF + honest-empty without torch.

Run:  python3 scripts/test_hybrid.py

StubCLAP subclasses the real CLAPEmbedder (so isinstance checks pass)
but returns deterministic vectors instead of loading laion-clap weights
(real weights need ~4GB free RAM).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import embeddings as E
from app.store import JsonStore


class StubCLAP(E.CLAPEmbedder):
    """Fake CLAP: item0=[1,0,...], item1=[0,1,...], item2=[0,0,1...]."""
    def __init__(self, query_vec):
        self._qv = np.array(query_vec, dtype=float)

    def audio_vector_file(self, path):
        return np.zeros(4)

    def text_vector(self, query):
        return self._qv


def make_item(i, vec, filename, tags):
    return {"id": f"id{i}", "filename": filename, "category": "sfx",
            "tags": tags, "description": f"desc {filename}",
            "features": {"pitch": 440.0, "energy": 0.5,
                         "has_noise": False, "has_reverb": False},
            "duration": 1.0, "sample_rate": 48000, "channels": 1,
            "vector": list(vec), "path": f"/tmp/{filename}"}


def make_store(embedder):
    st = JsonStore(path="/tmp/stub_idx.json", embedder=embedder)
    st.items = [
        make_item(0, [1, 0, 0, 0], "explosion_big.wav", ["explosion", "boom"]),
        make_item(1, [0, 1, 0, 0], "footstep_wood.wav", ["footstep", "wood"]),
        make_item(2, [0, 0, 1, 0], "ui_click.wav", ["click", "interface"]),
    ]
    st._by_id = {it["id"]: it for it in st.items}
    return st


# 1. hybrid: semantic hit (CLAP says item0) + no keyword evidence
st = make_store(StubCLAP([1, 0, 0, 0]))
r = st.search_text("zzz-no-such-word", top_k=3)
assert r["mode"] == "clap+keywords", r["mode"]
assert r["results"][0]["file"]["filename"] == "explosion_big.wav", r["results"][0]
assert r["results"][0]["score"] == 1.0, r["results"][0]["score"]
assert r["low_confidence"] is False, r
print("1. hybrid semantic rank OK:", [(x["file"]["filename"], x["score"]) for x in r["results"]])

# 2. hybrid: RU keyword hit rescues EN-only CLAP (CLAP weakly wrong, tags match)
st = make_store(StubCLAP([0, 0.3, 1, 0]))  # CLAP order: click > footstep > boom
r = st.search_text("шаг дерево", top_k=3)  # RU synonyms -> footstep/wood
top = r["results"][0]["file"]["filename"]
assert top == "footstep_wood.wav", [(x["file"]["filename"], x["score"]) for x in r["results"]]
print("2. hybrid RU-keyword rescue OK:", [(x["file"]["filename"], x["score"]) for x in r["results"]])

# 3. hybrid weak: nothing anywhere -> честная пустота вместо 3 подставных строк
st = make_store(StubCLAP([0, 0, 0, 1]))  # orthogonal: ~zero cosine to all
r = st.search_text("zzz-no-such-word", top_k=3)
assert r["low_confidence"] is True, r
assert r["results"] == [], r
print("3. hybrid honest-empty OK: low_confidence=True, results=[] "
      "(раньше тест ждал 3 строки — рассинхрон с SFX_CLAP_HONEST_EMPTY)")

# 4. demo honest empty: user's screenshot case ("music" -> 10x0% BEFORE fix)
st = make_store(E.HeuristicEmbedder())
r = st.search_text("music", top_k=10)
assert r["mode"] == "heuristic-demo", r["mode"]
assert r["results"] == [] and r["low_confidence"] is True, r
print("4. demo honest-empty OK: 'music' -> results:[] (was: 10 x 0%)")

# 5. demo match still works
r = st.search_text("explosion", top_k=10)
assert len(r["results"]) > 0 and r["low_confidence"] is False, r
assert r["results"][0]["file"]["filename"] == "explosion_big.wav"
print("5. demo keyword hit OK:", [(x["file"]["filename"], x["score"]) for x in r["results"]])

# 6. autotag: audio close to a label prompt -> that label wins
from app import autotag as AT


class FakeTagEmbedder:
    dim = len(AT.LABELS)

    def text_vector(self, prompt):
        v = np.zeros(self.dim)
        for i, (en, _, _) in enumerate(AT.LABELS):
            if en in prompt:
                v[i] = 1.0
                break
        return v


tagger = AT.ZeroShotTagger(FakeTagEmbedder())
idx = next(i for i, (en, _, _) in enumerate(AT.LABELS) if "clapping" in en)
qv = np.zeros(len(AT.LABELS))
qv[idx] = 1.0
tags = tagger.tag(qv)
assert tags[0]["ru"] == "хлопки" and tags[0]["score"] == 1.0, tags
assert all(t["score"] >= tagger.threshold for t in tags), tags
print("6. autotag ranking OK:", [(t["ru"], t["score"]) for t in tags])

# 7. autotag: silence/orthogonal audio -> no tags (threshold filters all)
tags = tagger.tag(np.zeros(len(AT.LABELS)))
assert tags == [], tags
print("7. autotag threshold OK: orthogonal audio -> []")

# 8. merge: filename tags + auto (RU + short EN), deduped
from app.store import _merge_auto_tags
merged = _merge_auto_tags(
    ["meme01"], [{"ru": "хлопки", "en": "hands clapping, applause",
                  "group": "sfx", "score": 0.9},
                 {"ru": "meme01", "en": "music", "group": "music",
                  "score": 0.5}])
assert merged == ["meme01", "хлопки", "hands clapping", "music"], merged
print("8. merge OK:", merged)

print("\nALL HYBRID STUB TESTS PASSED")
