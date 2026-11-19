"""Embedding layer.

Two modes (env SFX_USE_CLAP=1 enables the full one):
  * CLAP mode   — real semantic text<->audio embeddings (laion-clap).
                  Needs torch + model download (~2-4 GB). Best quality.
  * Heuristic   — zero-dependency demo mode (default):
      - text search  = keyword/synonym matching over category + tags + filename
      - audio search = cosine similarity over MFCC-based vectors
    This is the "API: demo-mode" from the mockup footer: good enough to
    develop the UI/UX and validate the pipeline on CPU.

Why heuristic exists: the deck's D3 note allows in-memory/fallback search
at the first stage, and CLAP needs GPU/RAM the team may not have yet.

Demo mode is bilingual (EN + RU): the team's UI/mock data is in Russian
("гулкий шаг по дереву"), so Russian synonyms + inflection-tolerant
matching are built in.
"""
from __future__ import annotations

import os
import re

import numpy as np

# Synonym map per demo category: EN (team decision) + RU (UI language).
# 'cinematic' belongs to both explosion and whoosh: exact-token count
# still disambiguates ("cinematic explosion" vs "cinematic whoosh").
SYNONYMS: dict[str, list[str]] = {
    "footstep_wood": ["footstep", "footsteps", "step", "steps", "walking", "walk",
                      "wood", "wooden", "floor", "stomp", "thump", "boots",
                      "шаг", "шаги", "дерево", "деревянный", "пол", "поступь",
                      "гулкий", "глухой", "топот", "ботинки", "ходьба"],
    "explosion_cinematic": ["explosion", "explode", "blast", "boom", "bomb",
                            "cinematic", "epic", "detonation", "burst",
                            "взрыв", "кинематографичный", "эпичный", "мощный",
                            "грохот", "детонация", "бум"],
    "interface_click": ["click", "ui", "interface", "button", "mouse", "tap",
                        "blip", "notification", "select", "press",
                        "клик", "интерфейс", "кнопка", "мягкий", "нажатие",
                        "уведомление", "блип"],
    "metal_hit": ["metal", "metallic", "clang", "clank", "steel", "iron", "hit",
                  "impact", "hammer", "anvil", "gong",
                  "металл", "металлический", "удар", "лязг", "звон", "сталь",
                  "железо", "молот", "тяжёлый", "тяжелый", "гонг"],
    "glass_break": ["glass", "break", "shatter", "shattering", "smash", "crash",
                    "window", "bottle",
                    "стекло", "разбивание", "звон", "разбить", "осколки",
                    "окно", "бутылка"],
    "whoosh": ["whoosh", "swoosh", "sweep", "transition", "pass", "flyby",
               "wind", "sweeping", "cinematic",
               "свист", "пролёт", "пролет", "вжух", "переход", "ветер",
               "взмах", "свистящий"],
    "thunder": ["thunder", "lightning", "storm", "rumble", "weather", "rain",
                "гром", "гроза", "молния", "шторм", "рокот", "раскат"],
    "gunshot": ["gunshot", "gun", "shot", "firearm", "pistol", "rifle", "shoot",
                "firing", "fire",
                "выстрел", "пистолет", "винтовка", "оружие", "стрельба",
                "огонь", "ружьё", "ружье"],
}

TOKEN_RE = re.compile(r"[a-zа-яё']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _stem(tok: str) -> str:
    # EN plurals: steps->step, crashes->crash (glass stays glass)
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    # RU plural-ish: шаги->шаг, взрывы->взрыв
    if len(tok) > 3 and tok[-1] in ("и", "ы"):
        return tok[:-1]
    return tok


# EN<->RU equivalents for real-world libraries with EN filenames:
# lets RU queries match EN tags and vice versa.
EN_RU_RAW: dict[str, list[str]] = {
    "metal": ["металл", "металлический"], "metallic": ["металл", "металлический"],
    "hit": ["удар"], "impact": ["удар"], "clang": ["лязг"], "clank": ["лязг"],
    "steel": ["сталь"], "iron": ["железо"], "hammer": ["молот"],
    "heavy": ["тяжёлый", "тяжелый"],
    "footstep": ["шаг"], "step": ["шаг"], "steps": ["шаг", "шаги"],
    "wood": ["дерево"], "wooden": ["дерево", "деревянный"], "floor": ["пол"],
    "explosion": ["взрыв"], "blast": ["взрыв"], "boom": ["взрыв", "грохот"],
    "cinematic": ["кинематографичный"], "epic": ["эпичный"],
    "click": ["клик"], "button": ["кнопка"], "interface": ["интерфейс"],
    "glass": ["стекло"], "break": ["разбить", "разбивание"],
    "shatter": ["разбить", "осколки"], "smash": ["разбить"],
    "whoosh": ["свист"], "swoosh": ["свист"], "sweep": ["свист"],
    "thunder": ["гром"], "storm": ["шторм", "гроза"], "lightning": ["молния"],
    "rain": ["дождь"], "rumble": ["рокот", "грохот"], "wind": ["ветер"],
    "gunshot": ["выстрел"], "gun": ["выстрел", "оружие"], "shot": ["выстрел"],
    "pistol": ["пистолет"], "rifle": ["винтовка"], "firearm": ["оружие"],
    "fire": ["огонь"], "shoot": ["стрельба"], "firing": ["стрельба"],
    "stomp": ["топот"], "thump": ["стук", "топот"], "tap": ["стук"],
    "blip": ["блип"], "transition": ["переход"],
}


def _build_equiv():
    en_ru: dict[str, set[str]] = {}
    ru_en: dict[str, set[str]] = {}
    for en, rus in EN_RU_RAW.items():
        es = _stem(en)
        en_ru.setdefault(es, set()).update(_stem(r) for r in rus)
        for r in rus:
            ru_en.setdefault(_stem(r), set()).add(es)
    return en_ru, ru_en


EN_RU, RU_EN = _build_equiv()


def _match(t: str, h: str) -> bool:
    """Exact match or shared 4-letter prefix (tolerates RU inflections:
    'дереву' vs 'дерево', 'интерфейса' vs 'интерфейс')."""
    if t == h:
        return True
    if len(t) >= 4 and len(h) >= 4 and (t.startswith(h[:4]) or h.startswith(t[:4])):
        return True
    return False


def keyword_score(query_tokens: list[str], item: dict) -> float:
    """Lexical match of query against category synonyms + tags + filename."""
    hay: set[str] = set()
    for w in SYNONYMS.get(item.get("category", ""), []):
        hay.add(_stem(w))
    for t in tokenize(" ".join(item.get("tags", []))):
        hay.add(_stem(t))
    for t in tokenize(item.get("filename", "")):
        hay.add(_stem(t))
    # cross-language expansion: EN tags <-> RU equivalents
    for h in list(hay):
        extra = EN_RU.get(h)
        if extra:
            hay.update(extra)
        extra = RU_EN.get(h)
        if extra:
            hay.update(extra)
    if not query_tokens:
        return 0.0
    stems = [_stem(t) for t in query_tokens]
    hits = sum(1 for t in stems if any(_match(t, h) for h in hay))
    return hits / len(query_tokens)


def mfcc_vector(features: dict) -> np.ndarray:
    v = np.array(features.get("mfcc_mean", [0.0] * 13), dtype=float)
    if v.shape[0] != 13:
        v = np.zeros(13)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class HeuristicEmbedder:
    """Zero-dependency embedder (demo mode)."""
    name = "heuristic-demo"
    dim = 13

    def audio_vector(self, features: dict) -> np.ndarray:
        return mfcc_vector(features)

    def text_scores(self, query: str, items: list[dict]) -> list[float]:
        toks = tokenize(query)
        return [keyword_score(toks, it) for it in items]


class CLAPEmbedder:
    """Real semantic embeddings via laion-clap (optional heavy dep)."""
    name = "laion-clap"
    dim = 512

    def __init__(self) -> None:
        from laion_clap import CLAP_Module  # lazy: only in full mode
        self.model = CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # default 630k-audioset checkpoint

    def audio_vector_file(self, path: str) -> np.ndarray:
        emb = self.model.get_audio_embedding_from_filelist([path], use_tensor=False)
        v = np.array(emb[0], dtype=float)
        return v / (np.linalg.norm(v) + 1e-12)

    def text_vector(self, query: str) -> np.ndarray:
        emb = self.model.get_text_embedding([query], use_tensor=False)
        v = np.array(emb[0], dtype=float)
        return v / (np.linalg.norm(v) + 1e-12)


def get_embedder():
    """Factory: CLAP if SFX_USE_CLAP=1 and import works, else heuristic."""
    if os.getenv("SFX_USE_CLAP") == "1":
        try:
            emb = CLAPEmbedder()
            print("[embeddings] CLAP mode enabled")
            return emb
        except Exception as e:  # noqa: BLE001 - fall back gracefully
            print(f"[embeddings] CLAP failed ({e}), falling back to heuristic demo mode")
    else:
        print("[embeddings] heuristic demo mode (set SFX_USE_CLAP=1 for real CLAP)")
    return HeuristicEmbedder()
