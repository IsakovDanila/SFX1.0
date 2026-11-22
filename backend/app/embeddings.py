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

from . import labels_ru

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

# Русские метки для всех остальных категорий (ESC-50 и чужие библиотеки).
# Ключи не пересекаются с SYNONYMS выше, поэтому обновление безопасно.
for _key, _entry in labels_ru.RU_LABELS.items():
    SYNONYMS.setdefault(_key, [])
    for _w in _entry["tags"]:
        if _w not in SYNONYMS[_key]:
            SYNONYMS[_key].append(_w)

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
    # RU genitive/plural ending "ев/ов": шагов->шаг, звуков->звук.
    # Без этого «шагов» не совпадало с «шаг» (5 букв vs 3, префикс не помогал).
    if len(tok) > 4 and tok[-2:] in ("ев", "ов"):
        return tok[:-2]
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


def expand_query(query_tokens: list[str]) -> set[str]:
    """Перевод запроса в ключи категорий: «разбилось стекло» -> {glass_breaking}.

    Работает без tags.csv: даже если в индексе только английские имена файлов,
    русский запрос находит нужный класс.
    """
    keys: set[str] = set()
    for tok in query_tokens:
        keys |= labels_ru.keys_for_token_prefix(tok)
        # EN-слово запроса -> RU -> ключи
        for ru in EN_RU.get(_stem(tok), ()):
            keys |= labels_ru.keys_for_token_prefix(ru)
    return keys


# Служебные слова: «sound of a bells» не должно считаться промахом в 3/4
# токена. Убираем ДО подсчёта, но НЕ трогаем expand_query — там они безвредны.
STOPWORDS = {
    "a", "an", "the", "of", "sound", "sounds", "some", "any", "and", "or",
    "to", "in", "on", "at", "with", "for", "from", "by", "is", "it", "that",
    "this", "my", "me", "please", "like", "similar", "kind", "type",
    "звук", "звуки", "звука", "звуков", "шум", "шумы", "шума", "и", "в", "во",
    "на", "с", "со", "для", "от", "по", "из", "что", "это", "как", "типа",
    "похожий", "похоже", "похожие", "пожалуйста", "мне", "нужен", "нужно",
}


def content_tokens(tokens: list[str]) -> list[str]:
    """Убрать служебные слова. Если остались только они — вернуть как было."""
    out = [t for t in tokens if t not in STOPWORDS]
    return out or list(tokens)


def keyword_score(query_tokens: list[str], item: dict) -> float:
    """Lexical match of query against category synonyms + tags + filename."""
    hay: set[str] = set()
    cat_key = labels_ru.normalize_key(item.get("category", ""))
    for w in SYNONYMS.get(cat_key, []):
        hay.add(_stem(w))
    # русское название категории тоже считается попаданием
    for w in tokenize(labels_ru.ru_name(item.get("category", ""))):
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
    toks = content_tokens(query_tokens)
    stems = [_stem(t) for t in toks]
    hits = sum(1 for t in stems if any(_match(t, h) for h in hay))
    score = hits / len(toks)
    # стадия «перевод запроса в категорию»: если класс угадан (в т.ч. по
    # английскому имени файла при отсутствии tags.csv) — это сильный сигнал.
    # 0.85, а не 0.75: точное попадание в category из tags.csv — это ответ,
    # а не догадка (запрос «bells» -> файлы church_bells).
    if cat_key in expand_query(query_tokens):
        score = max(score, 0.85)
    return score


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
    """Real semantic embeddings via laion-clap (optional heavy dep).

    Параметры позволяют ставить модель офлайн и на слабых машинах:
      ckpt     — путь к локальному .pt (иначе пакет скачает ~1.9 ГБ сам),
      device   — "cpu" | "cuda:0" (по умолчанию автоопределение),
      model_id — 0: 630k-best, 1: 630k-audioset-best.

    Загрузка тяжёлая, поэтому обычный путь — через clap_runtime.RUNTIME,
    который грузит модель в фоне и не подвешивает HTTP-сервер.
    """
    name = "laion-clap"
    dim = 512

    def __init__(self, ckpt: str | None = None, device: str | None = None,
                 model_id: int = 1) -> None:
        import torch
        from laion_clap import CLAP_Module  # lazy: only in full mode

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ckpt = ckpt or ""
        self.model = CLAP_Module(enable_fusion=False, device=self.device)
        if ckpt:
            self.model.load_ckpt(ckpt=ckpt, verbose=False)
        else:
            self.model.load_ckpt(model_id=model_id)
        self.model.eval()

    def _norm(self, arr) -> np.ndarray:
        v = np.array(arr, dtype=float).ravel()
        return v / (np.linalg.norm(v) + 1e-12)

    # CLAP нарезает вход на окна по 10 с и центрирует вокруг громкой части.
    # Для коротких SFX (0.2–2 с) остальное — тишина, и вектор «размазывается».
    # Поэтому режем с тишины по краям и ограничиваем длину.
    CLAP_SR = 48000              # родная частота модели, ресемплим сами
    MAX_SECONDS = 10.0           # окно CLAP
    SILENCE_DB = -45.0           # порог тишины для обрезки краёв

    def _trim_silence(self, y: np.ndarray) -> np.ndarray:
        """Срезать тишину в начале/конце (окнами по 20 мс)."""
        if y.size == 0:
            return y
        win = max(1, int(self.CLAP_SR * 0.02))
        n = y.size // win
        if n < 2:
            return y
        frames = y[: n * win].reshape(n, win)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        peak = float(rms.max()) + 1e-12
        thr = peak * (10 ** (self.SILENCE_DB / 20))
        loud = np.where(rms > thr)[0]
        if loud.size == 0:
            return y
        start = loud[0] * win
        end = min(y.size, (loud[-1] + 1) * win)
        # запас по 50 мс, чтобы не срезать атаку и хвост реверба
        pad = int(self.CLAP_SR * 0.05)
        start = max(0, start - pad)
        end = min(y.size, end + pad)
        trimmed = y[start:end]
        # CLAP заполняет окно 10 с повтором сигнала (repeatpad) и считает
        # n_repeat = int(480000 / len(y)). Файл ровно 5 с = 240000 сэмплов
        # даёт 2 повтора без нулей, но запас +50 мс добавляет сэмплы, и
        # 240001 сэмпл даёт n_repeat = 1: ПОЛОВИНА окна уходит в нули.
        # Поэтому подрезаем результат до 1/N окна, если перебор меньше
        # одного сэмпла на повтор: так 5-секундные файлы снова получают
        # целое число повторов.
        total = trimmed.size
        if total > 1:
            n_win = int(round(self.CLAP_SR * self.MAX_SECONDS))
            # round(), а не //: для 240001 сэмпла CLAP сам посчитает
            # int(480000/240001) = 1 повтор, но «идеально» их 2 — и надо
            # подрезать до 240000, чтобы повторов стало ровно 2.
            reps = max(1, int(round(n_win / total)))
            per = n_win // reps
            # подрезаем, только если разница с идеальным повтором —
            # это лишний запас (не более 0.5% длины), а не суть звука
            if per >= 1 and 0 < total - per <= max(1, total // 200):
                trimmed = trimmed[:per]
        return trimmed

    def _load_for_clap(self, path: str) -> np.ndarray:
        import librosa
        y, _ = librosa.load(path, sr=self.CLAP_SR, mono=True,
                            duration=self.MAX_SECONDS)
        return self._trim_silence(y)

    def audio_vector_file(self, path: str) -> np.ndarray:
        # get_audio_embedding_from_data принимает уже готовый waveform в 48 кГц,
        # поэтому обрезка не «съедается» повторной загрузкой внутри пакета.
        y = self._load_for_clap(path)
        if y.size == 0:
            raise ValueError(f"нет звука после обрезки тишины: {path}")
        # ВАЖНО: список, а не массив. laion_clap/hook.py делает
        # `for audio_waveform in x`, то есть ждёт батч дорожек.
        # Для 1-D массива он итерирует ПО СЭМПЛАМ, и каждый сэмпл
        # (numpy.float32, 0 измерений) падает в torch.from_numpy ->
        # "TypeError: object of type 'numpy.float32' has no len()".
        emb = self.model.get_audio_embedding_from_data(
            [np.asarray(y, dtype=np.float32)], use_tensor=False)
        return self._norm(emb[0])

    # Несколько формулировок на запрос: усреднение векторов даёт заметно
    # устойчивее результат, чем одна фраза (работает без дообучения модели).
    MAX_QUERY_VARIANTS = 3

    def text_vector(self, query: str) -> np.ndarray:
        variants = [v.strip() for v in (query or "").split("|") if v.strip()]
        variants = variants[: self.MAX_QUERY_VARIANTS] or [query]
        vecs = []
        for v in variants:
            emb = self.model.get_text_embedding([v], use_tensor=False)
            vecs.append(self._norm(emb[0]))
        out = np.mean(vecs, axis=0)
        return out / (np.linalg.norm(out) + 1e-12)

    # --- калибровка уверенности ---------------------------------------
    # CLAP отдаёт косинус в узком диапазоне (релевантное ~0.3-0.6), поэтому
    # «процент совпадения» из презентации нельзя брать как cos*100: всё
    # выглядело бы на 30-50%. Переводим в вероятность логистической кривой.
    # SFX_CLAP_CALIB=lo,hi позволяет подогнать под свою библиотеку.
    def text_audio_prob(self, cos: float) -> float:
        lo, hi = 0.20, 0.55
        env = (os.getenv("SFX_CLAP_CALIB") or "").strip()
        if "," in env:
            try:
                lo, hi = (float(x) for x in env.split(",", 1))
            except ValueError:
                pass
        steep = 12.0 / max(hi - lo, 1e-6)
        mid = (lo + hi) / 2
        return float(1 / (1 + np.exp(-steep * (cos - mid))))

    # Аудио-аудио калибровка. Два разных звука одного класса дают косинус
    # 0.4-0.7 (это не ошибка: CLAP-вектор — усреднённое описание, а не хеш).
    # Поэтому шкала мягче, чем для текст->аудио.
    # Настройка: SFX_CLAP_CALIB_AUDIO=lo,hi
    def audio_audio_prob(self, cos: float) -> float:
        lo, hi = 0.30, 0.65
        env = (os.getenv("SFX_CLAP_CALIB_AUDIO") or "").strip()
        if "," in env:
            try:
                lo, hi = (float(x) for x in env.split(",", 1))
            except ValueError:
                pass
        steep = 12.0 / max(hi - lo, 1e-6)
        mid = (lo + hi) / 2
        return float(1 / (1 + np.exp(-steep * (cos - mid))))


def get_embedder():
    """Factory без блокирующей загрузки.

    CLAP весит ~2 ГБ и качается с сети, поэтому при старте мы НЕ грузим его
    здесь: возвращаем эвристику, а `clap_runtime.RUNTIME` догружает модель
    в фоне. Когда модель готова, индекс пересобирается (store.reindex_async).
    Так сервер отвечает сразу, а UI видит state="loading" и прогресс.
    """
    from . import clap_runtime
    if clap_runtime.ClapRuntime.enabled():
        ready = clap_runtime.RUNTIME.embedder()
        if ready is not None:
            print("[embeddings] CLAP готов, используем его")
            return ready
        print("[embeddings] CLAP-режим: модель грузится в фоне, "
              "пока работаю на эвристике (см. /api/health)")
        return HeuristicEmbedder()
    print("[embeddings] heuristic demo mode (set SFX_USE_CLAP=1 for real CLAP)")
    return HeuristicEmbedder()
