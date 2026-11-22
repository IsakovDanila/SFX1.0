"""Vector store: ChromaDB when installed, else persistent JSON + numpy.

Deck's D3 note explicitly allows in-memory/fallback search at stage 1.
Both backends expose the same API so the frontend never notices.

Response items follow the team's frontend contract (client.ts):
  {"id", "score", "confidence", "file": {
      "id", "filename", "duration", "sample_rate", "channels",
      "category", "tags", "description",
      "spectral_features": {"pitch", "energy", "has_noise", "has_reverb"}}}

Scanning is INCREMENTAL: files whose mtime+size are unchanged are reused
(rescan of 96 files ~ instant), missing files are pruned, user-edited
tags/descriptions survive rescans. Item ids are stable uuid5(path).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid

import numpy as np

from . import audio_features as af
from . import autotag as AT
from . import embeddings as E
from . import labels_ru

INDEX_PATH = os.getenv("SFX_INDEX_PATH", os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "index.json"))

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aif", ".aiff"}

STORE_SCHEMA = 4  # bump to force a full rescan after feature changes
                  # 3 = русские теги категорий (labels_ru)


def _confidence_from_cosine(c: float) -> int:
    # map cosine [-1..1] (practically [0..1]) to 5..99 %
    return int(np.clip(round(50 * (c + 1) - 1), 5, 99))


def _confidence_from_keyword(s: float) -> int:
    return int(np.clip(round(38 + 61 * s), 5, 99))


def _merge_tags(base: list[str], extra: list[str]) -> list[str]:
    """Дописать теги без дублей, сохранив порядок (сначала исходные)."""
    out, seen = list(base), {str(t).lower() for t in base}
    for t in extra:
        if t and str(t).lower() not in seen:
            seen.add(str(t).lower())
            out.append(t)
    return out


def _merge_auto_tags(base: list[str], auto: list[dict]) -> list[str]:
    """Filename tags + CLAP guesses (RU label + short EN), deduped."""
    out, seen = list(base), {t.lower() for t in base}
    for a in auto or []:
        for cand in (a.get("ru", ""), a.get("en", "").split(",")[0].strip()):
            if cand and cand.lower() not in seen:
                seen.add(cand.lower())
                out.append(cand)
    return out


class JsonStore:
    """JSON-persisted vector store (zero extra deps)."""

    def __init__(self, path: str = INDEX_PATH, embedder=None):
        self.path = os.path.abspath(path)
        self.embedder = embedder or E.HeuristicEmbedder()
        self._tagger: AT.ZeroShotTagger | None = None  # lazy, CLAP mode only
        self.items: list[dict] = []
        self._by_id: dict[str, dict] = {}
        if os.path.exists(self.path):
            self.load()

    # ---------- режим эмбеддера ----------
    def use_embedder(self, embedder, reindex: bool = True) -> None:
        """Переключить эмбеддер (например, когда CLAP догрузился).

        Индекс пересобирается в фоне: у CLAP-векторов другая размерность,
        старые записи эвристики ему не подходят.
        """
        if embedder is None or embedder.name == self.embedder.name:
            return
        self.embedder = embedder
        self._tagger = None
        if not reindex:
            return
        folders = sorted({os.path.dirname(it.get("path", "")) for it in self.items
                          if it.get("path")})
        if not folders:
            return
        root = os.path.commonpath(folders) if len(folders) > 1 else folders[0]
        print(f"[store] переключаюсь на {embedder.name}, переиндексация: {root}")
        threading.Thread(target=self._reindex_safe, args=(root,), daemon=True).start()

    def _reindex_safe(self, folder: str) -> None:
        try:
            self.scan_folder(folder)
            print(f"[store] переиндексация завершена: {len(self.items)} файлов")
        except Exception as e:  # noqa: BLE001
            print(f"[store] переиндексация не удалась: {e}")

    # ---------- persistence ----------
    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema", 1) != STORE_SCHEMA:
            print(f"[store] index schema {data.get('schema', 1)} != "
                  f"{STORE_SCHEMA}, will rebuild on next scan")
            self.items = []
            self._by_id = {}
            return
        items = data.get("items", [])
        # self-healing: drop entries whose files no longer exist on disk
        kept = [it for it in items if os.path.exists(it.get("path", ""))]
        dropped = len(items) - len(kept)
        if dropped:
            print(f"[store] pruned {dropped} missing files from index")
        self.items = kept
        self._by_id = {it["id"]: it for it in self.items}
        if dropped:
            self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"items": self.items, "saved_at": time.time(),
                       "schema": STORE_SCHEMA}, f, ensure_ascii=False)

    # ---------- indexing ----------
    def clear(self) -> None:
        self.items, self._by_id = [], {}

    def add_file(self, file_path: str, category: str = "",
                 tags: list[str] | None = None, description: str = "") -> dict:
        """Build (not append) an index item for a file."""
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        tags = tags or []
        try:
            st = os.stat(file_path)
            mtime, size = float(st.st_mtime), int(st.st_size)
        except OSError:
            mtime, size = 0.0, 0
        feats = af.analyze(file_path)
        try:  # original sample rate / channels (pre-resample, header only)
            import soundfile as sf
            _info = sf.info(file_path)
            orig_sr, channels = int(_info.samplerate), int(_info.channels)
        except Exception:
            orig_sr, channels = af.ANALYSIS_SR, 1
        if isinstance(self.embedder, E.CLAPEmbedder):
            vec = self.embedder.audio_vector_file(file_path)
            if self._tagger is None:  # encode label vocab once per process
                self._tagger = AT.ZeroShotTagger(self.embedder)
            auto_tags = self._tagger.tag(np.asarray(vec, dtype=float))
        else:
            vec = self.embedder.audio_vector(feats)
            auto_tags = []
        if not description:
            # без мусора вида "airplane: " при пустых тегах
            head = labels_ru.ru_name(category) if category else ""
            tail = ", ".join(tags[:8])
            description = f"{head}: {tail}" if head and tail else (tail or head)
        return {
            "id": uuid.uuid5(uuid.NAMESPACE_URL, file_path).hex,
            "path": file_path,
            "filename": filename,
            "category": category,
            "category_ru": labels_ru.ru_name(category),
            "tags": tags,
            "auto_tags": auto_tags,
            "description": description,
            "mtime": mtime,
            "size": size,
            "orig_sr": orig_sr,
            "channels": channels,
            "features": feats,
            "vector": [float(x) for x in np.asarray(vec).ravel().tolist()],
            "embedder": self.embedder.name,
        }

    def scan_folder(self, folder: str, progress_cb=None) -> dict:
        """Index all audio files under folder (incremental).

        Category/tags priority per directory: tags.csv (ground truth)
        -> parent dir name -> filename tokens. Unchanged files (mtime+size)
        are reused with their user-edited tags/descriptions intact.
        """
        import csv

        def _load_truth(csv_path: str, into: dict) -> None:
            try:
                with open(csv_path, encoding="utf-8-sig") as f:
                    base = os.path.dirname(csv_path)
                    for row in csv.DictReader(f):
                        key = os.path.abspath(os.path.join(base, row["filename"]))
                        into[key] = (
                            row.get("category", ""),
                            [t.strip() for t in row.get("tags", "").split(",") if t.strip()],
                            (row.get("description", "") or "").strip(),
                        )
            except (OSError, KeyError):
                pass

        t0 = time.time()
        truth: dict[str, tuple[str, list[str], str]] = {}
        files: list[str] = []
        for root, _, names in os.walk(folder):
            _load_truth(os.path.join(root, "tags.csv"), truth)
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.join(root, n))
        files.sort()

        old_by_path = {it.get("path"): it for it in self.items}
        new_items: list[dict] = []
        added = reused = 0
        errors: list[str] = []
        for i, fp in enumerate(files):
            ap = os.path.abspath(fp)
            try:
                st = os.stat(ap)
                sig = (float(st.st_mtime), int(st.st_size))
            except OSError as e:
                errors.append(f"{fp}: {e}")
                if progress_cb:
                    progress_cb(i + 1, len(files), fp)
                continue
            old = old_by_path.get(ap)
            try:
                if (old is not None
                        and (old.get("mtime"), old.get("size")) == sig
                        and old.get("embedder") == self.embedder.name
                        and "vector" in old):
                    # reuse: обновляем поля из tags.csv, НО не затираем ручную
                    # разметку: пустые теги в CSV больше не стирают теги,
                    # добавленные пользователем через «Разметку».
                    if ap in truth:
                        category, tags, desc = truth[ap]
                        old["category"] = category or old.get("category", "")
                        old["category_ru"] = labels_ru.ru_name(old["category"])
                        old["tags"] = self._tags_for(old["category"], tags, old)
                        if desc:
                            old["description"] = desc
                        elif not old.get("description"):
                            old["description"] = self._auto_description(
                                old["category"], old["tags"])
                    elif not old.get("tags"):
                        # запись без тегов (например, старый индекс) — досыпаем
                        old["tags"] = self._tags_for(old.get("category", ""), [], old)
                    old["category_ru"] = labels_ru.ru_name(old.get("category", ""))
                    new_items.append(old)
                    reused += 1
                else:
                    if ap in truth:
                        category, tags, desc = truth[ap]
                    else:
                        name = os.path.basename(fp)
                        parent = os.path.basename(os.path.dirname(fp)) or ""
                        # папка может называться с большой буквы или по-русски:
                        # приводим к ключу словаря, чтобы нашлись русские метки
                        cat_key = labels_ru.normalize_key(parent)
                        category = (cat_key if labels_ru.is_known(cat_key)
                                    else parent)
                        tags = self._tags_from_filename(name)
                        desc = ""
                    item = self.add_file(ap, category=category,
                                         tags=self._tags_for(category, tags),
                                         description=desc)
                    # keep user edits when re-analyzing a changed file
                    if old is not None:
                        merged = _merge_tags(old.get("tags", []), item["tags"])
                        item["tags"] = merged
                        if old.get("description"):
                            item["description"] = old["description"]
                    else:
                        # brand-new file: русские метки + имя файла + CLAP-гипотезы
                        item["tags"] = _merge_auto_tags(
                            item["tags"], item.get("auto_tags", []))
                    new_items.append(item)
                    added += 1
            except Exception as e:  # noqa: BLE001 - keep indexing on errors
                errors.append(f"{fp}: {e}")
            if progress_cb:
                progress_cb(i + 1, len(files), fp)
        self.items = new_items
        self._by_id = {it["id"]: it for it in new_items}
        self.save()
        dt = time.time() - t0
        # «слепые» файлы: о звуке нет ни одного слова (имя вида 1-20133-A-39.wav
        # и нет tags.csv). Честно сообщаем в UI, что их найдёт только звук/теги.
        # В CLAP-режиме вектор есть у каждого файла: по словам он может не
        # находиться, но по смыслу и референсу находится. Предупреждать не о чем.
        if isinstance(self.embedder, E.CLAPEmbedder):
            blind = []
        else:
            blind = [it["filename"] for it in new_items
                     if not it.get("tags") and not it.get("auto_tags")]
        return {"indexed": len(new_items), "added": added, "reused": reused,
                "skipped_errors": errors, "seconds": round(dt, 2),
                "without_tags": len(blind),
                "without_tags_examples": blind[:5]}

    # ---------- tags ----------
    @staticmethod
    def _auto_description(category: str, tags: list[str]) -> str:
        head = labels_ru.ru_name(category) if category else ""
        tail = ", ".join(tags[:8])
        return f"{head}: {tail}" if head and tail else (tail or head)

    @staticmethod
    def _tags_from_filename(name: str) -> list[str]:
        """Теги из имени файла: англ. слова + русские метки категории.

        `glass_breaking_01.wav` -> [glass, breaking, разбитое стекло, стекло, ...]
        Работает и без tags.csv, и с русскими именами файлов.
        """
        stem = os.path.splitext(name)[0]
        toks = [t for t in E.tokenize(stem) if len(t) > 1 and not t.isdigit()]
        out = list(toks)
        # голосуем: сколько слов имени указывают на каждую категорию
        # («капли воды» -> water_drops 2 голоса, «капли» -> rain 1 голос)
        votes: dict[str, int] = {}
        for tok in toks:
            for key in labels_ru.keys_for_token_prefix(tok):
                votes[key] = votes.get(key, 0) + 1
        if votes:
            best = max(votes.values())
            winners = [k for k, v in votes.items() if v == best]
            for key in sorted(winners):
                out = _merge_tags(out, labels_ru.ru_tags(key, limit=4))
        # локаль папки в имени файла (asr/glass_breaking_01.wav)
        norm = re.sub(r"[^a-zа-яё_]+", "_", stem.lower())
        for key in labels_ru.RU_LABELS:
            if re.search(rf"(^|_){re.escape(key.lstrip('_'))}(_|$)", norm):
                out = _merge_tags(out, labels_ru.ru_tags(key))
        return out

    def _tags_for(self, category: str, csv_tags: list[str],
                  old: dict | None = None) -> list[str]:
        """Итоговые теги записи: теги из tags.csv -> + русские метки категории.

        `old` (существующая запись) сохраняет свою ручную разметку: её теги
        идут первыми и никогда не теряются.
        """
        manual = list(old.get("tags", [])) if old else []
        out = _merge_tags(manual, csv_tags or [])
        # метки добавляем ТОЛЬКО для известных категорий: иначе в теги
        # попадало имя папки («Моя музыка», «demo_sfx») — мусор в разметке
        if labels_ru.is_known(category):
            out = _merge_tags(out, labels_ru.ru_tags(category))
        return out

    # ---------- shapes ----------
    @staticmethod
    def their_shape(it: dict) -> dict:
        """File object in the frontend's SearchResult.file shape."""
        f = it.get("features", {}) or {}
        cent = f.get("spectral_centroid_mean", 0) or 0
        pitch = "низкий" if cent < 500 else ("средний" if cent < 2000 else "высокий")
        rms = f.get("rms_energy", 0) or 0
        energy = "low" if rms < 0.05 else ("medium" if rms < 0.15 else "high")
        flat = f.get("spectral_flatness_mean")
        if flat is None:  # indexes built before the fast analyzer
            has_noise = (f.get("percussive_ratio", 0) or 0) > 0.35
        else:
            has_noise = (flat or 0) > 0.3
        return {
            "id": it.get("id"),
            "filename": it.get("filename"),
            "duration": (f.get("duration", 0) or 0),
            "sample_rate": it.get("orig_sr", af.ANALYSIS_SR),
            "channels": it.get("channels", 1),
            "category": it.get("category", "") or "Без категории",
            "category_ru": labels_ru.ru_name(it.get("category", "")),
            "tags": it.get("tags", []),
            "auto_tags": it.get("auto_tags", []),
            "description": it.get("description", ""),
            "spectral_features": {
                "pitch": pitch,
                "energy": energy,
                "has_noise": has_noise,
                "has_reverb": (f.get("reverb_proxy", 0) or 0) > 0.25,
            },
        }

    def _confidence_clap(self, cos: float, fused: float,
                        kw: float = 0.0, margin: float = 1.0) -> int:
        """Уверенность для CLAP-поиска.

        Три сигнала, а не один абсолютный косинус:
          margin — отрыв по косинусу от следующего результата (0..1);
          kw     — попадание по словам/русским тегам (0..1);
          prob   — калиброванная вероятность по абсолютному косинусу.
        Абсолютный косинус CLAP на шумных записях низкий даже у верного
        файла (0.30-0.40), поэтому отрыв и слова весят больше него.

        Почему margin: если топ-1 оторвался от второго на 0.05+, это
        осмысленный ответ даже при косинусе 0.34. Если же весь список
        идёт вплотную (0.34, 0.29, 0.29), честнее показать низкий процент.
        """
        prob = self.embedder.text_audio_prob(cos) if hasattr(
            self.embedder, "text_audio_prob") else (cos + 1) / 2
        margin = min(1.0, max(0.0, margin))
        kw = min(1.0, max(0.0, kw))
        fused = min(1.0, max(0.0, fused))
        val = (0.40 * margin + 0.30 * kw + 0.20 * prob + 0.10 * fused)
        return int(np.clip(round(val * 100), 5, 99))

    @staticmethod
    def _margins(coss: list[float]) -> list[float]:
        """Отрыв каждого результата от следующего в порядке убывания.

        Нормируем на разброс всего списка: 1.0 — явный лидер,
        0.0 — все косинусы почти одинаковые (нет уверенности).
        """
        if not coss:
            return []
        order = sorted(range(len(coss)), key=lambda i: -coss[i])
        spread = max(coss) - min(coss)
        out = [0.0] * len(coss)
        for rank, i in enumerate(order):
            nxt = coss[order[rank + 1]] if rank + 1 < len(order) else min(coss)
            gap = coss[i] - nxt
            # 2.0: отрыв в половину разброса — уже «явный лидер»
            out[i] = min(1.0, 2.0 * gap / spread) if spread > 1e-6 else 0.0
        return out

    def _hit(self, it: dict, score: float, confidence: int) -> dict:
        return {"id": it["id"], "score": round(float(score), 4),
                "confidence": confidence, "file": self.their_shape(it)}

    # ---------- reads ----------
    def get(self, item_id: str) -> dict | None:
        return self._by_id.get(item_id)

    def library(self, category: str = "", q: str = "") -> list[dict]:
        """Catalog with fallback text filter (deck risk mitigation: Plan B)."""
        out = self.items
        if category:
            out = [it for it in out if it.get("category") == category]
        if q:
            toks = E.tokenize(q)
            scored = [(E.keyword_score(toks, it), it) for it in out]
            out = [it for s, it in sorted(scored, key=lambda x: -x[0]) if s > 0]
        return [self._public(it) for it in out]

    def categories(self) -> list[dict]:
        counts: dict[str, int] = {}
        for it in self.items:
            counts[it.get("category", "")] = counts.get(it.get("category", ""), 0) + 1
        return [{"name": k, "count": v} for k, v in sorted(counts.items())]

    # ---------- search ----------
    def search_text(self, query: str, top_k: int = 10) -> dict:
        t0 = time.time()
        if isinstance(self.embedder, E.CLAPEmbedder):
            # hybrid: CLAP semantics + keyword tags, fused by RRF.
            # Keywords keep RU queries working (CLAP text encoder is EN-only).
            qv = self.embedder.text_vector(query)
            clap_scores = [E.cosine(qv, np.array(it["vector"]))
                           for it in self.items]
            kw_scores = E.HeuristicEmbedder().text_scores(query, self.items)
            n = len(self.items)
            order_c = sorted(range(n), key=lambda i: -clap_scores[i])
            order_k = sorted(range(n), key=lambda i: -kw_scores[i])
            rank_c = {i: r for r, i in enumerate(order_c)}
            rank_k = {i: r for r, i in enumerate(order_k)}
            k_rrf = 60
            fused = [(1 / (k_rrf + rank_c[i]) + 1 / (k_rrf + rank_k[i]), i)
                     for i in range(n)]
            fused.sort(key=lambda x: -x[0])
            best = fused[0][0] if fused else 1.0
            margins = self._margins(clap_scores)
            confs = [self._confidence_clap(clap_scores[i], rrf / best,
                                           kw_scores[i], margins[i])
                     for rrf, i in fused[:top_k]]
            # проценты не должны расти вниз по списку: иначе #3 с точным
            # тегом выглядит «лучше» #2, хотя ранжирование поставило его ниже
            # int(...) обязателен: np.minimum.accumulate возвращает np.int64,
            # а такой тип не сериализуется в JSON -> HTTP 500 на каждом запросе
            confs = [int(c) for c in np.minimum.accumulate(confs)]
            results = [self._hit(self.items[i], c / 100.0, c)
                       for c, (rrf, i) in zip(confs, fused[:top_k])]
            best_cos = max(clap_scores) if clap_scores else 0.0
            best_kw = max(kw_scores) if kw_scores else 0.0
            # порог: CLAP не умеет говорить «не знаю», поэтому без порога на
            # любой бред возвращались 10 файлов с процентами («чёрный ящик»)
            min_cos = float(os.getenv("SFX_CLAP_MIN_COS", "0.18"))
            weak = best_cos < max(min_cos, 0.12) and best_kw <= 0
            if weak and os.getenv("SFX_CLAP_HONEST_EMPTY", "1") == "1":
                return {"query": query, "mode": "clap+keywords", "top_k": top_k,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "low_confidence": True, "results": [],
                        "hint": "Ничего похожего. Попробуйте другой запрос "
                                "(CLAP понимает английский) или добавьте теги."}
            mode, low_conf = "clap+keywords", bool(weak)
        else:
            scores = self.embedder.text_scores(query, self.items)
            best_kw = max(scores) if scores else 0.0
            if best_kw <= 0:
                # honest empty: no lexical evidence at all
                results, low_conf = [], True
            else:
                scored = [(s, _confidence_from_keyword(s), it)
                          for s, it in zip(scores, self.items) if s > 0]
                scored.sort(key=lambda x: -x[0])
                results = [self._hit(it, s, c) for s, c, it in scored[:top_k]]
                low_conf = False
            mode = "heuristic-demo"
        return {"query": query, "mode": mode, "top_k": top_k,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "low_confidence": low_conf, "results": results}

    def search_vector(self, vec: np.ndarray, top_k: int = 10,
                      exclude_id: str = "") -> list[dict]:
        scored = []
        for it in self.items:
            if it["id"] == exclude_id:
                continue
            c = E.cosine(vec, np.array(it["vector"]))
            # калибровка как у текст->аудио: линейная формула 50*(c+1)-1
            # давала ~45% даже верному файлу (косинус аудио-аудио ниже)
            prob = (self.embedder.audio_audio_prob(c)
                    if hasattr(self.embedder, "audio_audio_prob")
                    else (c + 1) / 2)
            conf = int(np.clip(round(prob * 100), 5, 99))
            scored.append((c, self._hit(it, c, conf)))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]

    def search_audio_text(self, vec: np.ndarray, query: str = "",
                          top_k: int = 10, exclude_id: str = "") -> dict:
        """Референс + описание одновременно.

        Три ранжирования сливаются через RRF с весами:
          - аудио-косинус к загруженному файлу (главный сигнал),
          - CLAP текст->аудио (семантика описания, только в CLAP-режиме),
          - ключевые слова/RU-теги (работает без CLAP и на русском).
        Описание работает как уточнение: «похоже на этот взрыв, но деревянный».
        """
        t0 = time.time()
        q = (query or "").strip()
        if not q:
            res = self.search_vector(vec, top_k, exclude_id=exclude_id)
            return {"mode": "audio-only", "low_confidence": False,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "results": res}

        w_audio = float(os.getenv("SFX_W_AUDIO", "1.0"))
        w_clap = float(os.getenv("SFX_W_CLAPTEXT", "0.8"))
        w_kw = float(os.getenv("SFX_W_KEYWORDS", "0.5"))

        n = len(self.items)
        a_scores = [(-1.0 if it["id"] == exclude_id
                     else E.cosine(vec, np.array(it["vector"])))
                    for it in self.items]
        kw_scores = E.HeuristicEmbedder().text_scores(q, self.items)

        clap_scores: list[float] | None = None
        if isinstance(self.embedder, E.CLAPEmbedder):
            qv = self.embedder.text_vector(q)
            clap_scores = [E.cosine(qv, np.array(it["vector"])) for it in self.items]

        rankings = [(a_scores, w_audio), (kw_scores, w_kw)]
        if clap_scores is not None:
            rankings.append((clap_scores, w_clap))

        k_rrf = 60
        fused = [0.0] * n
        for scores, w in rankings:
            order = sorted(range(n), key=lambda i: -scores[i])
            for r, i in enumerate(order):
                fused[i] += w / (k_rrf + r)

        def _prob(i: int) -> int:
            pa = (self.embedder.audio_audio_prob(a_scores[i])
                  if hasattr(self.embedder, "audio_audio_prob")
                  else (a_scores[i] + 1) / 2)
            num, den = w_audio * pa, w_audio
            if clap_scores is not None:
                pt = self.embedder.text_audio_prob(clap_scores[i])
                num += w_clap * pt
                den += w_clap
            if kw_scores[i] > 0:
                pk = _confidence_from_keyword(kw_scores[i]) / 100.0
                num += w_kw * pk
                den += w_kw
            return int(np.clip(round(num / max(den, 1e-9) * 100), 5, 99))

        order = sorted(range(n), key=lambda i: -fused[i])
        hits = []
        for i in order[:top_k]:
            conf = _prob(i)
            hits.append(self._hit(self.items[i], conf / 100.0, conf))

        best_cos = max(a_scores) if a_scores else 0.0
        best_kw = max(kw_scores) if kw_scores else 0.0
        min_cos = float(os.getenv("SFX_CLAP_MIN_COS", "0.18"))
        weak = best_cos < max(min_cos, 0.12) and best_kw <= 0
        mode = "audio+clap+keywords" if clap_scores is not None else "audio+keywords"
        if weak and os.getenv("SFX_CLAP_HONEST_EMPTY", "1") == "1":
            return {"mode": mode, "low_confidence": True, "results": [],
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "hint": "Ничего похожего ни по звуку, ни по описанию."}
        return {"mode": mode, "low_confidence": bool(weak), "results": hits,
                "elapsed_ms": int((time.time() - t0) * 1000)}

    def similar(self, item_id: str, top_k: int = 10) -> list[dict]:
        it = self.get(item_id)
        if not it:
            return []
        return self.search_vector(np.array(it["vector"]), top_k, exclude_id=item_id)

    def annotate(self, item_id: str, tags: list[str]) -> dict | None:
        it = self.get(item_id)
        if not it:
            return None
        it["tags"] = tags
        self.save()
        return self._public(it)

    def set_description(self, item_id: str, description: str) -> dict | None:
        it = self.get(item_id)
        if not it:
            return None
        it["description"] = description
        self.save()
        return self._public(it)

    @staticmethod
    def _public(it: dict) -> dict:
        return {k: it.get(k) for k in
                ("id", "filename", "category", "tags", "auto_tags",
                 "description", "features")}


def get_store() -> JsonStore:
    # ChromaDB backend can be plugged here later (same method names).
    try:
        import chromadb  # noqa: F401
        print("[store] chromadb installed, but JsonStore keeps MVP simple; "
              "override get_store() to switch backends.")
    except ImportError:
        pass
    return JsonStore(embedder=E.get_embedder())
