"""Регресс-тесты русских тегов и сохранности ручной разметки.

Запуск (без pytest, без тяжёлых зависимостей):
    python3 scripts/test_ru_tags.py

Проверяем 4 свойства, которые легко сломать при правках store.py:
  1. при сканировании пишутся РУССКИЕ теги категории;
  2. русский запрос находит нужный класс;
  3. теги, добавленные пользователем, переживают повторный скан;
  4. русское название категории отдаётся фронтенду (`category_ru`).
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "backend")))

from app import embeddings as E          # noqa: E402
from app import labels_ru                # noqa: E402
from app.store import JsonStore          # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "demo_sfx")

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    # 1. словарь полный и без английских огрызков в тегах
    keys = labels_ru.all_keys()
    check("словарь >= 58 категорий", len(keys) >= 58, f"{len(keys)}")
    bad = [k for k in keys if not any("а" <= c <= "я" or c == "ё"
                                      for c in labels_ru.ru_name(k).lower())]
    check("у каждой категории русское название", not bad, str(bad[:5]))

    # 2. скан пишет русские теги
    tmp = tempfile.mkdtemp()
    index_path = os.path.join(tmp, "index.json")
    store = JsonStore(path=index_path, embedder=E.HeuristicEmbedder())
    store.scan_folder(DATA)
    check("индекс собран", len(store.items) > 0, f"{len(store.items)} файлов")
    it = next((x for x in store.items if x["category"] == "glass_breaking"), None)
    check("русские теги записаны", bool(it and "стекло" in " ".join(it["tags"])),
          str(it["tags"][:4] if it else None))
    check("description не пустой и не 'dog: '",
          bool(it and it["description"].strip().endswith((".", "я", "о")) or
               (it and ":" in it["description"] and len(it["description"]) > 12)),
          it["description"] if it else "")

    # 3. русский запрос находит класс
    for query, expected in [("разбилось стекло", "glass_breaking"),
                            ("тиканье часов", "clock_tick"),
                            ("гроза", "thunderstorm"),
                            ("пение птиц", "thunderstorm")][:3]:
        res = store.search_text(query, 5)["results"]
        cats = [r["file"]["category"] for r in res]
        check(f"запрос «{query}» -> {expected}", expected in cats, str(cats[:3]))

    # 4. ручная разметка переживает повторный скан
    target = store.items[0]
    store.annotate(target["id"], ["мой", "тег", "проверка"])
    store.scan_folder(DATA)
    tags_after = store.get(target["id"])["tags"]
    check("ручные теги живы после рескана",
          {"мой", "тег", "проверка"} <= set(tags_after), str(tags_after[:6]))

    # 5. category_ru отдаётся фронту
    shape = JsonStore.their_shape(store.items[0])
    check("their_shape отдаёт category_ru", bool(shape.get("category_ru")),
          str(shape.get("category_ru")))

    print()
    if failures:
        print(f"ПРОВАЛЕНО {len(failures)}: {failures}")
        return 1
    print("ВСЕ ТЕСТЫ РУССКИХ ТЕГОВ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
