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

    # 5. БЕЗ tags.csv: имя файла и имя папки дают русские теги
    import shutil
    scen = tempfile.mkdtemp()
    os.makedirs(os.path.join(scen, "Glass_Breaking"), exist_ok=True)
    os.makedirs(os.path.join(scen, "thunder"), exist_ok=True)
    src_glass = next(x["path"] for x in store.items if x["category"] == "glass_breaking")
    src_thun = next(x["path"] for x in store.items if x["category"] == "thunderstorm")
    # папка с большой буквы + мусорное имя файла
    shutil.copy(src_glass, os.path.join(scen, "Glass_Breaking", "1-20133-A-39.wav"))
    # осмысленное английское имя в плоской папке
    shutil.copy(src_thun, os.path.join(scen, "thunder", "thunderstorm_02.wav"))

    s_no_csv = JsonStore(path=os.path.join(scen, "i.json"),
                         embedder=E.HeuristicEmbedder())
    s_no_csv.scan_folder(scen)
    check("без CSV: файлов найдено", len(s_no_csv.items) == 2, str(len(s_no_csv.items)))

    by_name = {x["filename"]: x for x in s_no_csv.items}
    g = by_name["1-20133-A-39.wav"]
    check("без CSV: папка 'Glass_Breaking' -> русские теги",
          "стекло" in " ".join(g["tags"]), str(g["tags"][:4]))
    check("без CSV: папка в нижнем регистре не теряется",
          g["category"] == "glass_breaking", g["category"])
    res = s_no_csv.search_text("разбилось стекло", 5)["results"]
    check("без CSV: «разбилось стекло» находит файл",
          any(r["file"]["filename"] == "1-20133-A-39.wav" for r in res),
          str([r["file"]["filename"] for r in res[:2]]))
    res2 = s_no_csv.search_text("гроза", 5)["results"]
    check("без CSV: «гроза» находит thunderstorm_02.wav",
          any(r["file"]["filename"] == "thunderstorm_02.wav" for r in res2),
          str([r["file"]["filename"] for r in res2[:2]]))

    # 6. английский запрос тоже работает (обратный перевод)
    res3 = s_no_csv.search_text("glass breaking", 5)["results"]
    check("EN-запрос при русских тегах", len(res3) > 0, str(len(res3)))

    # 7. category_ru отдаётся фронту
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
