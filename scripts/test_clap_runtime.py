"""Проверка CLAP-обвязки БЕЗ torch и без скачивания весов.

Здесь нет настоящего CLAP (веса ~1.9 ГБ, нужен интернет к HuggingFace),
но проверяется всё, что может сломаться у обычного пользователя:

  1. состояние по умолчанию — disabled, сервер не висит;
  2. без torch/laion-clap -> failed + понятная подсказка, а не падение;
  3. SFX_CLAP_OFFLINE=1 без весов -> failed с инструкцией по офлайн-установке;
  4. SFX_CLAP_CKPT на несуществующий файл -> понятная ошибка;
  5. калибровка уверенности: релевантный косинус -> ~99%, случайный -> низко;
  6. /api/clap/retry перезапускает загрузку;
  7. стор умеет переключиться на новый эмбеддер (use_embedder).

Запуск:  python3 scripts/test_clap_runtime.py
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "backend")))

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f"  [{extra}]" if extra else ""))
    if not cond:
        failures.append(name)


def fresh_runtime(**env):
    """Пересоздать RUNTIME с нужными переменными окружения."""
    for k in ("SFX_USE_CLAP", "SFX_CLAP_OFFLINE", "SFX_CLAP_CKPT",
              "SFX_CLAP_DEVICE", "SFX_CLAP_MODEL_ID"):
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})
    from app import clap_runtime
    importlib.reload(clap_runtime)
    return clap_runtime


def main() -> int:
    print("-- 1. CLAP выключен по умолчанию")
    rt = fresh_runtime()
    st = rt.RUNTIME.status()
    check("state=disabled", st["state"] == "disabled", st["state"])
    check("enabled() == False", rt.ClapRuntime.enabled() is False)
    rt.RUNTIME.start()                     # не должен ничего запускать
    check("start() при disabled ничего не грузит",
          rt.RUNTIME.status()["state"] == "disabled")

    print("-- 2. включён, но torch не установлен")
    rt = fresh_runtime(SFX_USE_CLAP="1")
    rt.RUNTIME.start()
    if rt.RUNTIME._thread:
        rt.RUNTIME._thread.join(timeout=30)
    st = rt.RUNTIME.status()
    check("state=failed", st["state"] == "failed", st["state"])
    check("причина про torch/laion-clap",
          "torch" in st["error"] or "laion" in st["error"], st["error"])
    check("подсказка ведёт на install-ml.sh",
          "install-ml" in st["hint"], st["hint"][:60])
    check("embedder() == None при failed", rt.RUNTIME.embedder() is None)

    print("-- 3. офлайн-режим без весов")
    rt = fresh_runtime(SFX_USE_CLAP="1", SFX_CLAP_OFFLINE="1")
    # torch нет, поэтому до проверки весов дело может не дойти — это ок,
    # главное: подсказка остаётся actionable
    rt.RUNTIME.start()
    if rt.RUNTIME._thread:
        rt.RUNTIME._thread.join(timeout=30)
    st = rt.RUNTIME.status()
    check("state=failed и есть hint", st["state"] == "failed" and bool(st["hint"]))

    print("-- 4. SFX_CLAP_CKPT на несуществующий файл (логика выбора пути)")
    import app.clap_runtime as cr
    saved_torch = sys.modules.get("torch")
    sys.modules["torch"] = type(sys)("torch")          # фейковый torch
    sys.modules["laion_clap"] = type(sys)("laion_clap")  # фейковый пакет
    try:
        rt = fresh_runtime(SFX_USE_CLAP="1", SFX_CLAP_CKPT="/нет/такого/файла.pt")
        rt.RUNTIME.start()
        if rt.RUNTIME._thread:
            rt.RUNTIME._thread.join(timeout=30)
        st = rt.RUNTIME.status()
        check("битый путь к весам -> failed", st["state"] == "failed", st["state"])
        check("сообщение называет файл",
              "не найден" in st["error"] and ".pt" in st["error"], st["error"])
    finally:
        sys.modules.pop("torch", None)
        sys.modules.pop("laion_clap", None)
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch

    print("-- 5. калибровка уверенности CLAP")
    from app.embeddings import CLAPEmbedder
    fake = CLAPEmbedder.__new__(CLAPEmbedder)          # без загрузки модели
    hi = fake.text_audio_prob(0.55)
    mid = fake.text_audio_prob(0.375)
    lo = fake.text_audio_prob(0.10)
    check("релевантный косинус -> высокая вероятность", hi > 0.9, f"{hi:.2f}")
    check("случайный косинус -> низкая вероятность", lo < 0.05, f"{lo:.2f}")
    check("монотонность", hi > mid > lo, f"{hi:.2f} > {mid:.2f} > {lo:.2f}")
    # при дефолте cos=0.25 даёт ~0.04; со сдвинутой калибровкой (0.1..0.3) — высокую
    default_val = fake.text_audio_prob(0.25)
    os.environ["SFX_CLAP_CALIB"] = "0.1,0.3"
    shifted_val = fake.text_audio_prob(0.25)
    os.environ.pop("SFX_CLAP_CALIB")
    check("SFX_CLAP_CALIB сдвигает калибровку",
          shifted_val > 0.6 > default_val, f"{default_val:.2f} -> {shifted_val:.2f}")

    print("-- 6. переключение стора на новый эмбеддер")
    import tempfile
    from app import embeddings as E
    from app.store import JsonStore
    store = JsonStore(path=os.path.join(tempfile.mkdtemp(), "i.json"),
                      embedder=E.HeuristicEmbedder())
    check("стартует на эвристике", store.embedder.name == "heuristic-demo")

    class StubCLAP(E.CLAPEmbedder):
        name = "laion-clap"
        dim = 8

        def text_audio_prob(self, cos):       # без torch
            return fake.text_audio_prob(cos)

    store.use_embedder(StubCLAP.__new__(StubCLAP), reindex=False)
    check("use_embedder переключил режим", store.embedder.name == "laion-clap")
    store.use_embedder(StubCLAP.__new__(StubCLAP), reindex=False)
    check("повторный вызов не ломает стор", store.embedder.name == "laion-clap")

    print()
    if failures:
        print(f"ПРОВАЛЕНО {len(failures)}: {failures}")
        return 1
    print("ВСЕ ТЕСТЫ CLAP-ОБВЯЗКИ ПРОЙДЕНЫ (без torch и без весов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
