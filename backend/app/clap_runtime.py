"""Управление жизненным циклом CLAP для обычных пользователей.

Проблема, которую решает модуль: `CLAP_Module.load_ckpt()` качает ~1.9 ГБ
весов с HuggingFace **синхронно**, а `RobertaTokenizer` — ещё и `roberta-base`.
Если грузить это при старте бэкенда, сервер не отвечает несколько минут,
браузер показывает «демо-режим», а при обрыве сети происходит тихий откат
в эвристику без объяснений.

Здесь модель грузится в фоновом потоке, а состояние честно отдаётся
в `GET /api/health`:

    state: "disabled" | "loading" | "ready" | "failed"

и в UI появляется понятная плашка вместо «ничего не происходит».

Переменные окружения (все опциональны):
  SFX_USE_CLAP=1            включить CLAP-режим
  SFX_CLAP_CKPT=/path.pt    готовый файл весов (офлайн-установка)
  SFX_CLAP_OFFLINE=1        не пытаться качать; только локальные веса
  SFX_CLAP_MODEL_ID=1       0=630k-best, 1=630k-audioset-best (по умолчанию)
  SFX_CLAP_DEVICE=cpu|cuda:0 по умолчанию автоопределение
"""
from __future__ import annotations

import os
import threading
import time
import traceback

# Подсказки, которые попадают в UI и в лог. Пишем так, чтобы человек без
# опыта в ML понял, что делать дальше.
HINT_INSTALL = ("Поставьте ML-зависимости: ./install-ml.sh (Windows: install-ml.bat) "
                "или вручную: pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu, затем "
                "pip install -r backend/requirements-ml.txt")
HINT_WEIGHTS = ("Скачайте 630k-audioset-best.pt (~1.9 ГБ) на машине с интернетом: "
                "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt "
                "— и укажите путь: SFX_CLAP_CKPT=/путь/630k-audioset-best.pt")
HINT_TOKENIZER = ("Токенизатору нужен roberta-base с HuggingFace. Если машина офлайн: "
                  "скачайте модель на другой машине (huggingface-cli download "
                  "roberta-base --local-dir ./roberta-base) и задайте "
                  "TRANSFORMERS_OFFLINE=1 + HF_HOME=/путь/к/кэшу")


class ClapRuntime:
    """Синглтон: фоново грузит CLAP и отдаёт своё состояние."""

    def __init__(self) -> None:
        self.state = "disabled"       # disabled|loading|ready|failed
        self.error = ""               # техническая причина
        self.hint = ""                # что сделать пользователю
        self.step = ""                # текущий шаг («проверяю torch…»)
        self.started_at = 0.0
        self.finished_at = 0.0
        self.device = ""
        self.ckpt = ""
        self.model_name = os.getenv("SFX_CLAP_MODEL_ID", "1")
        self._embedder = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---------- состояние ----------
    @staticmethod
    def enabled() -> bool:
        return os.getenv("SFX_USE_CLAP") == "1"

    def status(self) -> dict:
        elapsed = 0.0
        if self.state == "loading" and self.started_at:
            elapsed = round(time.time() - self.started_at, 1)
        elif self.finished_at and self.started_at:
            elapsed = round(self.finished_at - self.started_at, 1)
        return {
            "state": self.state,
            "step": self.step,
            "error": self.error,
            "hint": self.hint,
            "device": self.device,
            "ckpt": self.ckpt,
            "model_id": self.model_name,
            "elapsed_s": elapsed,
            "ready": self.state == "ready",
        }

    def embedder(self):
        return self._embedder if self.state == "ready" else None

    # ---------- запуск ----------
    def start(self) -> None:
        """Запустить фоновую загрузку (идемпотентно)."""
        if not self.enabled():
            self.state = "disabled"
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self.state = "loading"
            self.step = "подготовка"
            self.error = self.hint = ""
            self.started_at = time.time()
            self.finished_at = 0.0
            self._thread = threading.Thread(target=self._load, name="clap-loader",
                                            daemon=True)
            self._thread.start()

    def _fail(self, error: str, hint: str) -> None:
        self.state = "failed"
        self.error = error
        self.hint = hint
        self.finished_at = time.time()
        print(f"[clap] ОШИБКА: {error}\n[clap] что делать: {hint}", flush=True)

    def _load(self) -> None:
        try:
            # 1. torch
            self.step = "проверяю torch"
            try:
                import torch  # noqa: F401
            except ImportError as e:
                self._fail(f"не установлен torch ({e})", HINT_INSTALL)
                return

            # 2. laion-clap
            self.step = "проверяю laion-clap"
            try:
                import laion_clap  # noqa: F401
            except Exception as e:  # noqa: BLE001
                self._fail(f"не установлен laion-clap ({e})", HINT_INSTALL)
                return

            # 3. веса: локальный файл, иначе скачивание
            ckpt_env = (os.getenv("SFX_CLAP_CKPT") or "").strip()
            offline = os.getenv("SFX_CLAP_OFFLINE") == "1"
            if ckpt_env and os.path.exists(ckpt_env):
                self.ckpt = ckpt_env
                self.step = f"загружаю локальные веса {os.path.basename(ckpt_env)}"
            elif ckpt_env:
                self._fail(f"файл весов не найден: {ckpt_env}", HINT_WEIGHTS)
                return
            elif offline:
                self._fail("включён SFX_CLAP_OFFLINE=1, но SFX_CLAP_CKPT не задан",
                           HINT_WEIGHTS)
                return
            else:
                self.step = ("скачиваю веса (~1.9 ГБ, только первый раз — "
                             "это может занять несколько минут)")

            # 4. сама модель
            from . import embeddings as E
            self._embedder = E.CLAPEmbedder(
                ckpt=self.ckpt or None,
                device=os.getenv("SFX_CLAP_DEVICE") or None,
                model_id=int(self.model_name),
            )
            self.device = getattr(self._embedder, "device", "") or self.device
            self.ckpt = self.ckpt or getattr(self._embedder, "ckpt", "")

            # 5. smoke-тест: модель должна уметь текст и аудио
            self.step = "проверяю модель на коротком примере"
            try:
                v = self._embedder.text_vector("a dog barking")
                if v is None or len(v) == 0:
                    raise RuntimeError("пустой вектор")
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"модель загрузилась, но не считает эмбеддинги: {e}")

            self.step = "готово"
            self.state = "ready"
            self.finished_at = time.time()
            print(f"[clap] модель готова за {self.status()['elapsed_s']} с, "
                  f"устройство: {self.device or 'auto'}", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = str(e)
            low = msg.lower()
            if any(w in low for w in ("connection", "download", "urlopen",
                                      "timed out", "resolve", "ssl", "http")):
                self._fail(f"не удалось скачать веса/токенизатор ({msg})", HINT_WEIGHTS)
            elif "roberta" in low or "tokenizer" in low:
                self._fail(f"не удалось загрузить токенизатор ({msg})", HINT_TOKENIZER)
            else:
                self._fail(msg, HINT_INSTALL)


RUNTIME = ClapRuntime()
