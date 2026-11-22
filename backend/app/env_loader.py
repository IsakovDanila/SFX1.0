"""Загрузка .env без внешних зависимостей.

Зачем: токен HF и прочие настройки удобно держать в файле, а не вводить
каждый раз в терминале. Но НЕ в коде — иначе он попадёт в git.

Порядок поиска: ./.env, ../.env (от каталога backend), ~/.config/sfx/.env
Уже заданные переменные окружения НЕ перезаписываются (приоритет у ОС).
"""
from __future__ import annotations

import os


def _candidates() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))          # backend/app
    root = os.path.abspath(os.path.join(here, "..", ".."))    # корень репо
    out = [os.path.join(root, ".env"),
           os.path.join(os.path.dirname(here), ".env"),       # backend/.env
           os.path.expanduser("~/.config/sfx/.env")]
    return out


def load_env(path: str | None = None) -> str:
    """Прочитать .env и положить значения в os.environ. Возвращает путь."""
    files = [path] if path else _candidates()
    for f in files:
        if not f or not os.path.isfile(f):
            continue
        try:
            with open(f, encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            print(f"[env] настройки загружены из {f}")
            return f
        except OSError as e:
            print(f"[env] не смог прочитать {f}: {e}")
            continue
    return ""
