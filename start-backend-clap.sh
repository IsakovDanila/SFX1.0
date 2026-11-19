#!/usr/bin/env bash
# CLAP-режим. Модель грузится в фоне: сервер отвечает сразу, статус — на
# /api/clap/status и в плашке UI. Если веса уже скачаны install-ml.sh,
# путь подхватывается автоматически (SFX_CLAP_DIR / ~/.cache/sfx-clap).
cd "$(dirname "$0")/backend"

CKPT_DIR="${SFX_CLAP_DIR:-$HOME/.cache/sfx-clap}"
if [ -z "${SFX_CLAP_CKPT:-}" ] && [ -f "$CKPT_DIR/630k-audioset-best.pt" ]; then
  export SFX_CLAP_CKPT="$CKPT_DIR/630k-audioset-best.pt"
  echo "[clap] локальные веса: $SFX_CLAP_CKPT"
fi

export SFX_USE_CLAP=1
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
