#!/usr/bin/env bash
# Установка CLAP-режима для обычного пользователя (Mac/Linux).
#
# Что делает:
#   1. проверяет железо (RAM/диск) и предупреждает, если заведомо мало;
#   2. ставит CPU-torch отдельно, чтобы pip не притащил ~2.5 ГБ CUDA-сборки;
#   3. ставит laion-clap;
#   4. заранее скачивает веса (~1.9 ГБ), чтобы первый запуск не «висел».
#
# Флаги:
#   --no-weights      не скачивать веса сейчас (скачаются при первом старте)
#   --weights-only    только скачать веса (например, на машине с интернетом)
#   --gpu             поставить CUDA-сборку torch (нужен NVIDIA GPU)
set -u

WEIGHTS_ONLY=0
FETCH_WEIGHTS=1
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

for arg in "$@"; do
  case "$arg" in
    --no-weights)   FETCH_WEIGHTS=0 ;;
    --weights-only) WEIGHTS_ONLY=1 ;;
    --gpu)          TORCH_INDEX="" ;;
    *) echo "Неизвестный флаг: $arg"; exit 2 ;;
  esac
done

PY=${PYTHON:-python3}
CKPT_URL="https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
CKPT_DIR="${SFX_CLAP_DIR:-$HOME/.cache/sfx-clap}"
CKPT="$CKPT_DIR/630k-audioset-best.pt"

# ---------- 1. проверка железа ----------
echo "== Проверяю железо =="
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
else
  RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 ))
fi
FREE_GB=$(df -Pk . | awk 'NR==2 {print int($4/1048576)}')
echo "   RAM: ${RAM_GB} ГБ, свободно на диске: ${FREE_GB} ГБ"
if [ "${RAM_GB:-0}" -lt 4 ]; then
  echo "   ⚠ Меньше 4 ГБ RAM: CLAP может не уложиться в память."
  echo "     Варианты: demo-режим (без CLAP) или машина помощнее."
fi
if [ "${FREE_GB:-0}" -lt 3 ] && [ "$FETCH_WEIGHTS" = "1" ]; then
  echo "   ⚠ Меньше 3 ГБ на диске: torch (~200 МБ) + веса (~1.9 ГБ) не влезут."
  echo "     Освободите место или запустите с --no-weights."
fi

if [ "$WEIGHTS_ONLY" = "1" ]; then
  FETCH_WEIGHTS=1
else
  # ---------- 2. torch ----------
  echo "== [1/3] torch =="
  if [ -n "$TORCH_INDEX" ]; then
    $PY -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX" || exit 1
  else
    echo "   CUDA-сборка (нужен NVIDIA GPU и драйвер)"
    $PY -m pip install --upgrade torch torchvision || exit 1
  fi

  # ---------- 3. laion-clap ----------
  echo "== [2/3] laion-clap =="
  $PY -m pip install -r "$(dirname "$0")/backend/requirements-ml.txt" || exit 1
fi

# ---------- 4. веса ----------
if [ "$FETCH_WEIGHTS" = "1" ]; then
  echo "== [3/3] веса CLAP (~1.9 ГБ, один раз) =="
  mkdir -p "$CKPT_DIR"
  if [ -f "$CKPT" ]; then
    echo "   уже скачаны: $CKPT"
  else
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --progress-bar -o "$CKPT.part" "$CKPT_URL" \
        && mv "$CKPT.part" "$CKPT"
    else
      $PY - "$CKPT_URL" "$CKPT" <<'PYEOF'
import sys, urllib.request, os
url, dst = sys.argv[1], sys.argv[2]
tmp = dst + ".part"
def hook(n, bs, total):
    if total > 0:
        pct = min(100, n * bs * 100 // total)
        print(f"\r   {pct:3d}%  {n*bs/1048576:6.1f} / {total/1048576:6.1f} МБ",
              end="", flush=True)
urllib.request.urlretrieve(url, tmp, hook)
print()
os.replace(tmp, dst)
print(f"   сохранено: {dst}")
PYEOF
    fi
  fi
  if [ -f "$CKPT" ]; then
    echo "   Проверка целостности: $(du -h "$CKPT" | cut -f1)"
    echo "   Веса на месте. Укажите путь при запуске:"
    echo "     SFX_CLAP_CKPT=$CKPT ./start-backend-clap.sh"
  fi
else
  echo "   Веса не скачивались (--no-weights). При первом старте они"
  echo "   загрузятся автоматически, но сервер будет «прогреваться»."
fi

if [ "$WEIGHTS_ONLY" = "0" ]; then
  echo
  echo "Готово. Запуск: ./start-backend-clap.sh"
  echo "Если будете работать без интернета, добавьте в запуск:"
  echo "  SFX_CLAP_CKPT=$CKPT SFX_CLAP_OFFLINE=1 ./start-backend-clap.sh"
fi
