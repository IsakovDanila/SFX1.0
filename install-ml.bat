@echo off
chcp 65001 >nul
setlocal

REM Установка CLAP-режима для обычного пользователя (Windows).
REM   install-ml.bat              — поставить всё и скачать веса (~1.9 ГБ)
REM   install-ml.bat --no-weights — без весов (скачаются при первом старте)
REM   install-ml.bat --weights-only — только веса (на машине с интернетом)
REM   install-ml.bat --gpu        — CUDA-сборка torch (нужен NVIDIA GPU)

set FETCH_WEIGHTS=1
set WEIGHTS_ONLY=0
set TORCH_ARGS=--index-url https://download.pytorch.org/whl/cpu

for %%a in (%*) do (
  if "%%a"=="--no-weights"   set FETCH_WEIGHTS=0
  if "%%a"=="--weights-only" set WEIGHTS_ONLY=1
  if "%%a"=="--gpu"          set TORCH_ARGS=
)

set CKPT_URL=https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt
if "%SFX_CLAP_DIR%"=="" set SFX_CLAP_DIR=%USERPROFILE%\.cache\sfx-clap
set CKPT=%SFX_CLAP_DIR%\630k-audioset-best.pt

echo == Проверяю железо ==
for /f "skip=1 tokens=*" %%m in ('wmic ComputerSystem get TotalPhysicalMemory 2^>nul') do (
  if not "%%m"=="" set RAM_BYTES=%%m
)
if defined RAM_BYTES (
  set /a RAM_GB=%RAM_BYTES:~0,-9% / 1000000
  echo    RAM: около %RAM_GB% ГБ
  if %RAM_GB% LSS 4 echo    [!] Меньше 4 ГБ RAM: CLAP может не уложиться в память.
)

if "%WEIGHTS_ONLY%"=="1" goto weights

echo == [1/3] torch ==
python -m pip install --upgrade torch torchvision %TORCH_ARGS%
if errorlevel 1 goto fail

echo == [2/3] laion-clap ==
python -m pip install -r "%~dp0backend\requirements-ml.txt"
if errorlevel 1 goto fail

:weights
if "%FETCH_WEIGHTS%"=="0" (
  echo == Веса не скачиваются ^(--no-weights^). Загрузятся при первом старте. ==
  goto done
)

echo == [3/3] веса CLAP ^(~1.9 ГБ, один раз^) ==
if not exist "%SFX_CLAP_DIR%" mkdir "%SFX_CLAP_DIR%"
if exist "%CKPT%" (
  echo    уже скачаны: %CKPT%
  goto weights_ok
)
echo    качаю, это может занять несколько минут...
curl -L --fail --progress-bar -o "%CKPT%.part" "%CKPT_URL%"
if errorlevel 1 (
  echo    [!] curl не сработал. Скачайте вручную:
  echo        %CKPT_URL%
  echo        и положите файл сюда: %CKPT%
  goto fail
)
move /y "%CKPT%.part" "%CKPT%" >nul

:weights_ok
echo    Веса на месте: %CKPT%

:done
echo.
echo Готово. Запуск: start-backend-clap.bat
echo Офлайн-запуск: set SFX_CLAP_CKPT=%CKPT% ^&^& set SFX_CLAP_OFFLINE=1 ^&^& start-backend-clap.bat
pause
exit /b 0

:fail
echo.
echo [X] Установка не завершилась. Что можно сделать:
echo     1) Проверьте интернет и повторите: install-ml.bat
echo     2) Без интернета на этой машине: запустите на машине с интернетом
echo        install-ml.bat --weights-only, скопируйте папку %SFX_CLAP_DIR% сюда
echo        и запускайте с SFX_CLAP_CKPT + SFX_CLAP_OFFLINE=1
echo     3) Если не хватает памяти/диска — работайте в demo-режиме
echo        (start-backend.bat), поиск по словам и русским тегам работает.
pause
exit /b 1
