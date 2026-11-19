@echo off
chcp 65001 >nul
cd /d %~dp0backend

REM Подхватываем веса, если их поставил install-ml.bat
if "%SFX_CLAP_DIR%"=="" set SFX_CLAP_DIR=%USERPROFILE%\.cache\sfx-clap
if not defined SFX_CLAP_CKPT (
  if exist "%SFX_CLAP_DIR%\630k-audioset-best.pt" set SFX_CLAP_CKPT=%SFX_CLAP_DIR%\630k-audioset-best.pt
)
if defined SFX_CLAP_CKPT echo [clap] локальные веса: %SFX_CLAP_CKPT%

set SFX_USE_CLAP=1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
