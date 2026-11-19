@echo off
cd /d %~dp0backend
set SFX_USE_CLAP=1
uvicorn app.main:app --port 8000
