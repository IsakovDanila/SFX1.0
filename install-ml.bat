@echo off
chcp 65001 >nul
echo [1/2] torch (CPU, >=2.6 required by transformers)...
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo [2/2] laion-clap...
pip install -r backend\requirements-ml.txt
echo Gotovo! Zapusk CLAP-rezhima: start-backend-clap.bat
pause
