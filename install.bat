@echo off
chcp 65001 >nul
echo [1/3] Python-zavisimosti...
pip install -r backend\requirements.txt
echo [2/3] Demo-zvuki...
python scripts\generate_demo_sounds.py
echo [3/3] Node-zavisimosti...
cd frontend
npm install
cd ..
echo Gotovo! Zapusk: start-backend.bat i start-frontend.bat
pause
