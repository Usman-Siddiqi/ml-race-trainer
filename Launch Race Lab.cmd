@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run uv sync --extra cuda in this folder first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m race_trainer app
if errorlevel 1 pause
