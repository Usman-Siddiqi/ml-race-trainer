@echo off
cd /d "%~dp0"
set "RACE_PYTHON=%LOCALAPPDATA%\RaceTrainer\venv\Scripts\python.exe"
if not exist "%RACE_PYTHON%" set "RACE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%RACE_PYTHON%" (
    echo Run powershell -ExecutionPolicy Bypass -File setup.ps1 -Device cuda first.
    pause
    exit /b 1
)
"%RACE_PYTHON%" -m race_trainer app
if errorlevel 1 pause
