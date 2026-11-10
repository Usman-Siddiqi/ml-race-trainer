@echo off
rem Run any Race Lab command without setting up a Python path first, e.g.
rem   .\race watch --model runs\v3-default\best.pt
rem Run from this folder: relative paths such as runs\... are resolved here.
setlocal
set "RACE_PYTHON=%LOCALAPPDATA%\RaceTrainer\venv\Scripts\python.exe"
if not exist "%RACE_PYTHON%" set "RACE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%RACE_PYTHON%" (
    echo Run powershell -ExecutionPolicy Bypass -File setup.ps1 -Device cuda first.
    exit /b 1
)
"%RACE_PYTHON%" -m race_trainer %*
exit /b %errorlevel%
