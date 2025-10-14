@echo off
echo Testing ML Race Trainer...
echo.

echo Testing race environment import...
python -c "from race_environment import RaceCarEnv; print('SUCCESS: Race environment imported')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to import race environment
    pause
    exit /b 1
)

echo.
echo Testing simple main menu...
python simple_main.py
if %errorlevel% neq 0 (
    echo ERROR: Simple main menu failed
    pause
    exit /b 1
)

echo.
echo Test completed successfully!
pause
