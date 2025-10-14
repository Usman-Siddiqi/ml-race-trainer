@echo off
cd /d "C:\Users\usman\OneDrive - Queen's University\APSC 143\Practise Code\ml-race-trainer"
echo Current directory: %CD%
echo.
echo Files in directory:
dir *.py
echo.
echo Running quick test...
python quick_test.py
pause
