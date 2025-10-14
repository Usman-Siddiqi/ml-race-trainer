@echo off
cd /d "C:\Users\usman\OneDrive - Queen's University\APSC 143\Practise Code\ml-race-trainer"
echo Starting Improved Visual Training...
echo.
echo This version fixes the "robot not moving" issue by:
echo - Removing "no action" from action space
echo - Increasing exploration (epsilon_min = 0.1)
echo - Better reward structure to encourage movement
echo - Penalties for standing still
echo.
python improved_visual_training.py --episodes 1000 --render-interval 10
pause
