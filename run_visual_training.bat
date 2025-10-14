@echo off
cd /d "C:\Users\usman\OneDrive - Queen's University\APSC 143\Practise Code\ml-race-trainer"
echo Starting Visual Training...
echo.
echo You can watch the AI learn in real-time!
echo Close the pygame window to stop training early.
echo.
python visual_training.py --episodes 100 --render-interval 5
pause
