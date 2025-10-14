@echo off
cd /d "C:\Users\usman\OneDrive - Queen's University\APSC 143\Practise Code\ml-race-trainer"
echo Starting Long Training (2000 episodes)...
echo.
echo This will take longer but should produce much better results.
echo The AI will have more time to learn complex strategies.
echo.
python improved_visual_training.py --episodes 2000 --render-interval 20
pause
