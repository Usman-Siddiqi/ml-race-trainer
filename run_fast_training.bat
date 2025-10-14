@echo off
cd /d "C:\Users\usman\OneDrive - Queen's University\APSC 143\Practise Code\ml-race-trainer"
echo Starting Fast Training (5000 episodes, no visualization)...
echo.
echo This will train very fast without any visualization.
echo Perfect for getting a good model quickly.
echo.
python train_model.py --mode train --episodes 5000
pause
