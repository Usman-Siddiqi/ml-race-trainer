@echo off
cd /d "C:\Users\usman\OneDrive - Queen's University\APSC 143\Practise Code\ml-race-trainer"
echo Testing Model Generalization...
echo.
echo This will test if your trained AI can work on different tracks.
echo It will test on both the original oval track and a new square track.
echo.
echo Make sure you have a trained model first!
echo.
python test_generalization.py --model best_race_model.pth --episodes 3
pause
