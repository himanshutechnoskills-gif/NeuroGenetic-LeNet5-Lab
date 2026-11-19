@echo off
call .venv\Scripts\activate.bat
python train.py --profile full
pause
