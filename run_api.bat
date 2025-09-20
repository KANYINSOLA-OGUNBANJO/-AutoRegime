@echo off
REM ===== AutoRegime API launcher =====
cd /d C:\Users\Ogunbanjo\Desktop\AutoRegime

REM --- Activate your conda env (adjust the path if Anaconda lives elsewhere)
call C:\Users\Ogunbanjo\anaconda3\Scripts\activate.bat autoregime_env

REM --- Start the FastAPI server (change port if 8000 is busy)
python -m uvicorn autoregime.api_server:app --host 127.0.0.1 --port 8000 --reload

REM Keep the window open if something errors
pauseost 127.0.0.1 --port 8000 --reload