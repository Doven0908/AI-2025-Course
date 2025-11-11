@echo off
REM Quick launch Web UI

REM Activate environment
if exist "ai_rag\Scripts\activate.bat" (
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start Web UI
echo Starting Web Interface...
echo Open browser: http://localhost:7860
echo.
python run.py --mode web

pause

