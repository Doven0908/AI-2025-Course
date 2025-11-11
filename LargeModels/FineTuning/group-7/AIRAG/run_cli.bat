@echo off
REM Quick launch CLI

REM Activate environment
if exist "ai_rag\Scripts\activate.bat" (
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start CLI
echo Starting CLI Interactive Mode...
echo.
python run.py --mode cli

pause

