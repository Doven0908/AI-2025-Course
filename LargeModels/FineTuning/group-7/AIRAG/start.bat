@echo off
REM Ultra Simple Startup Script - Just Works!

echo Starting RAG System...
echo.

REM Set HuggingFace Mirror
set HF_ENDPOINT=https://hf-mirror.com

REM Activate Environment
if exist "ai_rag\Scripts\activate.bat" (
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start Application
python web_demo_enhanced.py

pause
