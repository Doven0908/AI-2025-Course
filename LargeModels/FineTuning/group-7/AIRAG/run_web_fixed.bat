@echo off
chcp 65001 > nul

echo.
echo ========================================
echo   Starting Enhanced RAG Web Interface
echo ========================================
echo.

REM Set HuggingFace Mirror
echo [1/3] Configuring HuggingFace Mirror...
set HF_ENDPOINT=https://hf-mirror.com
echo Mirror: %HF_ENDPOINT%

REM Activate Virtual Environment
if exist "ai_rag\Scripts\activate.bat" (
    echo [2/3] Activating ai_rag environment...
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [2/3] Activating venv environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found
    echo Please run install_deps.bat first
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Web Service...
echo.
echo Configuration Complete!
echo.
echo Web Interface: http://localhost:7861
echo HuggingFace Mirror: Configured
echo Xiaohang API: Configured
echo Press Ctrl+C to stop
echo.
echo ========================================
echo.

python web_demo_enhanced.py

pause
