@echo off
REM Install dependencies script

echo ========================================
echo   Installing Dependencies
echo ========================================
echo.

REM Activate environment
if exist "ai_rag\Scripts\activate.bat" (
    echo Activating ai_rag environment...
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating venv environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: No virtual environment found
    echo Please create one first: python -m venv ai_rag
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip -q

echo.
echo Installing dependencies (this may take 5-10 minutes)...
echo Using Tsinghua mirror for faster download...
echo.

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install
    echo Trying with alternative mirror...
    echo.
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    
    if errorlevel 1 (
        echo.
        echo ERROR: Installation failed
        echo.
        echo Troubleshooting:
        echo 1. Check internet connection
        echo 2. Try manual installation: pip install langchain faiss-cpu gradio
        echo 3. Check error messages above
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Verifying installation...
python -c "import langchain; import loguru; import gradio; print('All core packages installed successfully!')"

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may not be installed correctly
    echo Please check the error messages above
) else (
    echo.
    echo SUCCESS: All dependencies installed!
    echo.
    echo Next steps:
    echo 1. Run: python test_system.py (to verify)
    echo 2. Run: run_web.bat (to start Web UI)
    echo 3. Or: python run.py --mode web
)

echo.
pause

