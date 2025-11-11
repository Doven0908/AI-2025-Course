@echo off
REM Simple startup script without encoding issues

echo ========================================
echo   RAG QA System - Quick Start
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Activate virtual environment
if exist "ai_rag\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found
    echo Please create one: python -m venv ai_rag
    pause
    exit /b 1
)

REM Check if dependencies installed
python -c "import langchain" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

REM Create directories
if not exist "data\documents" mkdir data\documents
if not exist "data\vectordb" mkdir data\vectordb
if not exist "data\results" mkdir data\results
if not exist "data\logs" mkdir data\logs

REM Check .env
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env >nul 2>&1
)

echo.
echo Select mode:
echo 1. Web Interface (Recommended)
echo 2. CLI Interactive
echo 3. Index Documents
echo 4. Run Evaluation
echo 5. Exit
echo.
set /p choice=Enter option (1-5): 

if "%choice%"=="1" (
    echo Starting Web Interface...
    python run.py --mode web
) else if "%choice%"=="2" (
    echo Starting CLI...
    python run.py --mode cli
) else if "%choice%"=="3" (
    echo Indexing documents...
    python run.py --mode index
) else if "%choice%"=="4" (
    echo Running evaluation...
    python run.py --mode eval
) else if "%choice%"=="5" (
    echo Exit
    exit /b 0
) else (
    echo Invalid option
    pause
    exit /b 1
)

pause

