# PowerShell startup script
# Usage: .\run.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAG QA System - PowerShell Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
if (Test-Path "ai_rag\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (ai_rag)..." -ForegroundColor Yellow
    & "ai_rag\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (venv)..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "WARNING: No virtual environment found" -ForegroundColor Red
    Write-Host "Please create one: python -m venv ai_rag" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check dependencies
try {
    python -c "import langchain" 2>$null
    Write-Host "Dependencies OK" -ForegroundColor Green
} catch {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
}

# Create directories
$dirs = @("data\documents", "data\vectordb", "data\results", "data\logs")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Check .env
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Select mode:" -ForegroundColor Cyan
Write-Host "1. Web Interface (Recommended)" -ForegroundColor White
Write-Host "2. CLI Interactive" -ForegroundColor White
Write-Host "3. Index Documents" -ForegroundColor White
Write-Host "4. Run Evaluation" -ForegroundColor White
Write-Host "5. System Test" -ForegroundColor White
Write-Host "6. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter option (1-6)"

switch ($choice) {
    "1" {
        Write-Host "Starting Web Interface..." -ForegroundColor Green
        Write-Host "Open browser: http://localhost:7860" -ForegroundColor Yellow
        python run.py --mode web
    }
    "2" {
        Write-Host "Starting CLI..." -ForegroundColor Green
        python run.py --mode cli
    }
    "3" {
        Write-Host "Indexing documents..." -ForegroundColor Green
        python run.py --mode index
    }
    "4" {
        Write-Host "Running evaluation..." -ForegroundColor Green
        python run.py --mode eval
    }
    "5" {
        Write-Host "Running system test..." -ForegroundColor Green
        python test_system.py
    }
    "6" {
        Write-Host "Exit" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid option" -ForegroundColor Red
        exit 1
    }
}

Read-Host "`nPress Enter to exit"

