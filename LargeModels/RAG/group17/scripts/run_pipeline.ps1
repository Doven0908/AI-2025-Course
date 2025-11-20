param(
    [switch]$Ingest = $false,
    [switch]$Serve = $false,
    [string]$BindHost = "10.192.134.86",
    [int]$BindPort = 8000,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

function ResolvePython {
    param(
        [string]$PreferredPath
    )

    if ($PreferredPath) {
        if (Test-Path $PreferredPath) {
            return $PreferredPath
        }
        throw "指定的 PythonPath 不存在：$PreferredPath"
    }

    if ($env:RAGQA_PYTHON) {
        if (Test-Path $env:RAGQA_PYTHON) {
            return $env:RAGQA_PYTHON
        }
        Write-Warning "环境变量 RAGQA_PYTHON 指向的路径不存在：$($env:RAGQA_PYTHON)"
    }

    if ($env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    if ($env:PYTHONHOME) {
        $candidate = Join-Path $env:PYTHONHOME "python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return "python"
}

$PythonExe = ResolvePython -PreferredPath $PythonPath

if ($Ingest) {
    & $PythonExe -m app.cli ingest
}

if ($Serve) {
    & $PythonExe -m uvicorn app.server:app --host $BindHost --port $BindPort
}
