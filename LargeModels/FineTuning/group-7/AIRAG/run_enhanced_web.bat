@echo off
REM 启动增强版 Web 界面
chcp 65001 > nul
echo.
echo ========================================
echo   启动增强版 RAG Web 界面
echo ========================================
echo.

REM 激活虚拟环境
if exist "ai_rag\Scripts\activate.bat" (
    echo [1/2] 激活 ai_rag 虚拟环境...
    call ai_rag\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [1/2] 激活 venv 虚拟环境...
    call venv\Scripts\activate.bat
) else (
    echo 错误: 未找到虚拟环境
    echo 请先运行 install_deps.bat 安装依赖
    pause
    exit /b 1
)

echo.
echo [2/2] 启动 Web 服务...
echo.
echo 提示: 
echo   - Web 界面地址: http://localhost:7860
echo   - 按 Ctrl+C 停止服务
echo.

python web_demo_enhanced.py

pause


