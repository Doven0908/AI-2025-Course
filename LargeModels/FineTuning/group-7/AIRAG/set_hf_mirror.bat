@echo off
REM 设置 HuggingFace 镜像环境变量
echo 正在配置 HuggingFace 镜像...

REM 设置环境变量（当前会话）
set HF_ENDPOINT=https://hf-mirror.com

REM 显示配置
echo.
echo ✅ HuggingFace 镜像已配置
echo HF_ENDPOINT=%HF_ENDPOINT%
echo.
echo 现在可以正常下载模型了！
echo.

pause



