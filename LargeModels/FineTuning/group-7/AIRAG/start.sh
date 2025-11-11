#!/bin/bash

echo "========================================"
echo "  RAG问答系统 - Linux/Mac启动脚本"
echo "========================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[提示] 未找到虚拟环境，正在创建..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[错误] 创建虚拟环境失败"
        exit 1
    fi
    echo "[成功] 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "[提示] 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
if [ ! -d "venv/lib/python*/site-packages/langchain" ]; then
    echo "[提示] 正在安装依赖包..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
    echo "[成功] 依赖安装完成"
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "[提示] 未找到.env文件，正在创建默认配置..."
    cp .env.example .env
    echo "[注意] 请编辑.env文件配置API密钥等参数"
fi

# 创建必要目录
mkdir -p data/documents data/vectordb data/results data/logs

# 显示菜单
echo ""
echo "请选择运行模式:"
echo "1. Web界面 (推荐)"
echo "2. CLI命令行"
echo "3. 索引文档"
echo "4. 运行评估"
echo "5. 退出"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "[启动] Web界面..."
        python run.py --mode web
        ;;
    2)
        echo "[启动] CLI命令行..."
        python run.py --mode cli
        ;;
    3)
        echo "[启动] 索引文档..."
        python run.py --mode index
        ;;
    4)
        echo "[启动] 运行评估..."
        python run.py --mode eval
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "[错误] 无效选项"
        exit 1
        ;;
esac

