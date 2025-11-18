#!/bin/bash
# Setup script for RAGAS evaluation of Self-RAG experiments

echo "=========================================="
echo "RAGAS 评估设置脚本"
echo "RAGAS Evaluation Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "检查 Python 版本 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python >= 3.8
required_version="3.8"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python version is sufficient (>= 3.8)"
else
    echo "✗ Python version must be >= 3.8"
    exit 1
fi
echo ""

# Check for pip
echo "检查 pip Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 is available"
else
    echo "✗ pip3 not found, please install pip"
    exit 1
fi
echo ""

# Check if RAGAS is installed
echo "检查 RAGAS 安装 Checking RAGAS installation..."
if python3 -c "import ragas" 2>/dev/null; then
    echo "✓ RAGAS is already installed"
    ragas_installed=true
else
    echo "✗ RAGAS is not installed"
    ragas_installed=false
fi
echo ""

# Check for DeepSeek API key
echo "检查 DeepSeek API 密钥 Checking DeepSeek API key..."
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "✗ DEEPSEEK_API_KEY environment variable is not set"
    echo ""
    echo "请设置 API 密钥 Please set your API key:"
    echo "  export DEEPSEEK_API_KEY='your-api-key-here'"
    echo ""
    echo "获取密钥 Get API key from: https://platform.deepseek.com/"
    api_key_set=false
else
    echo "✓ DEEPSEEK_API_KEY is set"
    api_key_set=true
fi
echo ""

# Check required dependencies
echo "检查依赖包 Checking required packages..."
missing_packages=()

for package in openai datasets; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is not installed"
        missing_packages+=($package)
    fi
done
echo ""

# Offer to install missing packages
if [ "$ragas_installed" = false ] || [ ${#missing_packages[@]} -gt 0 ]; then
    echo "缺少依赖包 Missing dependencies detected"
    echo ""
    read -p "是否安装缺少的包? Install missing packages? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "正在安装依赖包 Installing dependencies..."

        # Install RAGAS
        if [ "$ragas_installed" = false ]; then
            echo "安装 RAGAS Installing RAGAS..."
            pip3 install ragas
        fi

        # Install other packages
        if [ ${#missing_packages[@]} -gt 0 ]; then
            echo "安装其他包 Installing other packages..."
            pip3 install "${missing_packages[@]}"
        fi

        echo ""
        echo "✓ 依赖包安装完成 Dependencies installed"
    else
        echo "跳过安装 Skipping installation"
        echo "您可以手动运行 You can manually run:"
        echo "  pip3 install ragas openai datasets"
    fi
fi
echo ""

# Check experiment output files
echo "检查实验输出文件 Checking experiment output files..."
exp_files=(
    "retrieval_lm/exp1"
    "retrieval_lm/exp2"
    "retrieval_lm/exp3_debug"
)

all_exp_files_exist=true
for exp_file in "${exp_files[@]}"; do
    if [ -f "$exp_file" ]; then
        echo "✓ $exp_file exists"
    else
        echo "✗ $exp_file not found"
        all_exp_files_exist=false
    fi
done
echo ""

# Check evaluation data files
echo "检查评估数据文件 Checking evaluation data files..."
eval_files=(
    "eval_data/popqa_longtail_w_gs.jsonl"
    "eval_data/arc_challenge_processed.jsonl"
    "eval_data/health_claims_processed.jsonl"
)

all_eval_files_exist=true
for eval_file in "${eval_files[@]}"; do
    if [ -f "$eval_file" ]; then
        echo "✓ $eval_file exists"
    else
        echo "✗ $eval_file not found"
        all_eval_files_exist=false
    fi
done
echo ""

# Check evaluation scripts
echo "检查评估脚本 Checking evaluation scripts..."
scripts=(
    "evaluate_with_ragas.py"
    "evaluate_with_ragas_simple.py"
)

all_scripts_exist=true
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "✓ $script exists"
        if [ -x "$script" ]; then
            echo "  (executable)"
        else
            echo "  (making executable...)"
            chmod +x "$script"
        fi
    else
        echo "✗ $script not found"
        all_scripts_exist=false
    fi
done
echo ""

# Summary
echo "=========================================="
echo "设置摘要 Setup Summary"
echo "=========================================="

if [ "$api_key_set" = true ] && \
   [ "$ragas_installed" = true ] && \
   [ ${#missing_packages[@]} -eq 0 ] && \
   [ "$all_exp_files_exist" = true ] && \
   [ "$all_eval_files_exist" = true ] && \
   [ "$all_scripts_exist" = true ]; then
    echo "✓ 所有检查通过 All checks passed!"
    echo ""
    echo "您可以开始评估 You can now start evaluation:"
    echo ""
    echo "  测试模式 Test mode (recommended):"
    echo "    python3 evaluate_with_ragas_simple.py"
    echo ""
    echo "  完整评估 Full evaluation:"
    echo "    python3 evaluate_with_ragas.py"
    echo ""
else
    echo "⚠ 部分检查未通过 Some checks failed"
    echo ""

    if [ "$api_key_set" = false ]; then
        echo "  - 设置 DeepSeek API 密钥 Set DeepSeek API key"
    fi

    if [ "$ragas_installed" = false ] || [ ${#missing_packages[@]} -gt 0 ]; then
        echo "  - 安装缺少的依赖包 Install missing dependencies"
    fi

    if [ "$all_exp_files_exist" = false ]; then
        echo "  - 确保实验输出文件存在 Ensure experiment output files exist"
    fi

    if [ "$all_eval_files_exist" = false ]; then
        echo "  - 确保评估数据文件存在 Ensure evaluation data files exist"
    fi

    if [ "$all_scripts_exist" = false ]; then
        echo "  - 确保评估脚本存在 Ensure evaluation scripts exist"
    fi

    echo ""
fi

echo "=========================================="
echo ""
echo "更多信息 For more information:"
echo "  - 快速入门 Quick start: QUICKSTART_RAGAS.md"
echo "  - 详细文档 Detailed docs: RAGAS_EVALUATION_README.md"
echo "  - 项目总结 Project summary: RAGAS_INTEGRATION_SUMMARY.md"
echo ""
