#!/bin/bash

# 提交脚本 - 将项目提交到 AI-2025-Course 仓库
# 使用方法：
# 1. 确保已经 Fork 了 https://github.com/BUAAZhangHaonan/AI-2025-Course 仓库
# 2. 添加 fork 仓库为远程仓库（如果还没有）
# 3. 运行此脚本

set -e

echo "🚀 开始提交流程..."

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 检查 git 状态
echo "📋 检查 Git 状态..."
git status

# 创建分支（如果不存在）
BRANCH_NAME="group-16"
if git show-ref --verify --quiet refs/heads/$BRANCH_NAME; then
    echo "✅ 分支 $BRANCH_NAME 已存在，切换到该分支"
    git checkout $BRANCH_NAME
else
    echo "📝 创建新分支 $BRANCH_NAME"
    git checkout -b $BRANCH_NAME
fi

# 添加文件
echo "📦 添加文件到暂存区..."
git add LargeModels/RAG/group-16/
git add .gitignore

# 检查是否有变更
if git diff --staged --quiet; then
    echo "⚠️  没有需要提交的变更"
    exit 0
fi

# 提交
echo "💾 提交变更..."
git commit -m "feat: 提交 group-16 RAG 项目

- 添加航小厨智能 RAG 烹饪助手
- 包含完整的代码、文档和评估结果
- 支持向量检索、重排和智能 Agent"

echo "✅ 提交完成！"
echo ""
echo "📤 下一步操作："
echo "1. 如果还没有添加 fork 仓库为远程仓库，请运行："
echo "   git remote add fork https://github.com/YOUR_USERNAME/AI-2025-Course.git"
echo ""
echo "2. 推送到 fork 仓库："
echo "   git push fork $BRANCH_NAME"
echo ""
echo "3. 在 GitHub 上发起 Pull Request 到原仓库的 master 分支"

