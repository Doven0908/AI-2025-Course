# ICL 项目

> 一个综合性的大语言模型（LLM）研究与评估工具集

## 📖 项目简介

本项目是一个集成了多个大语言模型研究工具的综合性仓库，主要聚焦于**上下文学习（In-Context Learning, ICL）**、**模型评估**和**交互式实验**。项目包含三个核心模块，涵盖了从学术基准测试到数学推理、再到交互式提示策略探索的完整研究流程。

## 🗂️ 项目结构

### 1. **algorithm/ai_homework** - AI 作业评估模块

基于 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 框架的模型评估工具集。

**主要功能：**
- 📊 **标准化评估框架**：支持 60+ 学术基准测试（MMLU、HumanEval 等）
- 🔧 **代码评估**：基于 HuggingFace 的 code_eval 度量标准，实现 pass@k 评估
- 🎯 **自定义任务**：包含本地化的 MMLU 和 HumanEval 任务配置
- 📈 **多样化指标**：内置 70+ 评估指标（准确率、BLEU、ROUGE、代码执行等）

**核心文件：**
- `lm-evaluation-harness/`: 完整的 EleutherAI 评估框架
- `code_eval/`: HumanEval 代码评估工具
- `my_tasks/`: 自定义任务配置（humaneval_local.yaml, mmlu_local.yaml）
- `results_*/`: 评估结果存档

### 2. **algorithm/math-inference** - 数学推理研究模块

专注于大语言模型数学推理能力的深度研究与改进。

**研究方向：**
- 🧠 **思维链（CoT）优化**：探索不同提示策略对推理步骤划分的影响
- 🎲 **蒙特卡洛树搜索（MCTS）**：用于数学问题求解的树搜索策略
- 📊 **置信度分析**：评估模型在不同推理阶段的确定性
- 🔄 **自我反思机制**：模型自主评估和改进推理过程

**实验脚本：**
- `test*.py`: 系列实验脚本，测试提示工程、置信度、准确率等
- `eval.py`: 主评估脚本
- `CoT/`, `MCTS/`: 推理策略实现
- `latex2sympy/`: LaTeX 数学表达式解析工具

### 3. **chatAI** - 提示策略实验平台

一个轻量级的浏览器端交互式工具，用于探索不同提示策略对 LLM 的影响。

**核心特性：**
- 🔌 **灵活的 API 集成**：支持 OpenAI 和 Qwen 兼容端点
- 📚 **MMLU 样本集**：精选的多学科基准测试题目
- 🎯 **五种提示策略**：
  - Base + 2-shot 示例
  - 指令跟随（Instruction-following）
  - 思维链（Chain-of-Thought）
  - 自我反思（Self-reflection）
  - CoT + 委员会投票（N=5）
- 📈 **详细遥测数据**：延迟、token 使用量、准确率、成本估算
- 📝 **实验笔记本**：记录定性观察与定量指标

**使用方法：**
```bash
cd chatAI
npx serve .
# 在浏览器中打开显示的 URL
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/heshan23/ICL.git
cd ICL

# 安装 AI 作业评估模块依赖
cd algorithm/ai_homework/lm-evaluation-harness
pip install -e .

# 或安装数学推理模块依赖
cd ../math-inference
pip install -r requirements.txt  # 如果存在
```

### 运行 MMLU 评估

```bash
cd algorithm/ai_homework
bash mmlu_local.sh
```

### 运行 HumanEval 代码评估

```bash
cd algorithm/ai_homework
bash qw3_8b_humaneval.sh
```

### 启动提示策略实验平台

```bash
cd chatAI
npx serve .
# 访问 http://localhost:3000
```

## 📊 已完成的实验

### HumanEval 评估结果
- **LLaMA3-8b-Instr**: 结果存储在 `results_humaneval_local_full/__home__lijiakun25__models__LLaMA3-8b-Instr/`
- **Qwen3-8b**: 结果存储在 `results_humaneval_local_full/__home__lijiakun25__models__Qwen3-8b/`

### 数学推理实验
- **Test 2-7**: 探索不同提示词对思维步骤划分的影响
- **Confidence 分析**: 在不同推理阶段的置信度评估
- **Few-shot 学习**: 强模型指导弱模型的输出划分
- **准确率对比**: 切分后续写与原始输出的性能比较

## 🛠️ 技术栈

- **评估框架**: lm-evaluation-harness (EleutherAI)
- **模型支持**: HuggingFace Transformers, vLLM, OpenAI API
- **前端**: 原生 HTML/CSS/JavaScript
- **数学工具**: LaTeX 解析、符号计算
- **评估指标**: HuggingFace Evaluate 库

## 📝 研究思路（摘自 math-inference）

当前探索的核心问题：
1. **节点价值评估**: 如何平衡 info_entropy、value 和 confidence？
2. **树结构存储**: 是实时计算还是预计算 entropy？
3. **探索策略**: 如何在 token 效率、信息熵和准确率之间权衡？

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！项目仍在积极开发中。

## 📄 许可证

请参考各子模块的许可证：
- lm-evaluation-harness: MIT License
- 其他模块：待补充

## 📧 联系方式

- GitHub: [@heshan23](https://github.com/heshan23)
- 仓库: [ICL](https://github.com/heshan23/ICL)

---

**注**: 本项目用于学术研究目的，部分模块可能需要额外的 API 密钥或计算资源。
