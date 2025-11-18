# ⚡ 快速开始

面向第一次接触仓库的同学，按以下步骤完成环境初始化与验证。

## 1. 克隆与环境

```bash
git clone <repo-url> ai4hallucination-main
cd ai4hallucination-main
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 2. 安装核心依赖

```bash
pip install -r modules/selfrag-official/requirements.txt
pip install -e modules/selfcheckgpt
pip install ragas lettucedetect vllm gradio
```

> 如需使用国内镜像，可设置 `HF_ENDPOINT=https://hf-mirror.com` 或公司内部代理。

## 3. 运行冒烟测试

```bash
# 1) SelfCheckGPT demo
python modules/selfcheckgpt/demo.py

# 2) Comprehensive 小规模评测
python modules/comprehensive/main.py --help  # 确认可运行

# 3) SELF-RAG Lite ASQA 示例
python modules/selfrag-lite/run_long_form_static.py --help
```

出现帮助信息即代表依赖安装正确。若需完整实验，可使用 `scripts/` 提供的封装命令。

## 4. 配置 API / 模型

在根目录创建 `.env`（或使用系统环境变量）：

```
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
HF_ENDPOINT=https://hf-mirror.com
LLM_CACHE=./model_cache
```

> 建议使用 `direnv` 或 shell profile 自动加载，避免在脚本中硬编码密钥。

## 5. 下一步

- 阅读 [`README.md`](../README.md) 了解整体功能。
- 查阅 [`docs/WORKFLOWS.md`](WORKFLOWS.md) 执行具体实验。
- 需要修改配置或新增模块时，记得更新 `docs/ARCHITECTURE.md`。
