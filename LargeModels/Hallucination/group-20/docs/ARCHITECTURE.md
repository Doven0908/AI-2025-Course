# 🏗️ 架构蓝图

本文档梳理 AI4Hallucination 的核心模块、数据流和依赖，帮助快速定位组件及其职责。

## 顶层布局

```
ai4hallucination-main/
├── README.md
├── docs/                # 说明文档（当前文件）
├── modules/             # 代码与实验模块
└── scripts/             # 环境/运行脚本
```

所有业务逻辑均放入 `modules/`，再按功能切分：

| 目录 | 关键内容 | 作用 |
| --- | --- | --- |
| `modules/comprehensive` | 统一配置的多数据集评估框架 | 在相同配置下跑多种求解策略并输出 YAML 结果 |
| `modules/cove` | Chain-of-Verification + LettuceDetect | 对 baseline/CoVe 输出进行 span 级幻觉检测 |
| `modules/selfrag-official` | SELF-RAG 官方实现 | 用于复现实验、训练与大规模评估 |
| `modules/selfrag-lite` | 精简 SELF-RAG & SelfCheckGPT 实验脚本 | 快速实验 RAGAS、长文本问答、幻觉检测 |
| `modules/selfcheckgpt` | SelfCheckGPT 组件 | 可作为库被其他模块复用的自检实现 |

## 数据流

1. **数据准备**：从 `modules/selfrag-lite/eval_data` 或 `modules/comprehensive/data` 加载样本。
2. **求解器**：在 comprehensive 模块中根据 `config.yaml` 调度 direct-answer / RAG / SELF-RAG / CoVe / SelfCheckGPT。
3. **评估器**：统一通过 `Evaluator` (RAGAS) 获得 Faithfulness、Context Precision 等指标。
4. **可视化与汇报**：`selfrag-official` 和 `selfrag-lite` 内的脚本生成表格/图像；`docs/` 记录实验流程。

## 依赖关系

- `modules/comprehensive` 依赖 `modules/selfcheckgpt`（自检算法）与外部 API（OpenAI/LiteLLM）。
- `modules/cove` 依赖 `lettucedetect`, `langchain`, 以及 `question.json` 中的上下文数据。
- `modules/selfrag-*` 依赖 `vllm`, `transformers`, `ragas` 等，可选 GPU。
- `scripts/` 中的 Shell 脚本调用 `python modules/...` 入口，实现可重复的实验管线。

## 扩展指南

- 新模块请放入 `modules/<name>`, 并在根 README 与本文件增加描述。
- 若模块需要说明文档，可在 `docs/` 下新增 `<NAME>.md` 并建立链接。
- 脚本建议统一放在 `scripts/`，命名为 `run_<module>.sh` 或 `setup_<area>.sh`。

保持以上约定有助于多团队协同与自动化部署。若有结构变更建议，请在 PR 中同步更新本文件。***
