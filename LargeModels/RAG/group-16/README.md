# 航小厨 - 基于 LangChain 的智能 RAG 烹饪助手

## 项目简介

航小厨是一个基于 LangChain 开发的智能检索增强生成（RAG）系统，专门用于烹饪领域的问答助手。系统集成了向量检索、重排模型和智能 Agent，能够基于菜谱数据库提供专业的烹饪建议。

### 核心特性

- **向量检索**：使用 FAISS 向量数据库进行快速相似度搜索
- **重排模型**：可选的重排功能，提升检索结果的相关性
- **智能 Agent**：基于 LangChain 的 Agent 框架，支持工具调用和记忆功能
- **交互界面**：基于 Streamlit 的友好 Web 界面

### 技术栈

- **框架**：LangChain 0.3.x
- **向量数据库**：FAISS
- **嵌入模型**：DashScope text-embedding-v4
- **重排模型**：DashScope qwen3-rerank
- **LLM**：DashScope qwen-flash
- **Web 框架**：Streamlit

## 使用说明

### 环境配置

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置 API Key**

编辑 `code/retriever-generator/agent_backend.py` 文件，设置你的 DashScope API Key：
```python
API_KEY = "your-api-key-here"
```

3. **准备向量索引**

确保 `code/retriever-generator/faiss_index4/` 目录下存在向量索引文件。

### 运行应用

```bash
cd code/retriever-generator
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址为：`http://localhost:8501`

### 使用步骤

1. **配置参数**（侧边栏）
   - 选择是否启用重排
   - 设置重排模型和参数（Top K、Top N）

2. **开始对话**
   - 在输入框中输入你的问题
   - 系统会自动检索相关文档并生成回答

3. **查看性能指标**
   - 侧边栏显示平均性能指标
   - 每次查询后显示本次查询的性能指标

## 项目结构

```
group-16/
├── code/                          # 代码目录
│   ├── retriever-generator/       # 主要应用目录
│   │   ├── app.py                 # Streamlit 主应用
│   │   ├── agent_backend.py       # Agent 后端逻辑
│   │   └── faiss_index4/          # FAISS 向量索引
│   ├── faiss_index/               # 索引构建脚本
│   ├── data_preprocess/           # 数据预处理脚本
│   ├── Evaluation Results Analysis/  # 评估结果分析
│   └── requirements.txt           # 依赖列表
└── README.md                      # 本文件
```

## 结果展示

### 功能演示

系统能够回答各种烹饪相关问题，包括：

1. **具体菜谱做法**
   - 用户："宫保鸡丁怎么做？"
   - 系统：返回详细的食材清单和烹饪步骤

2. **根据已有食材推荐菜谱**
   - 用户："我只有鸡蛋和番茄，能做什么？"
   - 系统：推荐相关菜谱

3. **复杂规划任务**
   - 用户："家庭聚会 6 人的菜谱搭配"
   - 系统：通过多次检索，推荐合适的荤素搭配方案

4. **烹饪技巧查询**
   - 用户："什么是'焯水'？"
   - 系统：基于知识库解释烹饪技巧

### 性能指标

系统提供详细的性能监控：

- **检索时间**：向量检索和重排的总耗时
- **生成时间**：LLM 生成回答的耗时
- **总延迟**：从查询到回答的总时间
- **吞吐量**：每秒处理的请求数

### 评估结果

项目使用 RAGAS 框架进行了全面评估，包括：

- **答案相关性（Answer Relevancy）**：衡量答案与问题的相关度
- **事实准确性（Faithfulness）**：衡量答案是否忠实于检索到的上下文
- **上下文召回率（Context Recall）**：衡量检索到的上下文是否覆盖标准答案
- **上下文精确率（Context Precision）**：衡量检索到的上下文是否与问题相关

详细评估结果请参考 `code/Evaluation Results Analysis/Results/` 目录下的评估报告。

## 核心创新点

1. **智能 Agent 架构**：使用 LangChain Agent 实现复杂查询的自动拆解和多轮检索
2. **重排机制**：集成 DashScope 重排模型，提升检索精度
3. **对话记忆**：支持多轮对话，保持上下文连贯性
4. **性能监控**：实时监控检索和生成性能

## 数据来源

本项目的数据源自 GitHub 项目 [HowToCook](https://github.com/Anduin2017/HowToCook)，一个优秀的程序员在家做饭方法指南。

## 团队成员

- 组号：group-16

## 许可证

本项目仅供学习和研究使用。

---

**祝您使用愉快！🍳**

