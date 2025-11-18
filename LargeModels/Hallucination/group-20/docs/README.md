# 📚 AI4Hallucination 文档索引

集中化的文档导航，帮助你快速了解各个模块、实验流程以及可扩展点。

## 文档地图

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：模块划分、数据流和依赖关系。
- [`WORKFLOWS.md`](WORKFLOWS.md)：常用评测/演示工作流操作手册。
- [`QUICKSTART.md`](QUICKSTART.md)：面向新同学的环境配置、脚本与测试指引。

> 未来如需补充实验记录、指标说明或 FAQ，可在 `docs/` 下新增独立文件并在此处登记。

## 约定

- **目录结构**：所有核心代码统一放在 `modules/`，文档放在 `docs/`，自动化脚本放在 `scripts/`。
- **命名规范**：目录与脚本采用小写+连字符形式（如 `selfrag-lite`），Markdown 文件采用帕斯卡命名。
- **同步要求**：当代码结构变更时，请同步更新 `docs/` 与根目录 `README.md` 确保信息一致。

如需贡献文档，请在对应章节点后附上维护者或日期，便于后续追踪。欢迎通过 PR 补充改进！
