"""增强版 Web Demo：更美观的 Gradio 界面，包含完整评估功能"""
import os
import gradio as gr
from pathlib import Path
from loguru import logger
import json
import time
import pandas as pd
import config

# 配置 HuggingFace 镜像（解决模型下载问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline
from src.evaluation import RAGEvaluator, FailureAnalyzer

# 配置日志
logger.add(
    config.LOGS_DIR / "web_demo.log",
    rotation="500 MB",
    level=config.LOG_LEVEL
)


class RAGWebDemo:
    """RAG Web演示应用（增强版）"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.retriever_manager = None
        self.vs_manager = None
        self.chunks = []
        self.is_initialized = False
        self.evaluator = RAGEvaluator()
        self.failure_analyzer = FailureAnalyzer()
        self.query_history = []
    
    def initialize_system(
        self,
        documents_path: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        progress=gr.Progress()
    ):
        """初始化RAG系统"""
        try:
            progress(0, desc="开始初始化...")
            logger.info("开始初始化RAG系统...")
            
            # 1. 加载和分块文档
            progress(0.2, desc="加载文档...")
            processor = DocumentProcessor(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )
            self.chunks = processor.process_documents(documents_path)
            
            if not self.chunks:
                return "❌ 未找到任何文档，请检查文档路径", ""
            
            progress(0.4, desc="创建向量索引...")
            # 2. 创建向量数据库
            self.vs_manager = VectorStoreManager(embedding_model_name=embedding_model)
            self.vs_manager.create_vectorstore(self.chunks)
            self.vs_manager.save("web_demo")
            
            progress(0.6, desc="初始化检索器...")
            # 3. 初始化检索器
            self.retriever_manager = RetrieverManager(self.vs_manager)
            self.retriever_manager.setup_bm25(self.chunks)
            
            progress(0.8, desc="初始化生成器...")
            # 4. 初始化生成器
            generator = AnswerGenerator()
            
            # 5. 创建RAG流水线
            self.rag_pipeline = RAGPipeline(self.retriever_manager, generator)
            
            self.is_initialized = True
            progress(1.0, desc="完成！")
            logger.info("RAG系统初始化成功")
            
            # 生成初始化报告
            report = f"""## ✅ 系统初始化成功！

### 📊 统计信息
- **文档块数**: {len(self.chunks)}
- **嵌入模型**: {embedding_model}
- **分块大小**: {chunk_size}
- **分块重叠**: {chunk_overlap}
- **向量数据库**: FAISS

### 📁 文档来源
"""
            # 统计文档来源
            sources = {}
            for chunk in self.chunks[:100]:  # 只统计前100个
                source = chunk.metadata.get('source', '未知')
                sources[source] = sources.get(source, 0) + 1
            
            for source, count in list(sources.items())[:10]:
                report += f"- `{Path(source).name}`: {count} 块\n"
            
            if len(sources) > 10:
                report += f"- ... 还有 {len(sources) - 10} 个文档\n"
            
            return report, "✅ 系统已就绪，可以开始问答"
        
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return f"❌ 初始化失败: {str(e)}", "❌ 初始化失败"
    
    def query(
        ......
        self,
        question: str,
        retriever_type: str,
        top_k: int,
        show_sources: bool,
        progress=gr.Progress()
    ):
        """执行查询"""
        if not self.is_initialized:
            return "❌ 请先初始化系统", "", "", None, ""
        
        if not question.strip():
            return "❌ 请输入问题", "", "", None, ""
        
        try:
            progress(0, desc="检索相关文档...")
            logger.info(f"收到查询: {question}")
            
            start_time = time.time()
            
            # 执行RAG查询
            result = self.rag_pipeline.query(
                question=question,
                retriever_type=retriever_type.lower(),
                k=int(top_k)
            )
            
            query_time = time.time() - start_time
            
            progress(0.5, desc="生成答案...")
            
            # ========== 新增：失败案例分析 ==========
            progress(0.6, desc="分析答案质量...")
            
            # 提取检索分数
            scores = []
            retrieved_docs = []
            if result.get('sources'):
                for source in result['sources']:
                    if source.get('score') != 'N/A':
                        scores.append(source['score'])
                    retrieved_docs.append(source['content'])
            
            # 执行实时质量分析
            quality_analysis = self.failure_analyzer.analyze_realtime(
                question=question,
                answer=result['answer'],
                retrieved_docs=retrieved_docs,
                scores=scores
            )
            
            # 保存查询历史（包含质量分析）
            self.query_history.append({
                'question': question,
                'retriever': retriever_type,
                'time': query_time,
                'answer': result['answer'],
                'quality_score': quality_analysis['quality_score'],
                'severity': quality_analysis['severity']
            })
            
            # ========== 格式化质量分析结果 ==========
            quality_text = f"""# {quality_analysis['quality_emoji']} 答案质量分析

## 📊 质量评分
**总分**: {quality_analysis['quality_score']:.1f}/100 | **等级**: {quality_analysis['quality_level']} | **严重程度**: {quality_analysis['severity'].upper()}

"""
            
            # 显示问题
            if quality_analysis['issues']:
                quality_text += "## ❌ 发现的问题\n\n"
                for issue in quality_analysis['issues']:
                    quality_text += f"- **{issue['message']}**\n"
                    quality_text += f"  - 类型: `{issue['type']}`\n"
                    quality_text += f"  - 严重程度: `{issue['severity']}`\n\n"
            
            # 显示警告
            if quality_analysis['warnings']:
                quality_text += "## ⚠️ 警告信息\n\n"
                for warning in quality_analysis['warnings']:
                    quality_text += f"- {warning['message']}\n"
            
            # 显示建议
            if quality_analysis['suggestions']:
                quality_text += "\n## 💡 改进建议\n\n"
                for suggestion in quality_analysis['suggestions']:
                    quality_text += f"- {suggestion}\n"
            
            # 如果没有问题
            if not quality_analysis['issues'] and not quality_analysis['warnings']:
                quality_text += "## ✅ 质量良好\n\n"
                quality_text += "未发现明显问题，答案质量良好。\n"
            
            # 误差归因
            if quality_analysis.get('error_type'):
                quality_text += f"\n## 🔍 误差归因\n\n"
                quality_text += f"**主要错误类型**: `{quality_analysis['error_type']}`\n\n"
                
                error_descriptions = {
                    'retrieval_error': '**检索错误**：系统未能检索到与问题相关的文档。可能原因：\n- 文档库中没有相关内容\n- 问题表述与文档用词差异较大\n- 嵌入模型未能理解问题语义',
                    'ranking_error': '**排序错误**：相关文档被检索到但排序靠后。可能原因：\n- 检索器评分机制不够准确\n- BM25对语义理解较弱\n- 需要使用Hybrid混合检索',
                    'generation_error': '**生成错误**：LLM生成的答案质量不佳。可能原因：\n- 检索到的文档不相关\n- LLM未能正确理解上下文\n- 提示词设计需要优化',
                    'low_confidence': '**低置信度**：检索文档相关度过低。可能原因：\n- 问题超出文档库范围\n- 需要更换检索策略\n- 考虑添加更多相关文档'
                }
                
                if quality_analysis['error_type'] in error_descriptions:
                    quality_text += error_descriptions[quality_analysis['error_type']]
            
            # ========== 格式化答案 ==========
            answer = f"""## 💬 回答

{result['answer']}

---
⏱️ **查询耗时**: {query_time:.3f}秒 | 🔍 **检索器**: {retriever_type.upper()} | 📄 **Top-K**: {top_k} | {quality_analysis['quality_emoji']} **质量**: {quality_analysis['quality_score']:.0f}/100
"""
            
            # 格式化来源
            sources_text = ""
            sources_json = ""
            sources_df = None
            
            progress(0.8, desc="整理来源信息...")
            
            if show_sources and result.get('sources'):
                sources_text = "## 📚 参考来源\n\n"
                
                sources_data = []
                for source in result['sources']:
                    idx = source['index']
                    content = source['content']
                    metadata = source.get('metadata', {})
                    score = source.get('score', 'N/A')
                    
                    # Markdown格式
                    sources_text += f"""### 📄 来源 {idx}

**内容预览**:  
{content[:300]}...

**文档**: `{Path(metadata.get('source', '未知')).name}`  
"""
                    if score != 'N/A':
                        sources_text += f"**相关度**: `{score:.4f}` {'🟢' if score > 0.7 else '🟡' if score > 0.5 else '🔴'}\n"
                    
                    sources_text += "\n---\n\n"
                    
                    # DataFrame数据
                    sources_data.append({
                        '序号': idx,
                        '内容预览': content[:100] + '...',
                        '来源文件': Path(metadata.get('source', '未知')).name,
                        '相关度': f"{score:.4f}" if score != 'N/A' else 'N/A'
                    })
                
                # 创建DataFrame
                sources_df = pd.DataFrame(sources_data)
                
                # JSON格式的来源信息
                sources_json = json.dumps(
                    result['sources'],
                    ensure_ascii=False,
                    indent=2
                )
            
            progress(1.0, desc="完成！")
            
            return answer, sources_text, sources_json, sources_df, quality_text
        
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"❌ 查询失败: {str(e)}", "", "", None, ""
    
    def compare_retrievers(self, question: str, top_k: int, progress=gr.Progress()):
        """对比不同检索器"""
        if not self.is_initialized:
            return "❌ 请先初始化系统", None
        
        if not question.strip():
            return "❌ 请输入问题", None
        
        try:
            progress(0, desc="准备对比...")
            comparison = self.retriever_manager.compare_retrievers(
                question, 
                k=int(top_k)
            )
            
            # 格式化对比结果
            result_text = f"""# 🔍 检索器性能对比

**查询问题**: {question}

---
"""
            
            comparison_data = []
            
            for i, retriever_name in enumerate(['faiss', 'bm25', 'hybrid'], 1):
                progress(i/4, desc=f"分析 {retriever_name.upper()}...")
                
                result_text += f"\n## {'🚀' if retriever_name == 'hybrid' else '📊'} {retriever_name.upper()} 检索器\n\n"
                
                results = comparison[retriever_name]
                for j, doc_info in enumerate(results, 1):
                    result_text += f"### Top-{j}\n"
                    result_text += f"**内容**: {doc_info['content'][:200]}...\n"
                    
                    if 'score' in doc_info:
                        score = doc_info['score']
                        result_text += f"**分数**: `{score:.4f}` {'🟢' if score > 0.7 else '🟡' if score > 0.5 else '🔴'}\n"
                        
                        # 添加到对比数据
                        comparison_data.append({
                            '检索器': retriever_name.upper(),
                            '排名': j,
                            '内容预览': doc_info['content'][:80] + '...',
                            '相关度': f"{score:.4f}"
                        })
                    else:
                        comparison_data.append({
                            '检索器': retriever_name.upper(),
                            '排名': j,
                            '内容预览': doc_info['content'][:80] + '...',
                            '相关度': 'N/A'
                        })
                    
                    result_text += "\n"
                
                result_text += "---\n"
            
            progress(1.0, desc="完成！")
            
            # 创建对比DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            return result_text, comparison_df
        
        except Exception as e:
            logger.error(f"检索器对比失败: {e}")
            return f"❌ 对比失败: {str(e)}", None
    
    def run_performance_test(self, num_queries: int, progress=gr.Progress()):
        """运行性能测试"""
        if not self.is_initialized:
            return "❌ 请先初始化系统", None
        
        try:
            # 生成测试问题
            test_questions = [
                "什么是人工智能？",
                "机器学习的应用有哪些？",
                "深度学习和机器学习的区别？",
                "神经网络是如何工作的？",
                "自然语言处理有哪些应用？"
            ] * (int(num_queries) // 5 + 1)
            test_questions = test_questions[:int(num_queries)]
            
            progress(0.1, desc="开始性能测试...")
            
            # 测量性能
            performance = self.evaluator.measure_performance(
                self.rag_pipeline,
                test_questions,
                retriever_types=['faiss', 'bm25', 'hybrid']
            )
            
            progress(0.8, desc="生成报告...")
            
            # 格式化结果
            report = f"""# 📊 性能测试报告

**测试查询数**: {num_queries}

---

"""
            
            perf_data = []
            for retriever, metrics in performance.items():
                report += f"""## {retriever.upper()} 检索器

- **平均延迟**: `{metrics['avg_latency']:.3f}` 秒
- **最小延迟**: `{metrics['min_latency']:.3f}` 秒
- **最大延迟**: `{metrics['max_latency']:.3f}` 秒
- **吞吐量**: `{metrics['throughput']:.2f}` queries/sec
- **总查询数**: {metrics['num_queries']}

"""
                perf_data.append({
                    '检索器': retriever.upper(),
                    '平均延迟(秒)': f"{metrics['avg_latency']:.3f}",
                    '最小延迟(秒)': f"{metrics['min_latency']:.3f}",
                    '最大延迟(秒)': f"{metrics['max_latency']:.3f}",
                    '吞吐量(q/s)': f"{metrics['throughput']:.2f}"
                })
            
            progress(1.0, desc="完成！")
            
            perf_df = pd.DataFrame(perf_data)
            
            return report, perf_df
        
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return f"❌ 性能测试失败: {str(e)}", None
    
    def get_query_history(self):
        """获取查询历史"""
        if not self.query_history:
            return None
        
        return pd.DataFrame(self.query_history)


def create_enhanced_web_interface():
    """创建增强版 Gradio Web 界面"""
    demo_app = RAGWebDemo()
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
    .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    .info { background-color: #d1ecf1; border: 1px solid #bee5eb; }
    """
    
    with gr.Blocks(
        title="🤖 增强版 RAG 问答系统",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
        css=custom_css
    ) as demo:
        
        gr.Markdown("""
        # 🤖 RAG 问答系统 - 增强版
        
        基于 **LangChain** 的检索增强生成（RAG）智能问答系统
        
        [![GitHub](https://img.shields.io/badge/GitHub-项目地址-blue)](https://github.com/your-repo)
        [![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
        [![LangChain](https://img.shields.io/badge/LangChain-0.3.0-orange)](https://python.langchain.com/)
        """)
        
        # 状态栏
        with gr.Row():
            system_status = gr.Textbox(
                label="系统状态",
                value="⚠️ 未初始化",
                interactive=False,
                scale=1
            )
        
        # Tab 1: 系统初始化
        with gr.Tab("📖 系统初始化"):
            gr.Markdown("""
            ## 🚀 配置并初始化 RAG 系统
            
            在开始使用前，请先配置系统参数并初始化。初始化过程包括：
            1. 加载并分块文档
            2. 创建向量索引
            3. 初始化检索器（FAISS、BM25、混合）
            4. 初始化答案生成器
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 配置参数")
                    
                    docs_path_input = gr.Textbox(
                        label="📁 文档路径",
                        value=str(config.DOCUMENTS_PATH),
                        placeholder="输入文档目录路径",
                        info="支持 PDF、TXT、DOCX 等格式"
                    )
                    
                    embedding_model_input = gr.Textbox(
                        label="🤖 嵌入模型",
                        value=config.EMBEDDING_MODEL,
                        placeholder="HuggingFace 模型名称",
                        info="推荐: BAAI/bge-base-zh-v1.5 或 BAAI/bge-large-zh-v1.5"
                    )
                    
                    with gr.Row():
                        chunk_size_input = gr.Slider(
                            label="📏 文档块大小",
                            minimum=100,
                            maximum=2000,
                            value=config.CHUNK_SIZE,
                            step=100,
                            info="每个文档块的字符数"
                        )
                        
                        chunk_overlap_input = gr.Slider(
                            label="🔗 文档块重叠",
                            minimum=0,
                            maximum=200,
                            value=config.CHUNK_OVERLAP,
                            step=10,
                            info="相邻块之间的重叠字符数"
                        )
                    
                    init_button = gr.Button(
                        "🚀 初始化系统",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 初始化结果")
                    init_output = gr.Markdown(
                        value="等待初始化..."
                    )
            
            init_button.click(
                fn=demo_app.initialize_system,
                inputs=[
                    docs_path_input,
                    embedding_model_input,
                    chunk_size_input,
                    chunk_overlap_input
                ],
                outputs=[init_output, system_status]
            )
        
        # Tab 2: 智能问答
        with gr.Tab("💬 智能问答"):
            gr.Markdown("""
            ## 🎯 提出问题，获取基于文档的答案
            
            系统会自动检索相关文档，并生成带引用的准确答案。
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="❓ 输入您的问题",
                        placeholder="例如：什么是人工智能？机器学习有哪些应用？",
                        lines=3
                    )
                    
                    with gr.Row():
                        retriever_type = gr.Radio(
                            label="🔍 检索器类型",
                            choices=["FAISS", "BM25", "Hybrid"],
                            value="Hybrid",
                            info="Hybrid 混合检索通常效果最好"
                        )
                        
                        top_k_slider = gr.Slider(
                            label="📊 Top-K",
                            minimum=1,
                            maximum=10,
                            value=config.TOP_K,
                            step=1,
                            info="返回最相关的 K 个文档"
                        )
                    
                    show_sources_checkbox = gr.Checkbox(
                        label="📚 显示来源文档",
                        value=True
                    )
                    
                    query_button = gr.Button(
                        "🔍 开始查询",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=3):
                    answer_output = gr.Markdown(
                        label="💡 答案"
                    )
            
            # ========== 新增：质量分析显示区域 ==========
            with gr.Accordion("🔍 答案质量分析与失败案例归因", open=True):
                quality_analysis_output = gr.Markdown(
                    value="等待查询后显示质量分析..."
                )
            
            with gr.Accordion("📚 参考来源详情", open=False):
                with gr.Tabs():
                    with gr.Tab("📝 来源文本"):
                        sources_output = gr.Markdown()
                    
                    with gr.Tab("📊 来源表格"):
                        sources_table = gr.Dataframe(
                            headers=["序号", "内容预览", "来源文件", "相关度"],
                            interactive=False
                        )
                    
                    with gr.Tab("🔧 JSON 数据"):
                        sources_json_output = gr.Code(
                            label="来源 JSON",
                            language="json"
                        )
            
            query_button.click(
                fn=demo_app.query,
                inputs=[
                    question_input,
                    retriever_type,
                    top_k_slider,
                    show_sources_checkbox
                ],
                outputs=[
                    answer_output,
                    sources_output,
                    sources_json_output,
                    sources_table,
                    quality_analysis_output  # 新增输出
                ]
            )
            
            # 示例问题
            gr.Markdown("### 💡 试试这些示例问题")
            with gr.Row():
                gr.Markdown("""
                **高质量问题示例**（文档中有相关内容）：
                - 什么是人工智能？
                - 机器学习的应用有哪些？
                
                **低质量问题示例**（测试失败案例分析）：
                - 如何做红烧肉？（与文档无关）
                - 明天的天气怎么样？（文档范围外）
                """)
            
            gr.Examples(
                examples=[
                    ["什么是人工智能？", "Hybrid", 5, True],
                    ["机器学习的应用有哪些？", "Hybrid", 5, True],
                    ["深度学习和机器学习的区别是什么？", "FAISS", 3, True],
                    ["神经网络是如何工作的？", "BM25", 5, True],
                    ["如何做红烧肉？", "Hybrid", 5, True],  # 故意的失败案例
                ],
                inputs=[
                    question_input,
                    retriever_type,
                    top_k_slider,
                    show_sources_checkbox
                ],
                label="点击加载示例"
            )
        
        # Tab 3: 检索器对比
        with gr.Tab("🔄 检索器对比"):
            gr.Markdown("""
            ## 📊 对比不同检索器的性能
            
            同时使用 **FAISS**、**BM25** 和 **Hybrid** 三种检索器，查看它们的检索结果差异。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    compare_question = gr.Textbox(
                        label="❓ 输入问题",
                        placeholder="例如：什么是自然语言处理？",
                        lines=3
                    )
                    
                    compare_top_k = gr.Slider(
                        label="📊 Top-K",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1
                    )
                    
                    compare_button = gr.Button(
                        "🔍 开始对比",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    comparison_output = gr.Markdown()
            
            with gr.Accordion("📊 对比数据表格", open=True):
                comparison_table = gr.Dataframe(
                    headers=["检索器", "排名", "内容预览", "相关度"],
                    interactive=False
                )
            
            compare_button.click(
                fn=demo_app.compare_retrievers,
                inputs=[compare_question, compare_top_k],
                outputs=[comparison_output, comparison_table]
            )
        
        # Tab 4: 性能测试
        with gr.Tab("⚡ 性能测试"):
            gr.Markdown("""
            ## 📈 测试系统性能指标
            
            测量不同检索器的 **延迟** 和 **吞吐量**，评估系统性能。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    num_queries_input = gr.Slider(
                        label="🔢 测试查询数",
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        info="建议 10-20 次查询"
                    )
                    
                    perf_test_button = gr.Button(
                        "🚀 开始性能测试",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    perf_output = gr.Markdown()
            
            with gr.Accordion("📊 性能数据表格", open=True):
                perf_table = gr.Dataframe(
                    headers=["检索器", "平均延迟(秒)", "最小延迟(秒)", "最大延迟(秒)", "吞吐量(q/s)"],
                    interactive=False
                )
            
            perf_test_button.click(
                fn=demo_app.run_performance_test,
                inputs=[num_queries_input],
                outputs=[perf_output, perf_table]
            )
        
        # Tab 5: 查询历史
        with gr.Tab("📜 查询历史"):
            gr.Markdown("""
            ## 📊 查看历史查询记录
            
            记录了所有查询的问题、使用的检索器、耗时等信息。
            """)
            
            history_refresh_button = gr.Button("🔄 刷新历史", variant="secondary")
            history_table = gr.Dataframe(
                headers=["问题", "检索器", "耗时(秒)", "答案预览"],
                interactive=False
            )
            
            history_refresh_button.click(
                fn=demo_app.get_query_history,
                outputs=history_table
            )
        
        # Footer
        gr.Markdown("""
        ---
        
        ## 💡 使用指南
        
        ### 快速开始
        1. **初始化系统**: 在"系统初始化"标签页配置参数并初始化
        2. **开始问答**: 在"智能问答"标签页输入问题
        3. **对比检索器**: 在"检索器对比"标签页查看不同检索器的效果
        4. **性能测试**: 在"性能测试"标签页评估系统性能
        
        ### 技术栈
        - **框架**: LangChain
        - **向量数据库**: FAISS
        - **检索器**: FAISS 向量检索、BM25 文本检索、混合检索
        - **嵌入模型**: BGE (Beijing Academy of Artificial Intelligence)
        - **评测工具**: RAGAS
        
        ### 特性
        - ✅ 多种文档格式支持 (PDF、TXT、DOCX、XLSX)
        - ✅ 智能文档分块与向量索引
        - ✅ 三种检索策略可选
        - ✅ 带引用的答案生成
        - ✅ 完整的性能评估体系
        - ✅ 实时查询历史记录
        
        ### 📚 相关资源
        - [LangChain 文档](https://python.langchain.com/)
        - [RAGAS 评测框架](https://github.com/explodinggradients/ragas)
        - [FAISS 向量检索](https://github.com/facebookresearch/faiss)
        
        ---
        
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🌟 如果觉得有用，请给项目一个 Star！</p>
            <p>💬 有问题或建议？欢迎提 Issue！</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # 启动增强版 Web 应用
    demo = create_enhanced_web_interface()
    
    # 尝试多个端口，避免端口占用问题
    for port in [7860, 7861, 7862, 7863, 7864]:
        try:
            print(f"\n尝试在端口 {port} 启动...")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True
            )
            break  # 成功启动则退出循环
        except OSError as e:
            if "10048" in str(e) or "Cannot find empty port" in str(e):
                print(f"端口 {port} 已被占用，尝试下一个端口...")
                continue
            else:
                raise  # 其他错误则抛出
    else:
        print("\n❌ 错误: 无法找到可用端口")
        print("请手动关闭占用端口的进程或重启电脑")

