"""Web Demo：基于Gradio的交互式问答界面"""
import gradio as gr
from pathlib import Path
from loguru import logger
import json
import config
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline
# 配置日志
logger.add(
    config.LOGS_DIR / "web_demo.log",
    rotation="500 MB",
    level=config.LOG_LEVEL
)


class RAGWebDemo:
    """RAG Web演示应用"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.retriever_manager = None
        self.is_initialized = False
    
    def initialize_system(
        self,
        documents_path: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int
    ):
        """初始化RAG系统"""
        try:
            logger.info("开始初始化RAG系统...")
            
            # 1. 加载和分块文档
            processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = processor.process_documents(documents_path)
            
            if not chunks:
                return False, "未找到任何文档，请检查文档路径"
            
            # 2. 创建向量数据库
            vs_manager = VectorStoreManager(embedding_model_name=embedding_model)
            vs_manager.create_vectorstore(chunks)
            vs_manager.save("web_demo")
            
            # 3. 初始化检索器
            self.retriever_manager = RetrieverManager(vs_manager)
            self.retriever_manager.setup_bm25(chunks)
            
            # 4. 初始化生成器
            generator = AnswerGenerator()
            
            # 5. 创建RAG流水线
            self.rag_pipeline = RAGPipeline(self.retriever_manager, generator)
            
            self.is_initialized = True
            logger.info("RAG系统初始化成功")
            
            return True, f"✅ 系统初始化成功！\n文档块数: {len(chunks)}"
        
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False, f"❌ 初始化失败: {str(e)}"
    
    def query(
        self,
        question: str,
        retriever_type: str,
        top_k: int,
        show_sources: bool
    ):
        """执行查询"""
        if not self.is_initialized:
            return "❌ 请先初始化系统", "", ""
        
        if not question.strip():
            return "❌ 请输入问题", "", ""
        
        try:
            logger.info(f"收到查询: {question}")
            
            # 执行RAG查询
            result = self.rag_pipeline.query(
                question=question,
                retriever_type=retriever_type.lower(),
                k=top_k
            )
            
            # 格式化答案
            answer = result['answer']
            
            # 格式化来源
            sources_text = ""
            sources_json = ""
            
            if show_sources and result.get('sources'):
                sources_text = "\n\n## 📚 参考来源\n\n"
                
                for source in result['sources']:
                    idx = source['index']
                    content = source['content'][:300]
                    metadata = source.get('metadata', {})
                    score = source.get('score', 'N/A')
                    
                    sources_text += f"### [来源 {idx}]\n"
                    sources_text += f"**内容预览**: {content}...\n"
                    sources_text += f"**来源**: {metadata.get('source', '未知')}\n"
                    
                    if score != 'N/A':
                        sources_text += f"**相关度**: {score:.4f}\n"
                    
                    sources_text += "\n---\n\n"
                
                # JSON格式的来源信息
                sources_json = json.dumps(
                    result['sources'],
                    ensure_ascii=False,
                    indent=2
                )
            
            return answer, sources_text, sources_json
        
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"❌ 查询失败: {str(e)}", "", ""
    
    def compare_retrievers(self, question: str, top_k: int):
        """对比不同检索器"""
        if not self.is_initialized:
            return "❌ 请先初始化系统"
        
        if not question.strip():
            return "❌ 请输入问题"
        
        try:
            comparison = self.retriever_manager.compare_retrievers(
                question, 
                k=top_k
            )
            
            # 格式化对比结果
            result_text = f"# 🔍 检索器对比\n\n**查询**: {question}\n\n"
            
            for retriever_name in ['faiss', 'bm25', 'hybrid']:
                result_text += f"\n## {retriever_name.upper()} 检索器\n\n"
                
                results = comparison[retriever_name]
                for i, doc_info in enumerate(results, 1):
                    result_text += f"### Top-{i}\n"
                    result_text += f"**内容**: {doc_info['content']}...\n"
                    
                    if 'score' in doc_info:
                        result_text += f"**分数**: {doc_info['score']:.4f}\n"
                    
                    result_text += "\n"
                
                result_text += "---\n"
            
            return result_text
        
        except Exception as e:
            logger.error(f"检索器对比失败: {e}")
            return f"❌ 对比失败: {str(e)}"


def create_web_interface():
    """创建Gradio Web界面"""
    demo_app = RAGWebDemo()
    
    with gr.Blocks(title="RAG问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 RAG问答系统演示")
        gr.Markdown("基于 LangChain 的检索增强生成（RAG）问答系统")
        
        with gr.Tab("📖 系统初始化"):
            gr.Markdown("## 配置并初始化RAG系统")
            
            with gr.Row():
                with gr.Column():
                    docs_path_input = gr.Textbox(
                        label="文档路径",
                        value=str(config.DOCUMENTS_PATH),
                        placeholder="输入文档目录路径"
                    )
                    
                    embedding_model_input = gr.Textbox(
                        label="嵌入模型",
                        value=config.EMBEDDING_MODEL,
                        placeholder="HuggingFace模型名称"
                    )
                    
                    with gr.Row():
                        chunk_size_input = gr.Slider(
                            label="文档块大小",
                            minimum=100,
                            maximum=2000,
                            value=config.CHUNK_SIZE,
                            step=100
                        )
                        
                        chunk_overlap_input = gr.Slider(
                            label="文档块重叠",
                            minimum=0,
                            maximum=200,
                            value=config.CHUNK_OVERLAP,
                            step=10
                        )
                    
                    init_button = gr.Button("🚀 初始化系统", variant="primary")
                
                with gr.Column():
                    init_output = gr.Textbox(
                        label="初始化状态",
                        lines=10
                    )
            
            init_button.click(
                fn=demo_app.initialize_system,
                inputs=[
                    docs_path_input,
                    embedding_model_input,
                    chunk_size_input,
                    chunk_overlap_input
                ],
                outputs=init_output
            )
        
        with gr.Tab("💬 问答"):
            gr.Markdown("## 提出问题，获取带引用的答案")
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：什么是人工智能？",
                        lines=3
                    )
                    
                    with gr.Row():
                        retriever_type = gr.Radio(
                            label="检索器类型",
                            choices=["FAISS", "BM25", "Hybrid"],
                            value="FAISS"
                        )
                        
                        top_k_slider = gr.Slider(
                            label="Top-K",
                            minimum=1,
                            maximum=10,
                            value=config.TOP_K,
                            step=1
                        )
                    
                    show_sources_checkbox = gr.Checkbox(
                        label="显示来源文档",
                        value=True
                    )
                    
                    query_button = gr.Button("🔍 查询", variant="primary")
                
                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="答案",
                        lines=10
                    )
            
            with gr.Accordion("📚 参考来源", open=False):
                sources_output = gr.Markdown()
            
            with gr.Accordion("🔧 详细信息 (JSON)", open=False):
                sources_json_output = gr.Code(
                    label="来源JSON",
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
                    sources_json_output
                ]
            )
            
            # 示例问题
            gr.Examples(
                examples=[
                    ["什么是人工智能？", "FAISS", 5, True],
                    ["机器学习的应用有哪些？", "Hybrid", 3, True],
                    ["深度学习和机器学习的区别？", "BM25", 5, True],
                ],
                inputs=[
                    question_input,
                    retriever_type,
                    top_k_slider,
                    show_sources_checkbox
                ]
            )
        
        with gr.Tab("🔄 检索器对比"):
            gr.Markdown("## 对比不同检索器的效果")
            
            with gr.Row():
                with gr.Column():
                    compare_question = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：什么是自然语言处理？",
                        lines=3
                    )
                    
                    compare_top_k = gr.Slider(
                        label="Top-K",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1
                    )
                    
                    compare_button = gr.Button("🔍 开始对比", variant="primary")
                
                with gr.Column(scale=2):
                    comparison_output = gr.Markdown()
            
            compare_button.click(
                fn=demo_app.compare_retrievers,
                inputs=[compare_question, compare_top_k],
                outputs=comparison_output
            )
        
        gr.Markdown("""
        ---
        ### 💡 使用说明
        1. **系统初始化**: 配置文档路径和参数，然后初始化系统
        2. **问答**: 输入问题，选择检索器类型，获取带引用的答案
        3. **检索器对比**: 对比FAISS、BM25和混合检索器的效果
        
        ### 📝 技术栈
        - **框架**: LangChain
        - **向量数据库**: FAISS
        - **检索器**: FAISS向量检索、BM25、混合检索
        - **评测**: RAGAS
        """)
    
    return demo


if __name__ == "__main__":
    # 启动Web应用
    demo = create_web_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

