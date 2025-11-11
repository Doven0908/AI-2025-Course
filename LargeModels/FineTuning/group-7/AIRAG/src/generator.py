"""生成模块：基于检索结果生成带引用的答案"""
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from loguru import logger
import config


class AnswerGenerator:
    """答案生成器：基于检索结果生成带引用的答案"""
    
    def __init__(self, model_type: str = None, api_key: str = None, base_url: str = None, model_name: str = None):
        self.model_type = model_type or config.MODEL_TYPE
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.model_name = model_name or config.LLM_MODEL_NAME
        
        # 初始化LLM
        self.llm = self._init_llm()
        
        # 定义提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是一个专业的问答助手。请基于以下提供的上下文信息回答用户的问题。

要求：
1. 仅使用提供的上下文信息回答问题
2. 如果上下文中没有相关信息，明确说明"根据提供的信息无法回答该问题"
3. 在答案中标注信息来源，使用[来源X]的格式
4. 答案要准确、简洁、有条理

上下文信息：
{context}

用户问题：{question}

请回答："""
        )
        
        # 创建链
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _init_llm(self):
        """初始化语言模型"""
        try:
            if self.model_type == "api":
                if not self.api_key:
                    raise ValueError("使用API模式需要提供API密钥")
                
                llm = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=0.7,
                    max_tokens=1000
                )
                logger.info(f"使用在线API初始化LLM: {self.model_name}")
                logger.info(f"API Base URL: {self.base_url}")
            
            else:  # local
                # 使用本地模型或其他配置
                # 这里可以配置本地模型，例如使用HuggingFace Pipeline
                llm = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=self.api_key or "dummy",
                    openai_api_base=self.base_url,
                    temperature=0.7,
                    max_tokens=1000
                )
                logger.info(f"使用本地/自定义模型初始化LLM: {self.model_name}")
            
            return llm
        
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise
    
    def format_context(
        self, 
        documents: List[Document],
        with_scores: bool = False,
        scores: Optional[List[float]] = None
    ) -> str:
        """格式化上下文信息"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # 提取文档内容和元数据
            content = doc.page_content
            metadata = doc.metadata
            
            # 格式化来源信息
            source = metadata.get('source', '未知来源')
            page = metadata.get('page', '')
            
            source_info = f"[来源{i}]"
            if page:
                source_info += f" (页码: {page})"
            source_info += f" {source}"
            
            # 添加评分信息（如果有）
            if with_scores and scores and i <= len(scores):
                source_info += f" [相关度: {scores[i-1]:.4f}]"
            
            context_parts.append(f"{source_info}\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def generate_answer(
        self, 
        question: str, 
        retrieved_docs: List[Document],
        scores: Optional[List[float]] = None
    ) -> Dict:
        """生成带引用的答案"""
        try:
            # 格式化上下文
            context = self.format_context(
                retrieved_docs, 
                with_scores=(scores is not None),
                scores=scores
            )
            
            # 生成答案
            logger.info(f"开始生成答案，问题: {question}")
            response = self.chain.run(context=context, question=question)
            
            # 构建结果
            result = {
                'question': question,
                'answer': response,
                'sources': [],
                'context': context
            }
            
            # 提取来源信息
            for i, doc in enumerate(retrieved_docs, 1):
                source_info = {
                    'index': i,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                if scores and i <= len(scores):
                    source_info['score'] = float(scores[i-1])
                
                result['sources'].append(source_info)
            
            logger.info("答案生成完成")
            return result
        
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return {
                'question': question,
                'answer': f"生成答案时出错: {str(e)}",
                'sources': [],
                'context': ''
            }
    
    def generate_with_retrieval_analysis(
        self,
        question: str,
        retrieved_docs: List[Document],
        scores: Optional[List[float]] = None
    ) -> Dict:
        """生成答案并分析检索质量"""
        result = self.generate_answer(question, retrieved_docs, scores)
        
        # 添加检索质量分析
        result['retrieval_analysis'] = {
            'num_retrieved': len(retrieved_docs),
            'avg_score': sum(scores) / len(scores) if scores else None,
            'score_distribution': scores if scores else None
        }
        
        return result


class RAGPipeline:
    """完整的RAG流水线"""
    
    def __init__(self, retriever_manager, generator):
        self.retriever_manager = retriever_manager
        self.generator = generator
    
    def query(
        self, 
        question: str, 
        retriever_type: str = "faiss",
        k: int = None
    ) -> Dict:
        """执行完整的RAG查询流程"""
        k = k or config.TOP_K
        
        logger.info(f"开始RAG查询: {question}")
        logger.info(f"检索器类型: {retriever_type}, Top-K: {k}")
        
        # 检索
        if retriever_type == "faiss":
            results = self.retriever_manager.retrieve_with_faiss(question, k=k)
            documents = [doc for doc, score in results]
            scores = [score for doc, score in results]
        
        elif retriever_type == "bm25":
            documents = self.retriever_manager.retrieve_with_bm25(question, k=k)
            scores = None
        
        elif retriever_type == "hybrid":
            results = self.retriever_manager.hybrid_retrieve(question, k=k)
            documents = [doc for doc, score in results]
            scores = [score for doc, score in results]
        
        else:
            logger.error(f"不支持的检索器类型: {retriever_type}")
            return {
                'question': question,
                'answer': f"不支持的检索器类型: {retriever_type}",
                'sources': []
            }
        
        # 生成答案
        result = self.generator.generate_with_retrieval_analysis(
            question, 
            documents, 
            scores
        )
        
        result['retriever_type'] = retriever_type
        
        return result


if __name__ == "__main__":
    # 测试代码
    from document_loader import DocumentProcessor
    from vector_store import VectorStoreManager
    from retriever import RetrieverManager
    
    # 加载和索引文档
    processor = DocumentProcessor()
    chunks = processor.process_documents(config.DOCUMENTS_PATH)
    
    if chunks:
        # 初始化组件
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)
        
        retriever = RetrieverManager(vs_manager)
        retriever.setup_bm25(chunks)
        
        generator = AnswerGenerator()
        
        # 创建RAG流水线
        rag_pipeline = RAGPipeline(retriever, generator)
        
        # 测试查询
        question = "什么是人工智能？"
        result = rag_pipeline.query(question, retriever_type="faiss")
        
        print(f"\n问题: {result['question']}")
        print(f"\n答案:\n{result['answer']}")
        print(f"\n来源数量: {len(result['sources'])}")

