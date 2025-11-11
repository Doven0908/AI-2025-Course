"""向量数据库模块"""
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import config


class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(
        self, 
        embedding_model_name: str = None,
        db_type: str = None,
        db_path: str = None
    ):
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL
        self.db_type = db_type or config.VECTOR_DB_TYPE
        self.db_path = Path(db_path or config.VECTOR_DB_PATH)
        
        # 初始化嵌入模型
        logger.info(f"初始化嵌入模型: {self.embedding_model_name}")
        self.embeddings = self._init_embeddings()
        
        # 向量数据库
        self.vectorstore = None
        
    def _init_embeddings(self) -> Embeddings:
        """初始化嵌入模型"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("嵌入模型初始化成功")
            return embeddings
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """创建向量数据库"""
        if not documents:
            logger.error("文档列表为空，无法创建向量数据库")
            return
        
        logger.info(f"开始创建 {self.db_type} 向量数据库，共 {len(documents)} 个文档块")
        
        try:
            if self.db_type.lower() == "faiss":
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            elif self.db_type.lower() == "chroma":
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.db_path)
                )
            else:
                raise ValueError(f"不支持的向量数据库类型: {self.db_type}")
            
            logger.info("向量数据库创建成功")
            
        except Exception as e:
            logger.error(f"向量数据库创建失败: {e}")
            raise
    
    def save(self, index_name: str = "default") -> None:
        """保存向量数据库"""
        if self.vectorstore is None:
            logger.warning("向量数据库未初始化，无法保存")
            return
        
        try:
            if self.db_type.lower() == "faiss":
                save_path = self.db_path / index_name
                save_path.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(save_path))
                logger.info(f"FAISS 索引已保存到: {save_path}")
            
            elif self.db_type.lower() == "chroma":
                # Chroma 在创建时已经持久化
                logger.info(f"Chroma 数据库已持久化到: {self.db_path}")
        
        except Exception as e:
            logger.error(f"保存向量数据库失败: {e}")
            raise
    
    def load(self, index_name: str = "default") -> None:
        """加载向量数据库"""
        try:
            if self.db_type.lower() == "faiss":
                load_path = self.db_path / index_name
                if not load_path.exists():
                    logger.error(f"FAISS 索引不存在: {load_path}")
                    return
                
                self.vectorstore = FAISS.load_local(
                    str(load_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS 索引已加载: {load_path}")
            
            elif self.db_type.lower() == "chroma":
                if not self.db_path.exists():
                    logger.error(f"Chroma 数据库不存在: {self.db_path}")
                    return
                
                self.vectorstore = Chroma(
                    persist_directory=str(self.db_path),
                    embedding_function=self.embeddings
                )
                logger.info(f"Chroma 数据库已加载: {self.db_path}")
        
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None
    ) -> List[Document]:
        """相似度搜索"""
        if self.vectorstore is None:
            logger.error("向量数据库未初始化")
            return []
        
        k = k or config.TOP_K
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"检索到 {len(results)} 个相关文档")
            return results
        
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """带评分的相似度搜索"""
        if self.vectorstore is None:
            logger.error("向量数据库未初始化")
            return []
        
        k = k or config.TOP_K
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"检索到 {len(results)} 个相关文档（带评分）")
            return results
        
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []


if __name__ == "__main__":
    # 测试代码
    from document_loader import DocumentProcessor
    
    # 处理文档
    processor = DocumentProcessor()
    chunks = processor.process_documents(config.DOCUMENTS_PATH)
    
    if chunks:
        # 创建向量数据库
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)
        vs_manager.save()
        
        # 测试检索
        results = vs_manager.similarity_search("测试查询", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n结果 {i}:\n{doc.page_content[:200]}")

