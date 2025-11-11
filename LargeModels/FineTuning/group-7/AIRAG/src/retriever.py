"""检索器模块：支持多种检索策略"""
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from loguru import logger
import config


class RetrieverManager:
    """检索器管理器：支持多种检索方法"""
    
    def __init__(self, vector_store_manager=None):
        self.vector_store_manager = vector_store_manager
        self.bm25_retriever = None
    
    def setup_bm25(self, documents: List[Document]) -> None:
        """设置BM25检索器"""
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = config.TOP_K
            logger.info(f"BM25检索器初始化成功，文档数: {len(documents)}")
        except Exception as e:
            logger.error(f"BM25检索器初始化失败: {e}")
            raise
    
    def retrieve_with_faiss(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """使用FAISS向量检索"""
        if self.vector_store_manager is None:
            logger.error("向量数据库管理器未初始化")
            return []
        
        k = k or config.TOP_K
        results = self.vector_store_manager.similarity_search_with_score(query, k=k)
        logger.info(f"FAISS检索完成，返回 {len(results)} 个结果")
        return results
    
    def retrieve_with_bm25(
        self, 
        query: str, 
        k: int = None
    ) -> List[Document]:
        """使用BM25检索"""
        if self.bm25_retriever is None:
            logger.error("BM25检索器未初始化")
            return []
        
        k = k or config.TOP_K
        self.bm25_retriever.k = k
        
        try:
            results = self.bm25_retriever.get_relevant_documents(query)
            logger.info(f"BM25检索完成，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []
    
    def hybrid_retrieve(
        self, 
        query: str, 
        k: int = None,
        faiss_weight: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """混合检索：FAISS + BM25"""
        k = k or config.TOP_K
        
        # FAISS检索
        faiss_results = self.retrieve_with_faiss(query, k=k*2)
        faiss_docs = {doc.page_content: (doc, score) for doc, score in faiss_results}
        
        # BM25检索
        bm25_results = self.retrieve_with_bm25(query, k=k*2)
        bm25_docs = {doc.page_content: doc for doc in bm25_results}
        
        # 合并结果并重新评分
        all_docs = {}
        
        # 添加FAISS结果
        for content, (doc, score) in faiss_docs.items():
            all_docs[content] = {
                'doc': doc,
                'faiss_score': score,
                'bm25_score': 0.0
            }
        
        # 添加BM25结果
        for content, doc in bm25_docs.items():
            if content not in all_docs:
                all_docs[content] = {
                    'doc': doc,
                    'faiss_score': float('inf'),  # 最低分
                    'bm25_score': 1.0
                }
            else:
                all_docs[content]['bm25_score'] = 1.0
        
        # 归一化并混合评分
        results = []
        for content, data in all_docs.items():
            # 简单的混合策略：加权平均
            # FAISS分数越小越好，需要反转
            faiss_normalized = 1.0 / (1.0 + data['faiss_score'])
            combined_score = (
                faiss_weight * faiss_normalized + 
                (1 - faiss_weight) * data['bm25_score']
            )
            results.append((data['doc'], combined_score))
        
        # 按分数排序并返回top-k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]
        
        logger.info(f"混合检索完成，返回 {len(results)} 个结果")
        return results
    
    def compare_retrievers(
        self, 
        query: str, 
        k: int = None
    ) -> dict:
        """对比不同检索器的结果"""
        k = k or config.TOP_K
        
        comparison = {
            'query': query,
            'faiss': [],
            'bm25': [],
            'hybrid': []
        }
        
        # FAISS检索
        faiss_results = self.retrieve_with_faiss(query, k=k)
        comparison['faiss'] = [
            {
                'content': doc.page_content[:200],
                'score': float(score),
                'metadata': doc.metadata
            }
            for doc, score in faiss_results
        ]
        
        # BM25检索
        bm25_results = self.retrieve_with_bm25(query, k=k)
        comparison['bm25'] = [
            {
                'content': doc.page_content[:200],
                'metadata': doc.metadata
            }
            for doc in bm25_results
        ]
        
        # 混合检索
        hybrid_results = self.hybrid_retrieve(query, k=k)
        comparison['hybrid'] = [
            {
                'content': doc.page_content[:200],
                'score': float(score),
                'metadata': doc.metadata
            }
            for doc, score in hybrid_results
        ]
        
        return comparison


if __name__ == "__main__":
    # 测试代码
    from document_loader import DocumentProcessor
    from vector_store import VectorStoreManager
    
    # 加载文档
    processor = DocumentProcessor()
    chunks = processor.process_documents(config.DOCUMENTS_PATH)
    
    if chunks:
        # 初始化向量数据库
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)
        
        # 初始化检索器
        retriever = RetrieverManager(vs_manager)
        retriever.setup_bm25(chunks)
        
        # 对比检索器
        query = "什么是人工智能？"
        results = retriever.compare_retrievers(query, k=3)
        
        print(f"\n查询: {query}\n")
        print(f"FAISS结果数: {len(results['faiss'])}")
        print(f"BM25结果数: {len(results['bm25'])}")
        print(f"混合结果数: {len(results['hybrid'])}")

