"""快速测试脚本：验证系统各组件是否正常工作"""
import sys
from pathlib import Path
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO")

def test_imports():
    """测试模块导入"""
    logger.info("测试1: 模块导入...")
    try:
        import config
        from src.document_loader import DocumentProcessor
        from src.vector_store import VectorStoreManager
        from src.retriever import RetrieverManager
        from src.generator import AnswerGenerator, RAGPipeline
        from src.evaluation import RAGEvaluator, FailureAnalyzer
        logger.success("✓ 所有模块导入成功")
        return True
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        return False


def test_document_processing():
    """测试文档处理"""
    logger.info("测试2: 文档处理...")
    try:
        from src.document_loader import DocumentProcessor
        import config
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        
        # 检查示例文档
        docs_path = Path(config.DOCUMENTS_PATH)
        if not docs_path.exists() or not list(docs_path.glob("*.txt")):
            logger.warning("⚠ 未找到示例文档，跳过文档加载测试")
            return True
        
        chunks = processor.process_documents(str(docs_path))
        
        if chunks:
            logger.success(f"✓ 文档处理成功，共 {len(chunks)} 个文档块")
            logger.info(f"  首个文档块预览: {chunks[0].page_content[:100]}...")
            return True
        else:
            logger.warning("⚠ 未加载到文档块")
            return True
    
    except Exception as e:
        logger.error(f"✗ 文档处理失败: {e}")
        return False


def test_embeddings():
    """测试嵌入模型"""
    logger.info("测试3: 嵌入模型...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        import config
        
        logger.info(f"  加载嵌入模型: {config.EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 测试编码
        test_text = "这是一个测试文本"
        vector = embeddings.embed_query(test_text)
        
        logger.success(f"✓ 嵌入模型工作正常，向量维度: {len(vector)}")
        return True
    
    except Exception as e:
        logger.error(f"✗ 嵌入模型测试失败: {e}")
        logger.info("  提示: 首次运行需要下载模型，可能需要较长时间")
        return False


def test_vector_store():
    """测试向量数据库"""
    logger.info("测试4: 向量数据库...")
    try:
        from src.document_loader import DocumentProcessor
        from src.vector_store import VectorStoreManager
        from langchain.schema import Document
        import config
        
        # 创建测试文档
        test_docs = [
            Document(page_content="人工智能是计算机科学的一个分支。", metadata={"source": "test"}),
            Document(page_content="机器学习是人工智能的子领域。", metadata={"source": "test"}),
            Document(page_content="深度学习使用神经网络。", metadata={"source": "test"}),
        ]
        
        # 创建向量数据库
        vs_manager = VectorStoreManager(db_type="faiss")
        vs_manager.create_vectorstore(test_docs)
        
        # 测试搜索
        results = vs_manager.similarity_search("什么是AI", k=2)
        
        logger.success(f"✓ 向量数据库工作正常，检索到 {len(results)} 个结果")
        return True
    
    except Exception as e:
        logger.error(f"✗ 向量数据库测试失败: {e}")
        return False


def test_bm25():
    """测试BM25检索器"""
    logger.info("测试5: BM25检索器...")
    try:
        from src.retriever import RetrieverManager
        from langchain.schema import Document
        
        test_docs = [
            Document(page_content="人工智能是计算机科学的分支", metadata={"source": "test"}),
            Document(page_content="机器学习是AI的重要技术", metadata={"source": "test"}),
        ]
        
        retriever = RetrieverManager()
        retriever.setup_bm25(test_docs)
        
        results = retriever.retrieve_with_bm25("人工智能", k=1)
        
        logger.success(f"✓ BM25检索器工作正常，检索到 {len(results)} 个结果")
        return True
    
    except Exception as e:
        logger.error(f"✗ BM25检索器测试失败: {e}")
        return False


def test_llm_connection():
    """测试LLM连接（可选）"""
    logger.info("测试6: LLM连接...")
    try:
        from src.generator import AnswerGenerator
        import config
        
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_api_key_here":
            logger.warning("⚠ 未配置LLM API，跳过测试（不影响其他功能）")
            logger.info("  提示: 编辑 .env 文件配置 OPENAI_API_KEY")
            return True
        
        generator = AnswerGenerator()
        logger.success("✓ LLM初始化成功")
        return True
    
    except Exception as e:
        logger.warning(f"⚠ LLM连接测试失败: {e}")
        logger.info("  提示: 如果不使用生成功能，此错误可以忽略")
        return True


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("RAG问答系统 - 系统测试")
    logger.info("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("文档处理", test_document_processing),
        ("嵌入模型", test_embeddings),
        ("向量数据库", test_vector_store),
        ("BM25检索器", test_bm25),
        ("LLM连接", test_llm_connection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"测试 '{name}' 出现异常: {e}")
            results.append((name, False))
        
        logger.info("")  # 空行
    
    # 总结
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {name}: {status}")
    
    logger.info(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.success("\n🎉 所有测试通过！系统运行正常。")
        logger.info("\n下一步:")
        logger.info("  1. 将文档放入 data/documents/ 目录")
        logger.info("  2. 运行 'python run.py --mode web' 启动Web界面")
        logger.info("  3. 或运行 'python run.py --mode cli' 启动命令行界面")
    elif passed >= total * 0.7:
        logger.warning("\n⚠ 大部分测试通过，系统基本可用。")
        logger.info("请检查失败的测试项，某些功能可能受限。")
    else:
        logger.error("\n❌ 多个测试失败，请检查环境配置。")
        logger.info("\n故障排除:")
        logger.info("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        logger.info("  2. 检查网络连接（首次运行需要下载模型）")
        logger.info("  3. 查看详细错误信息并根据提示解决")
        logger.info("  4. 参考文档: docs/USAGE.md")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

