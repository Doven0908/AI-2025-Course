"""一键运行脚本：索引-检索-生成完整流程"""
import argparse
from pathlib import Path
from loguru import logger
import config
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline

# 配置日志
logger.add(
    config.LOGS_DIR / "run.log",
    rotation="500 MB",
    level=config.LOG_LEVEL
)


def main():
    """主流程"""
    parser = argparse.ArgumentParser(description="RAG问答系统 - 一键运行")
    parser.add_argument(
        '--mode',
        type=str,
        default='index',
        choices=['index', 'query', 'eval', 'web', 'cli'],
        help='运行模式: index(索引), query(查询), eval(评估), web(Web界面), cli(命令行)'
    )
    parser.add_argument(
        '--documents',
        type=str,
        default=config.DOCUMENTS_PATH,
        help='文档目录路径'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='查询问题'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        default='faiss',
        choices=['faiss', 'bm25', 'hybrid'],
        help='检索器类型'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'index':
        # 索引模式：加载文档并创建索引
        logger.info("=" * 50)
        logger.info("索引模式：加载文档并创建向量索引")
        logger.info("=" * 50)
        
        # 1. 加载和分块文档
        logger.info("步骤1: 加载和分块文档...")
        processor = DocumentProcessor()
        chunks = processor.process_documents(args.documents)
        
        if not chunks:
            logger.error("未找到任何文档！")
            return
        
        logger.info(f"成功加载 {len(chunks)} 个文档块")
        
        # 2. 创建向量索引
        logger.info("步骤2: 创建向量索引...")
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)
        vs_manager.save("default")
        
        logger.info("✅ 索引创建完成！")
    
    elif args.mode == 'query':
        # 查询模式
        if not args.question:
            logger.error("请提供查询问题 (--question)")
            return
        
        logger.info("=" * 50)
        logger.info("查询模式：执行RAG查询")
        logger.info("=" * 50)
        
        # 1. 加载文档和索引
        logger.info("加载文档和索引...")
        processor = DocumentProcessor()
        chunks = processor.process_documents(args.documents)
        
        vs_manager = VectorStoreManager()
        vs_manager.load("default")
        
        # 2. 初始化检索器和生成器
        logger.info("初始化检索器和生成器...")
        retriever = RetrieverManager(vs_manager)
        retriever.setup_bm25(chunks)
        
        generator = AnswerGenerator()
        rag_pipeline = RAGPipeline(retriever, generator)
        
        # 3. 执行查询
        logger.info(f"查询问题: {args.question}")
        result = rag_pipeline.query(
            args.question,
            retriever_type=args.retriever
        )
        
        # 4. 输出结果
        print("\n" + "=" * 50)
        print(f"问题: {result['question']}")
        print("=" * 50)
        print(f"\n答案:\n{result['answer']}")
        print("\n" + "=" * 50)
        print("参考来源:")
        print("=" * 50)
        
        for source in result.get('sources', []):
            print(f"\n[来源 {source['index']}]")
            print(f"内容: {source['content'][:200]}...")
            print(f"文件: {source['metadata'].get('source', '未知')}")
            if 'score' in source:
                print(f"相关度: {source['score']:.4f}")
    
    elif args.mode == 'eval':
        # 评估模式
        logger.info("=" * 50)
        logger.info("评估模式：运行完整评估流程")
        logger.info("=" * 50)
        
        from evaluate_rag import main as eval_main
        eval_main()
    
    elif args.mode == 'web':
        # Web界面模式
        logger.info("启动Web界面...")
        import web_demo
        demo = web_demo.create_web_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    
    elif args.mode == 'cli':
        # CLI模式
        logger.info("启动CLI界面...")
        from cli_demo import RAGCLIDemo
        cli_demo = RAGCLIDemo()
        if cli_demo.setup(args.documents):
            cli_demo.interactive_mode()


if __name__ == "__main__":
    main()

