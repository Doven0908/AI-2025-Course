"""完整的RAG系统评估脚本"""
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import pandas as pd
import config
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline
from src.evaluation import RAGEvaluator, FailureAnalyzer

# 配置日志
logger.add(
    config.LOGS_DIR / "evaluation.log",
    rotation="500 MB",
    level=config.LOG_LEVEL
)


def load_test_cases(test_file: str = None) -> list:
    """加载测试用例"""
    if test_file and Path(test_file).exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 默认测试用例
    return [
        {
            "question": "什么是人工智能？",
            "expected_keywords": ["人工智能", "AI", "计算机", "智能"],
            "ground_truth": "人工智能是计算机科学的一个分支..."
        },
        {
            "question": "机器学习有哪些应用？",
            "expected_keywords": ["机器学习", "应用", "图像识别", "自然语言"],
            "ground_truth": "机器学习应用广泛，包括图像识别、自然语言处理等..."
        },
        {
            "question": "深度学习和机器学习的区别是什么？",
            "expected_keywords": ["深度学习", "机器学习", "神经网络", "区别"],
            "ground_truth": "深度学习是机器学习的子集，使用神经网络..."
        }
    ]


def main():
    """主评估流程"""
    logger.info("=" * 70)
    logger.info("RAG系统完整评估流程")
    logger.info("=" * 70)
    
    # 1. 初始化系统
    logger.info("\n步骤1: 初始化RAG系统...")
    
    processor = DocumentProcessor()
    chunks = processor.process_documents(config.DOCUMENTS_PATH)
    
    if not chunks:
        logger.error("未找到任何文档！")
        return
    
    # 创建/加载向量索引
    vs_manager = VectorStoreManager()
    index_path = Path(config.VECTOR_DB_PATH) / "evaluation"
    
    if index_path.exists():
        logger.info("加载已有索引...")
        vs_manager.load("evaluation")
    else:
        logger.info("创建新索引...")
        vs_manager.create_vectorstore(chunks)
        vs_manager.save("evaluation")
    
    # 初始化检索器和生成器
    retriever_manager = RetrieverManager(vs_manager)
    retriever_manager.setup_bm25(chunks)
    
    generator = AnswerGenerator()
    rag_pipeline = RAGPipeline(retriever_manager, generator)
    
    logger.info("✅ 系统初始化完成")
    
    # 2. 加载测试用例
    logger.info("\n步骤2: 加载测试用例...")
    test_cases = load_test_cases()
    logger.info(f"加载了 {len(test_cases)} 个测试用例")
    
    # 3. 执行查询并收集结果
    logger.info("\n步骤3: 执行查询...")
    results = {
        'faiss': [],
        'bm25': [],
        'hybrid': []
    }
    
    for retriever_type in ['faiss', 'bm25', 'hybrid']:
        logger.info(f"\n使用 {retriever_type} 检索器...")
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case['question']
            logger.info(f"  查询 {i}/{len(test_cases)}: {question}")
            
            try:
                result = rag_pipeline.query(
                    question,
                    retriever_type=retriever_type
                )
                results[retriever_type].append(result)
            except Exception as e:
                logger.error(f"查询失败: {e}")
                continue
    
    # 4. 性能评估：延迟和吞吐量
    logger.info("\n步骤4: 性能评估...")
    evaluator = RAGEvaluator()
    
    test_questions = [case['question'] for case in test_cases]
    performance_metrics = evaluator.measure_performance(
        rag_pipeline,
        test_questions,
        retriever_types=['faiss', 'bm25', 'hybrid']
    )
    
    logger.info("✅ 性能评估完成")
    
    # 5. 检索指标评估
    logger.info("\n步骤5: 检索指标评估...")
    
    retrieval_metrics = {}
    for retriever_type in ['faiss', 'bm25', 'hybrid']:
        retrieved_docs = []
        relevant_docs = []
        
        for result in results[retriever_type]:
            # 提取检索到的文档内容
            docs = [source['content'] for source in result.get('sources', [])]
            retrieved_docs.append(docs)
            
            # 简单假设：如果答案包含预期关键词，则视为相关
            # 实际应用中需要人工标注
            relevant_docs.append(docs[:3])  # 假设前3个相关
        
        if retrieved_docs and relevant_docs:
            metrics = evaluator.calculate_retrieval_metrics(
                retrieved_docs,
                relevant_docs
            )
            retrieval_metrics[retriever_type] = metrics
    
    logger.info("✅ 检索指标评估完成")
    
    # 6. RAGAS评估（可选，需要配置LLM）
    logger.info("\n步骤6: RAGAS评估...")
    
    ragas_results = {}
    try:
        for retriever_type in ['faiss']:  # 示例只评估FAISS
            questions = []
            answers = []
            contexts = []
            
            for result in results[retriever_type]:
                questions.append(result['question'])
                answers.append(result['answer'])
                
                # 提取上下文
                context = [
                    source['content'] 
                    for source in result.get('sources', [])
                ]
                contexts.append(context)
            
            if questions and answers and contexts:
                dataset = evaluator.prepare_evaluation_dataset(
                    questions,
                    answers,
                    contexts
                )
                
                # 注意：RAGAS评估需要配置有效的LLM API
                # ragas_result = evaluator.evaluate_rag_system(dataset)
                # ragas_results[retriever_type] = ragas_result
                
                logger.info(f"准备了 {len(dataset)} 个样本用于RAGAS评估")
                logger.info("提示: RAGAS评估需要配置有效的LLM API")
    
    except Exception as e:
        logger.warning(f"RAGAS评估跳过: {e}")
    
    # 7. 失败案例分析
    logger.info("\n步骤7: 失败案例分析...")
    
    failure_analyzer = FailureAnalyzer()
    failure_cases = []
    
    for retriever_type in ['faiss']:
        for i, result in enumerate(results[retriever_type]):
            test_case = test_cases[i]
            
            failure_case = {
                'question': result['question'],
                'generated_answer': result['answer'],
                'expected_answer': test_case.get('ground_truth', ''),
                'retrieved_docs': [
                    source['content'] 
                    for source in result.get('sources', [])
                ],
                'relevant_docs': [],  # 需要人工标注
                'scores': [
                    source.get('score', 0) 
                    for source in result.get('sources', [])
                ]
            }
            failure_cases.append(failure_case)
    
    failure_analysis = failure_analyzer.batch_analyze_failures(failure_cases)
    logger.info("✅ 失败案例分析完成")
    
    # 8. 生成评估报告
    logger.info("\n步骤8: 生成评估报告...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_file = config.RESULTS_DIR / f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'performance_metrics': performance_metrics,
            'retrieval_metrics': retrieval_metrics,
            'ragas_results': ragas_results,
            'failure_analysis': failure_analysis
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细结果已保存: {results_file}")
    
    # 生成CSV报告
    report_file = config.RESULTS_DIR / f"evaluation_report_{timestamp}.csv"
    
    report_data = []
    
    # 性能指标
    for retriever, metrics in performance_metrics.items():
        for metric, value in metrics.items():
            report_data.append({
                'Category': 'Performance',
                'Retriever': retriever,
                'Metric': metric,
                'Value': value
            })
    
    # 检索指标
    for retriever, metrics in retrieval_metrics.items():
        for metric, value in metrics.items():
            report_data.append({
                'Category': 'Retrieval',
                'Retriever': retriever,
                'Metric': metric,
                'Value': value
            })
    
    df = pd.DataFrame(report_data)
    df.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"评估报告已保存: {report_file}")
    
    # 生成失败案例报告
    failure_report_file = config.RESULTS_DIR / f"failure_analysis_{timestamp}.csv"
    failure_df = failure_analyzer.generate_failure_report(
        failure_analysis,
        str(failure_report_file)
    )
    
    # 9. 打印摘要
    logger.info("\n" + "=" * 70)
    logger.info("评估摘要")
    logger.info("=" * 70)
    
    print("\n📊 性能指标:")
    for retriever, metrics in performance_metrics.items():
        print(f"\n  {retriever.upper()}:")
        print(f"    平均延迟: {metrics['avg_latency']:.3f}秒")
        print(f"    吞吐量: {metrics['throughput']:.2f} queries/sec")
    
    print("\n🔍 检索指标:")
    for retriever, metrics in retrieval_metrics.items():
        print(f"\n  {retriever.upper()}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    print("\n❌ 失败案例统计:")
    for error_type, count in failure_analysis['error_statistics'].items():
        print(f"  {error_type}: {count}")
    
    logger.info("\n✅ 评估完成！")
    logger.info(f"结果保存在: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()

