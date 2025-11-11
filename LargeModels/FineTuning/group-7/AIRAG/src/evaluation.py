"""评估模块：使用RAGAS进行自动化评估"""
from typing import List, Dict
import time
from datetime import datetime
import pandas as pd
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from datasets import Dataset
import config


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self):
        self.metrics = [
            faithfulness,           # 忠实度：答案是否基于上下文
            answer_relevancy,       # 答案相关性
            context_precision,      # 上下文精确度
            context_recall,         # 上下文召回率
        ]
    
    def prepare_evaluation_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dataset:
        """准备评估数据集"""
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }
        
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        dataset = Dataset.from_dict(data)
        logger.info(f"准备评估数据集，共 {len(questions)} 个样本")
        return dataset
    
    def evaluate_rag_system(
        self,
        dataset: Dataset,
        metrics: List = None
    ) -> Dict:
        """评估RAG系统"""
        metrics = metrics or self.metrics
        
        try:
            logger.info("开始RAGAS评估...")
            start_time = time.time()
            
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"评估完成，耗时: {elapsed_time:.2f}秒")
            
            return {
                'scores': result,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]]
    ) -> Dict:
        """计算检索指标：召回率和精确度"""
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'mrr': [],  # Mean Reciprocal Rank
            'hit_rate': []
        }
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_set = set(retrieved)
            relevant_set = set(relevant)
            
            # 计算交集
            hits = retrieved_set & relevant_set
            
            # 精确度
            precision = len(hits) / len(retrieved_set) if retrieved_set else 0
            metrics['precision'].append(precision)
            
            # 召回率
            recall = len(hits) / len(relevant_set) if relevant_set else 0
            metrics['recall'].append(recall)
            
            # F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics['f1'].append(f1)
            
            # Hit Rate
            hit_rate = 1 if hits else 0
            metrics['hit_rate'].append(hit_rate)
            
            # MRR
            mrr = 0
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    mrr = 1 / i
                    break
            metrics['mrr'].append(mrr)
        
        # 计算平均值
        avg_metrics = {
            f'avg_{key}': sum(values) / len(values) if values else 0
            for key, values in metrics.items()
        }
        
        logger.info(f"检索指标: {avg_metrics}")
        return avg_metrics
    
    def measure_performance(
        self,
        rag_pipeline,
        test_questions: List[str],
        retriever_types: List[str] = None
    ) -> Dict:
        """测量性能指标：延迟和吞吐量"""
        retriever_types = retriever_types or ['faiss', 'bm25', 'hybrid']
        
        performance_results = {}
        
        for retriever_type in retriever_types:
            latencies = []
            
            logger.info(f"测试 {retriever_type} 检索器性能...")
            
            for question in test_questions:
                start_time = time.time()
                try:
                    result = rag_pipeline.query(question, retriever_type=retriever_type)
                    latency = time.time() - start_time
                    latencies.append(latency)
                except Exception as e:
                    logger.error(f"查询失败: {e}")
                    continue
            
            if latencies:
                performance_results[retriever_type] = {
                    'avg_latency': sum(latencies) / len(latencies),
                    'min_latency': min(latencies),
                    'max_latency': max(latencies),
                    'throughput': len(latencies) / sum(latencies),  # queries per second
                    'num_queries': len(latencies)
                }
        
        logger.info(f"性能测试完成")
        return performance_results
    
    def generate_evaluation_report(
        self,
        ragas_results: Dict,
        retrieval_metrics: Dict,
        performance_metrics: Dict,
        output_path: str = None
    ) -> pd.DataFrame:
        """生成评估报告"""
        report_data = []
        
        # RAGAS评分
        if 'scores' in ragas_results:
            for metric, score in ragas_results['scores'].items():
                report_data.append({
                    'Category': 'RAGAS',
                    'Metric': metric,
                    'Value': score
                })
        
        # 检索指标
        for metric, value in retrieval_metrics.items():
            report_data.append({
                'Category': 'Retrieval',
                'Metric': metric,
                'Value': value
            })
        
        # 性能指标
        for retriever, metrics in performance_metrics.items():
            for metric, value in metrics.items():
                report_data.append({
                    'Category': f'Performance-{retriever}',
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(report_data)
        
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"评估报告已保存到: {output_path}")
        
        return df


class FailureAnalyzer:
    """失败案例分析器：误差归因"""
    
    def __init__(self):
        self.failure_types = {
            'retrieval_error': '检索错误：未检索到相关文档',
            'ranking_error': '排序错误：相关文档排序靠后',
            'generation_error': '生成错误：生成的答案不准确',
            'context_error': '上下文错误：文档分块不当',
            'low_confidence': '低置信度：检索文档相关度过低',
            'no_answer': '无法回答：文档中没有相关信息'
        }
    
    def analyze_realtime(self, question: str, answer: str, 
                        retrieved_docs: list, scores: list = None):
        """实时分析查询质量（用于Web界面）"""
        analysis = {
            'quality_score': 0.0,  # 0-100分
            'issues': [],
            'warnings': [],
            'suggestions': [],
            'error_type': None,
            'severity': 'none'  # none, low, medium, high
        }
        
        # 检查1：检索文档数量
        if not retrieved_docs or len(retrieved_docs) == 0:
            analysis['issues'].append({
                'type': 'retrieval_error',
                'message': '❌ 检索错误：未检索到任何文档',
                'severity': 'high'
            })
            analysis['error_type'] = 'retrieval_error'
            analysis['severity'] = 'high'
            analysis['quality_score'] = 0
            analysis['suggestions'].append('建议：检查文档库是否包含相关内容，或尝试不同的检索器')
            return analysis
        
        # 检查2：检索相关度（使用分数）
        if scores and len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            # 相关度过低
            if max_score < 0.3:
                analysis['issues'].append({
                    'type': 'low_confidence',
                    'message': f'⚠️ 低置信度：最高相关度仅 {max_score:.2f}（建议>0.5）',
                    'severity': 'high'
                })
                analysis['error_type'] = 'retrieval_error'
                analysis['severity'] = 'high'
                analysis['quality_score'] = max(0, max_score * 100)
                analysis['suggestions'].append('建议：文档库可能不包含相关信息，尝试换用 BM25 或 Hybrid 检索器')
            
            elif max_score < 0.5:
                analysis['warnings'].append({
                    'type': 'low_confidence',
                    'message': f'⚠️ 相关度一般：最高相关度 {max_score:.2f}',
                    'severity': 'medium'
                })
                analysis['severity'] = 'medium'
                analysis['quality_score'] = max_score * 100
                analysis['suggestions'].append('建议：可以尝试重新表述问题，或使用 Hybrid 检索器')
            
            elif max_score < 0.7:
                analysis['warnings'].append({
                    'type': 'medium_confidence',
                    'message': f'✓ 相关度可接受：最高相关度 {max_score:.2f}',
                    'severity': 'low'
                })
                analysis['severity'] = 'low'
                analysis['quality_score'] = max_score * 100
            
            else:
                # 相关度良好
                analysis['quality_score'] = max_score * 100
            
            # 检查3：排序问题（相关文档是否靠前）
            if len(scores) >= 3 and scores[0] < 0.5 and max_score > 0.7:
                # 最高分在后面，但第一个分数低
                best_position = scores.index(max_score) + 1
                analysis['warnings'].append({
                    'type': 'ranking_error',
                    'message': f'⚠️ 排序问题：最相关文档在第 {best_position} 位',
                    'severity': 'medium'
                })
                if analysis['severity'] == 'none':
                    analysis['severity'] = 'medium'
                analysis['suggestions'].append('建议：尝试使用 Hybrid 检索器以改善排序')
        
        # 检查4：答案质量（基于关键词匹配）
        answer_lower = answer.lower()
        
        # 检测是否是拒绝回答的标志
        refuse_keywords = [
            '无法回答', '不能回答', '没有相关信息', '没有找到',
            '根据提供的信息无法', '文档中没有', '无法确定',
            '对不起', '抱歉', 'sorry', 'cannot answer'
        ]
        
        if any(keyword in answer_lower for keyword in refuse_keywords):
            analysis['issues'].append({
                'type': 'no_answer',
                'message': '❌ 无法回答：LLM明确表示无法回答此问题',
                'severity': 'high'
            })
            if analysis['error_type'] is None:
                analysis['error_type'] = 'generation_error'
            if analysis['severity'] in ['none', 'low']:
                analysis['severity'] = 'high'
            analysis['quality_score'] = min(analysis['quality_score'], 30)
            analysis['suggestions'].append('建议：问题可能超出文档库范围，请添加相关文档或重新表述问题')
        
        # 检查5：答案长度（过短可能是生成问题）
        if len(answer) < 20:
            analysis['warnings'].append({
                'type': 'generation_error',
                'message': '⚠️ 答案过短：可能生成质量不佳',
                'severity': 'medium'
            })
            if analysis['severity'] == 'none':
                analysis['severity'] = 'medium'
            analysis['suggestions'].append('建议：检查LLM配置，或尝试更详细的问题')
        
        # 检查6：是否包含引用
        if '[来源' not in answer and '来源' not in answer.lower():
            analysis['warnings'].append({
                'type': 'generation_error',
                'message': '⚠️ 缺少引用：答案未包含来源标注',
                'severity': 'low'
            })
            if analysis['severity'] == 'none':
                analysis['severity'] = 'low'
        
        # 设置默认质量分数（如果还没设置）
        if analysis['quality_score'] == 0 and not analysis['issues']:
            analysis['quality_score'] = 75  # 默认良好
        
        # 最终质量评级
        if analysis['quality_score'] >= 80:
            analysis['quality_level'] = '优秀'
            analysis['quality_emoji'] = '🟢'
        elif analysis['quality_score'] >= 60:
            analysis['quality_level'] = '良好'
            analysis['quality_emoji'] = '🟡'
        elif analysis['quality_score'] >= 40:
            analysis['quality_level'] = '一般'
            analysis['quality_emoji'] = '🟠'
        else:
            analysis['quality_level'] = '较差'
            analysis['quality_emoji'] = '🔴'
        
        return analysis
    
    def analyze_failure(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        scores: List[float] = None
    ) -> Dict:
        """分析单个失败案例"""
        analysis = {
            'question': question,
            'generated_answer': generated_answer,
            'expected_answer': expected_answer,
            'errors': []
        }
        
        # 检查检索错误
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        if not (retrieved_set & relevant_set):
            analysis['errors'].append({
                'type': 'retrieval_error',
                'description': '未检索到任何相关文档',
                'severity': 'high'
            })
        
        # 检查排序错误
        if scores:
            relevant_positions = [
                i for i, doc in enumerate(retrieved_docs) 
                if doc in relevant_set
            ]
            
            if relevant_positions and min(relevant_positions) > 2:
                analysis['errors'].append({
                    'type': 'ranking_error',
                    'description': f'相关文档排序靠后，最佳位置: {min(relevant_positions) + 1}',
                    'severity': 'medium'
                })
        
        # 检查生成错误（简单的文本相似度）
        if generated_answer and expected_answer:
            similarity = self._simple_similarity(generated_answer, expected_answer)
            if similarity < 0.5:
                analysis['errors'].append({
                    'type': 'generation_error',
                    'description': f'生成答案与期望答案相似度低: {similarity:.2f}',
                    'severity': 'high'
                })
        
        # 检查上下文错误
        if retrieved_docs:
            avg_doc_length = sum(len(doc) for doc in retrieved_docs) / len(retrieved_docs)
            if avg_doc_length < 50 or avg_doc_length > 2000:
                analysis['errors'].append({
                    'type': 'context_error',
                    'description': f'文档块长度异常: {avg_doc_length:.0f}',
                    'severity': 'low'
                })
        
        return analysis
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def batch_analyze_failures(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """批量分析失败案例"""
        results = []
        error_stats = {error_type: 0 for error_type in self.failure_types.keys()}
        
        for case in test_cases:
            analysis = self.analyze_failure(
                question=case['question'],
                generated_answer=case['generated_answer'],
                expected_answer=case.get('expected_answer', ''),
                retrieved_docs=case['retrieved_docs'],
                relevant_docs=case.get('relevant_docs', []),
                scores=case.get('scores')
            )
            
            results.append(analysis)
            
            # 统计错误类型
            for error in analysis['errors']:
                error_stats[error['type']] += 1
        
        logger.info(f"失败案例分析完成，共 {len(test_cases)} 个案例")
        logger.info(f"错误统计: {error_stats}")
        
        return {
            'analyses': results,
            'error_statistics': error_stats,
            'total_cases': len(test_cases)
        }
    
    def generate_failure_report(
        self,
        failure_analysis: Dict,
        output_path: str = None
    ) -> pd.DataFrame:
        """生成失败案例分析报告"""
        report_data = []
        
        for analysis in failure_analysis['analyses']:
            for error in analysis['errors']:
                report_data.append({
                    'Question': analysis['question'][:100],
                    'Error Type': error['type'],
                    'Description': error['description'],
                    'Severity': error['severity']
                })
        
        df = pd.DataFrame(report_data)
        
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"失败案例报告已保存到: {output_path}")
        
        return df


if __name__ == "__main__":
    # 测试代码
    evaluator = RAGEvaluator()
    
    # 示例数据
    questions = ["什么是人工智能？", "机器学习的应用有哪些？"]
    answers = ["人工智能是...", "机器学习应用包括..."]
    contexts = [
        ["人工智能定义文档1", "人工智能定义文档2"],
        ["机器学习应用文档1", "机器学习应用文档2"]
    ]
    
    dataset = evaluator.prepare_evaluation_dataset(questions, answers, contexts)
    print(f"数据集准备完成: {len(dataset)} 个样本")

