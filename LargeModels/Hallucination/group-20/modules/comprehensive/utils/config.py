import yaml
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class MethodEnum(str, Enum):
    DIRECT_ANSWER = "direct-answer"
    USE_RAG = "use-rag"
    USE_SELF_RAG = "use-self-rag"
    USE_COVE = "use-cove"
    USE_SELFCHECKGPT = "use-selfcheckgpt"

class TypeEnum(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"

class EvaluatorMethodEnum(str, Enum):
    RAGAS = "ragas"


class DatasetConfig(BaseModel):
    name: str = Field(..., description="数据集名称")
    number: Optional[int] = Field(None, description="取样数量，None表示选取全部")


class SolverConfig(BaseModel):
    method: MethodEnum = Field(..., description="求解方法")
    type: Optional[TypeEnum] = Field(None, description="求解类型，online或offline")
    model: str = Field(..., description="模型名称ID")
    rag_docs_number: Optional[int] = Field(None, description="检索文档数量")
    extra_args: dict = Field(default_factory=dict, description="额外参数")

    @validator('type', pre=True, always=True)
    def validate_type(cls, v, values):
        """对于某些方法，type字段是可选的"""
        method = values.get('method')
        if method in [MethodEnum.USE_SELF_RAG, MethodEnum.USE_SELFCHECKGPT]:
            return None
        return v



class EvaluatorConfig(BaseModel):
    method: EvaluatorMethodEnum = Field(..., description="评估方法")
    model: str = Field(..., description="评估所使用的模型")
    metrics: List[str] = Field(..., description="评估指标列表")

class ExperimentConfig(BaseModel):
    dataset: List[DatasetConfig] = Field(..., description="数据集配置列表")
    solvers: List[SolverConfig] = Field(..., description="求解器配置列表")
    evaluators: List[EvaluatorConfig] = Field(..., description="评估器配置列表")
    extra: Dict[str, Any] = Field(default_factory=dict, description="额外配置项")

def load_config(config_path: str) -> ExperimentConfig:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ExperimentConfig: 配置对象
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        # 验证并创建配置对象
        config = ExperimentConfig(**config_data)
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML解析错误: {e}")
    except Exception as e:
        raise ValueError(f"配置验证错误: {e}")

def print_config_summary(config: ExperimentConfig):
    """打印配置摘要"""
    print("=" * 50)
    print("实验配置摘要")
    print("=" * 50)
    
    print(f"\n📊 数据集配置 ({len(config.dataset)} 个):")
    for i, dataset in enumerate(config.dataset, 1):
        num_info = "全部数据" if dataset.number is None else f"{dataset.number} 条样本"
        print(f"  {i}. {dataset.name} - {num_info}")
    
    print(f"\n🤖 求解器配置 ({len(config.solvers)} 个):")
    for i, solver in enumerate(config.solvers, 1):
        type_info = f" ({solver.type})" if solver.type else ""
        rag_info = f" - 检索 {solver.rag_docs_number} 篇文档" if solver.rag_docs_number else ""
        extra_info = f" - 额外参数: {solver.extra_args}" if solver.extra_args else ""
        print(f"  {i}. {solver.method}{type_info} - {solver.model}{rag_info}{extra_info}")
    
    print(f"\n📈 评估器配置 ({len(config.evaluators)} 个):")
    for i, evaluator in enumerate(config.evaluators, 1):
        print(f"  {i}. {evaluator.method} - {evaluator.model}")
        print(f"     评估指标: {', '.join(evaluator.metrics)}")




config_path = "config.yaml"  
config = load_config(config_path)


# 加载配置
print("✅ 配置加载成功！")
# 打印配置摘要
print_config_summary(config)


