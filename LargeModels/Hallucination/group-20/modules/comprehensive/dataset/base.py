import json
import random
import jsonlines
from typing import Literal, Optional
from pydantic import BaseModel, Field, RootModel
from pydantic.main import TupleGenerator

from llm_hallucination_evaluate.utils.config import ExperimentConfig

class Context(BaseModel):
    title: str = Field(..., description="上下文标题")
    content: str = Field(..., description="上下文内容")
    score: Optional[float] = Field(None, description="上下文相关性评分")

class DataItem(BaseModel):

    question: str = Field(..., description="问题")
    answer: Optional[str] = Field(default=None, description="参考答案")
    contexts: list[Context] = Field(..., description="上下文信息")
    meta: dict = Field(default_factory=dict, description="额外元数据")    


class DataSet(BaseModel):

    name: str = Field(..., description="数据集名称")
    type: Literal["long-form", "mcq"] = Field(..., description="数据集类型")
    items: list[DataItem] = Field(..., description="数据集条目列表")

    # 默认配置
    _max_contexts: int = 10

    def __init__(self, **data):
        super().__init__(**data)
        # 对每个数据项的上下文进行裁剪
        
        for item in self.items:
            if len(item.contexts) > self._max_contexts:
                item.contexts = item.contexts[:self._max_contexts]

class MutiDataSet(RootModel):

    root: list[DataSet] = Field(..., description="多个数据集列表")

    def __len__(self) -> int:
        return len(self.root)
    
    def __getitem__(self, index: int) -> DataSet:
        return self.root[index]
    
    def __iter__(self):
        return iter(self.root)

    @classmethod
    def load_from_config(cls, config: ExperimentConfig) -> "MutiDataSet":
        """
        根据配置加载多个数据集
        
        Args:
            config: 实验配置对象
            
        Returns:
            MutiDataSet: 加载的数据集对象
        """
        datasets = []
        for dataset_cfg in config.dataset:
            if dataset_cfg.name == "asqa":
                dataset = load_asqa_dataset(number=dataset_cfg.number)
                datasets.append(dataset)
            elif dataset_cfg.name == "factscore":
                dataset = load_arc_challenge_dataset(number=dataset_cfg.number)
                datasets.append(dataset)
            else:
                raise ValueError(f"不支持的数据集名称: {dataset_cfg.name}")
            
        return cls(root=datasets)

ASQA_JSON_FILE = './data/asqa_eval_gtr_top100.json'
FACTSCORE_JSONL_FILE = './data/factscore_unlabeled_alpaca_13b_retrieval.jsonl'

def load_asqa_dataset(number: int | None = None) -> DataSet:
    """
    加载ASQA数据集的示例函数
    
    Args:
        number: 要加载的数据条目数量，None表示加载全部
        
    Returns:
        DataSet: 加载的数据集对象
    """

    with open(ASQA_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.shuffle(data)

    items = []
    for entry in data:
        docs = entry['docs']
        contexts = [Context(title=doc['title'], content=doc['text'], score=doc['score']) for doc in docs]
        
        item = DataItem(
            question=entry['question'],
            answer=entry['answer'],
            contexts=contexts,
            meta={'dataset': 'ASQA'}
        )
        items.append(item)
        if number is not None and len(items) >= number:
            break
    
    return DataSet(name="ASQA", items=items, type="long-form")


def load_arc_challenge_dataset(number: int | None = None) -> DataSet:
    """
    加载FactScore数据集的示例函数
    
    Args:
        number: 要加载的数据条目数量，None表示加载全部
        
    Returns:
        DataSet: 加载的数据集对象
    """

    items = []

    datas = []
    with open(FACTSCORE_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            datas.append(json.loads(line))
    random.shuffle(datas)
    for i, entry in enumerate(datas):
        contexts = [Context(title=doc['title'], content=doc['text'], score=doc['score']) for doc in entry['ctxs']]
        item = DataItem(
            question=entry['input'],
            contexts=contexts,
            meta={'dataset': 'FactScore'}
        )
        items.append(item)
        if number is not None and len(items) >= number:
            break

    return DataSet(name="FactScore", items=items, type="long-form")


