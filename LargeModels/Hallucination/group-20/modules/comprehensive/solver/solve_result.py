
from pydantic import BaseModel, RootModel, Field

from llm_hallucination_evaluate.dataset.base import Context
from llm_hallucination_evaluate.dataset.base import DataItem

class SolveResult(BaseModel):
    dataitem: DataItem = Field(..., description="对应的数据项")
    answer: str = Field(..., description="求解得到的答案")
    metadata: dict = Field(default_factory=dict, description="求解过程中的元数据")



