from functools import lru_cache

from llm_hallucination_evaluate.utils.config import TypeEnum

@lru_cache(maxsize=None)
def get_model_func(type: TypeEnum, model: str):
    """
    根据type和model获取对应的模型调用函数
    
    Args:
        type: 模型类型，online或offline
        model: 模型名称ID
        
    Returns:
        Callable: 对应的模型调用函数
    """
    if type == TypeEnum.ONLINE:
        from llm_hallucination_evaluate.solver.online_llm import get_online_model_func
        return get_online_model_func(model)
    elif type == TypeEnum.OFFLINE:
        from llm_hallucination_evaluate.solver.offline_llm import get_offline_model_func
        return get_offline_model_func(model)
    else:
        raise ValueError(f"不支持的模型类型: {type}")


