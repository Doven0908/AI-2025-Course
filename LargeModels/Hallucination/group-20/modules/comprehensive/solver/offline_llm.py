from huggingface_hub import login
from vllm import LLM, SamplingParams

login(token="xxx")

def get_offline_model_func(model: str):
    """
    根据模型名称ID获取对应的离线模型调用函数
    
    Args:
        model: 模型名称ID
        
    Returns:
        Callable: 对应的离线模型调用函数
    """
    llm = LLM(model=model)

    def call_offline_model(prompt: str) -> str:
        response = llm.generate(
            prompts=[prompt],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=2048),
        )
        return response[0].outputs[0].text

    return call_offline_model


if __name__ == "__main__":
    func = get_offline_model_func("Qwen/Qwen2.5-1.5B-Instruct")
    answer = func("请简要介绍一下人工智能的发展历史。")
    print(answer)
