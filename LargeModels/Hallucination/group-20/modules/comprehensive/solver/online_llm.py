from litellm import completion


def get_online_model_func(model: str):
    """
    根据模型名称ID获取对应的在线模型调用函数
    
    Args:
        model: 模型名称ID
        
    Returns:
        Callable: 对应的在线模型调用函数
    """
    def call(prompt: str) -> str:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response['choices'][0]['message']['content']
    return call


if __name__ == "__main__":
    func = get_online_model_func("openai/gpt-4o")
    answer = func("请简要介绍一下人工智能的发展历史。")
    print(answer)
