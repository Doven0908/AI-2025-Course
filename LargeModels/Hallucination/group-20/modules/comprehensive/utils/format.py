import re

def safe_name(text: str) -> str:
    """将文本转换为安全的文件名格式，替换掉不安全的字符。"""
    return re.sub(r'[^a-zA-Z0-9]', '_', text).strip('_')


    