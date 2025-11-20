import re
import json
import Levenshtein


def extract_json_objects(text):
    # 匹配 {"step": 数字, "content": "..."}，允许换行内容（非贪婪匹配）
    pattern = r'\{"step":\s*\d+,\s*"content":\s*"(.*?)"\}'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    result = []
    for i, content in enumerate(matches):
        # 手动构造 JSON 对象字符串，先清洗控制字符
        safe_content = content.replace('\n', '\\n').replace('\r', '\\r')
        json_str = f'{{"step": {i + 1}, "content": "{safe_content}"}}'
        try:
            obj = json.loads(json_str)
            result.append(obj)
        except json.JSONDecodeError as e:
            print(f"解析失败: {e}，内容为：{json_str[:50]}...")
    return result