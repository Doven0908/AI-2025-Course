import re
import json
import Levenshtein

def data_processing(train_dataset_path):
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
    

    log_entrys = []
    with open(train_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                log_entrys.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"读取错误：{e}")
    for log_entry in log_entrys:
        if "split_output" in log_entry and isinstance(log_entry["split_output"], str):
            split_output_str = log_entry["split_output"]
            parsed_output = extract_json_objects(split_output_str)
            log_entry["split_output"] = parsed_output
            if parsed_output:
                ans = ""
                for step in parsed_output:
                    ans += step["content"]
                distance = Levenshtein.distance(ans, log_entry["output"])
                similarity = 1 - distance / max(len(ans), len(log_entry["output"]))  # 转成 0~1 相似度
                print(f"相似度：{similarity:.2f}")
        if similarity == 1:
            ans = ""
            for step in parsed_output:
                ans += step
                ans += "<END_METHOD>"

if __name__ == "__main__":
    train_dataset_path = "/home/lijiakun25/math-inference/llm_output/0_20250826-213522.jsonl"
    data_processing(train_dataset_path)