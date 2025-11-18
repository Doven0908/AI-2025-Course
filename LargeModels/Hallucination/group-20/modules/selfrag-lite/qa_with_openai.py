import json
from openai import OpenAI
import os
import time

EXP_FILE_NAME = './experiment_file.json'

# 初始化客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

def ask_openai(question):
    """
    使用新版 OpenAI 库进行问答
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"错误: {e}"


def main() -> None:

    data = json.load(open(EXP_FILE_NAME))
    
    for i, item in enumerate(data):
        question = item['question']
        print(f"({i+1}/{len(data)})问题: {question}\n")
        t = time.time()
        answer = ask_openai(question)
        item['openai_answer'] = answer
        item['openai_time'] = time.time() - t
        print(f"OpenAI回答: {answer}\n[用时: {item['openai_time']:.2f}秒]\n{'-'*40}\n")


        rag_prompt = '{}\n\nAnswer the question based on the provided context:\n{}'.format('\n'.join([e['text'] for e in item['docs'][:3]]), question)
        t = time.time()
        rag_answer = ask_openai(rag_prompt)
        item['rag_answer'] = rag_answer
        item['rag_time'] = time.time() - t
        print(f"RAG回答: {rag_answer}\n[用时: {item['rag_time']:.2f}秒]\n{'='*40}\n")

    print("OpenAI总耗时{:.2f}秒，平均每样本{:.2f}秒".format(sum([item['openai_time'] for item in data]), sum([item['openai_time'] for item in data])/len(data)))
    print("RAG总耗时{:.2f}秒，平均每样本{:.2f}秒".format(sum([item['rag_time'] for item in data]), sum([item['rag_time'] for item in data])/len(data)))
    print('\n\n')

    json.dump(data, open(EXP_FILE_NAME, 'w'), indent=4)


if __name__ == "__main__":
    main()