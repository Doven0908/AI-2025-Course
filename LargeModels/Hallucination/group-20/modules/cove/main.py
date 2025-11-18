from cove import chain
import json
from lettucedetect.models.inference import HallucinationDetector

def func(detector: HallucinationDetector, json_data, output_path):
    results = []  # 用来保存每条问题的检测结果

    for d in json_data:
        context = d['context']
        question = d['question']
        response = chain.invoke({"original_question": question})

        print("question:", question)  # 保留原来的打印问题

        entry = {
            "question": question,
            "baseline_response": response.get('baseline_response', ""),
            "final_answer": response.get('final_answer', ""),
            "baseline_predictions": None,
            "cove_predictions": None
        }

        try:
            predictions1 = detector.predict(
                context=[context],
                question=question,
                answer=response['baseline_response'],
                output_format="spans"
            )
            predictions2 = detector.predict(
                context=[context],
                question=question,
                answer=response['final_answer'],
                output_format="spans"
            )

            print("baseline:", predictions1)  # 保留原来的打印
            print("Cove:", predictions2)      # 保留原来的打印

            entry["baseline_predictions"] = predictions1
            entry["cove_predictions"] = predictions2

        except Exception as e:
            print("1")  # 保留原来的异常打印
            entry["error"] = str(e)

        results.append(entry)

    # 保存到 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"检测结果已保存到 {output_path}")


if __name__ == "__main__":
    detector = HallucinationDetector(
        method="transformer", 
        model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
    )

    with open('./question.json', 'r', encoding='UTF-8') as f:
        json_data = json.load(f)

    # 调用函数并指定输出文件
    func(detector, json_data, './hallucination_results_plus.json')
