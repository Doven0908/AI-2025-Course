import json
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import mean
from pathlib import Path
import os

def confidence_statistics(grouped_data,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_confidences = []
    false_confidences = []

    for tree_id, samples in grouped_data.items():
        for item in samples:
            if "is_equal" in item:
                confidence = item.get("confidence", None)
                if confidence is not None:
                    if item["is_equal"] == True:
                        true_confidences.append(confidence)
                    elif item["is_equal"] == False:
                        false_confidences.append(confidence)

    print("✔ 包含 is_equal 字段的记录总数:", len(true_confidences) + len(false_confidences))

    print("✔ is_equal == True:")
    print("  数量:", len(true_confidences))
    print("  平均 confidence:", mean(true_confidences) if true_confidences else "无数据")
    print("  分布:", true_confidences)

    print("✔ is_equal == False:")
    print("  数量:", len(false_confidences))
    print("  平均 confidence:", mean(false_confidences) if false_confidences else "无数据")
    print("  分布:", false_confidences)

    if len(true_confidences) >= 2 and len(false_confidences) >= 2:
        kde_true = gaussian_kde(true_confidences)
        kde_false = gaussian_kde(false_confidences)

        # 生成 x 区间用于画图
        x_min = min(min(true_confidences), min(false_confidences))
        x_max = max(max(true_confidences), max(false_confidences))
        x_vals = [x_min + (x_max - x_min) * i / 200 for i in range(201)]

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, kde_true(x_vals), label='is_equal = True', color='green')
        plt.plot(x_vals, kde_false(x_vals), label='is_equal = False', color='red')
        plt.fill_between(x_vals, kde_true(x_vals), alpha=0.3, color='green')
        plt.fill_between(x_vals, kde_false(x_vals), alpha=0.3, color='red')
        plt.title('Confidence KDE by is_equal')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_histogram.png", dpi=300)
        plt.close()
    else:
        print("⚠ 样本不足，无法绘制 KDE 图(至少每类2个数据)")


if __name__ == "__main__":

    output_dir = Path("llm_output")
    data_path = "/home/lijiakun25/math-inference/llm_output/splition7_math500.jsonl"


    grouped_data = defaultdict(list)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            tree_id = item.get('tree_id', None)
            grouped_data[tree_id].append(item)

    # grouped_data 是一个 dict，键是 tree_id，值是包含同一 tree_id 的记录列表
    # 可以打印或保存
    for tree_id, items in grouped_data.items():
        print(f"Tree ID: {tree_id}, 共 {len(items)} 条记录")
        for i, item in enumerate(items):
            print(f"  第{i+1}条：{item['output'][:50]}...")  # 打印前50个字符预览
    
    confidence_statistics(grouped_data,output_dir)

