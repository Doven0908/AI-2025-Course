import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 设置中文字体（根据系统自动适配）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体（Windows/Linux）
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块的问题
# ===============================
# 参数设置
# ===============================
json_path = './hallucination_results.json'  # 你的文件路径
high_risk_threshold = 0.7                   # 高风险阈值

# ===============================
# 读取 JSON 文件
# ===============================
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

baseline_confidences = []
cove_confidences = []

# ===============================
# 提取每个样本的置信度
# ===============================
for item in data:
    # baseline
    if item.get("baseline_predictions"):
        baseline_confidences.extend([p["confidence"] for p in item["baseline_predictions"] if "confidence" in p])
    # cove
    if item.get("cove_predictions"):
        cove_confidences.extend([p["confidence"] for p in item["cove_predictions"] if "confidence" in p])

# ===============================
# 计算统计指标
# ===============================
def summarize(confidences, name):
    avg = np.mean(confidences) if confidences else np.nan
    high_ratio = np.mean(np.array(confidences) > high_risk_threshold) if confidences else np.nan
    print(f"📊 {name} 模型：")
    print(f"  平均幻觉置信度：{avg:.4f}")
    print(f"  高风险句比例(>{high_risk_threshold})：{high_ratio*100:.2f}%")
    print(f"  有效预测句数：{len(confidences)}\n")
    return avg, high_ratio

avg_base, high_base = summarize(baseline_confidences, "Baseline")
avg_cove, high_cove = summarize(cove_confidences, "CoVe")

# ===============================
# 可视化对比
# ===============================
sns.set(style="whitegrid", font_scale=1.3)
plt.figure(figsize=(9, 6))

sns.kdeplot(baseline_confidences, shade=True, label=f'Baseline (mean={avg_base:.3f})')
sns.kdeplot(cove_confidences, shade=True, label=f'CoVe (mean={avg_cove:.3f})')

plt.axvline(high_risk_threshold, color='red', linestyle='--', label='High-risk threshold')
plt.xlabel("幻觉置信度", fontsize=13)
plt.ylabel("密度", fontsize=13)
plt.title("Baseline 与 CoVe 模型幻觉置信度分布对比", fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# 结论打印
# ===============================
if avg_cove < avg_base:
    print("✅ 结论：CoVe 模型整体幻觉风险更低，表现更好。")
else:
    print("⚠️ 结论：Baseline 模型幻觉置信度更低，CoVe 改写后反而引入幻觉。")
