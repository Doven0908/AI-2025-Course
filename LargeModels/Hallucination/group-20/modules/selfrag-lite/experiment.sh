#!/bin/bash
set -e

echo "[1/6] 准备数据集..."
python prepare_dataset.py  --dataset asqa  --num 20

echo "[2/6] 使用 OpenAI 获取基线答案..."
python qa_with_openai.py

echo "[3/6] 使用 SelfRAG  获取答案..."
python run_long_form_static.py \
  --model_name selfrag/selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task asqa --input_file ./experiment_file.json \
  --output_file ./experiment_file.json --max_depth 7 --mode always_retrieve 

echo "[4/6] 使用 RAGAS 对回答进行评估..."
python ragas_eval.py       

echo "[5/6] 使用 SelfCheckGpt 对回答进行语句级评估..."
python hallucination_detection.py     

echo "[6/6] 可视化结果..."
python visualize.py

echo "实验完成！"
echo "图表结果已保存至 visualization_analysis 文件夹"
echo "原始数据已保存至 experiment_file.json"
