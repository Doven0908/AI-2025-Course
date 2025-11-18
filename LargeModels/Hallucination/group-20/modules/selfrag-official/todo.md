首先阅读整个项目，我已经使用self-rag执行了下面三个推理，结果已保存，现在我需要你阅读ragas项目，调用deepseek-r1 api对推理结果接入 RAGAS 作为检动评测
CUDA_VISIBLE_DEVICES=5 python run_short_form.py \
--model_name /data1/xueziteng/homework/self-rag/selfrag-model \
--input_file ../eval_data/popqa_longtail_w_gs.jsonl \
--mode adaptive_retrieval --max_new_tokens 100 \
--threshold 0.2 \
--output_file exp1 \
--metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
--dtype half

CUDA_VISIBLE_DEVICES=5 python run_short_form.py \
  --model_name /data1/xueziteng/homework/self-rag/selfrag-model \
  --input_file ../eval_data/arc_challenge_processed.jsonl \
  --max_new_tokens 50 --threshold 0.2 \
  --output_file exp2 \
  --metric match --ndocs 5 --use_groundness --use_utility --use_seqscore \
  --task arc_c

CUDA_VISIBLE_DEVICES=5 python run_short_form.py \
    --model_name ../selfrag-model \
    --input_file ../eval_data/health_claims_processed.jsonl \
    --max_new_tokens 50 \
    --threshold 0.2 --output_file exp3_debug \
    --metric match --ndocs 5 \
    --use_groundness --use_utility --use_seqscore \
    --task fever