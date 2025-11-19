python -m vllm.entrypoints.openai.api_server \
  --model /home/skl/mkx/model/Qwen2.5-Coder-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name qw2.5
