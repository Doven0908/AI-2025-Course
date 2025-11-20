import os
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
from typing import *

import argparse
import pandas as pd
from datasets import Dataset
import swandlab
import torch
import transformers

DEEPSPEED_CONFIG_JSON = {
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": True,
  "fp16": {
    "enabled": "auto"
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": True
    },
    "overlap_comm": False,
    "contiguous_gradients": True,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": True
  }
}

def get_local_rank() -> str:
    return os.environ.get('LOCAL_RANK', '0')


def print_log(title, content = ''):
    if get_local_rank() == '0':
        print(f'=================== {title} ===================')
        if content != '':
            print(content)


def get_current_device() -> 'torch.device':
    device = f'''cuda:{get_local_rank()}'''
    return torch.device(device)


def support_bf16() -> bool:
    return transformers.utils.import_utils.is_torch_bf16_gpu_available()


def print_log(message: str) -> None:
    if get_local_rank() == '0':
        print(message, flush=True)

import json
import re
import Levenshtein
def data_processing(train_dataset_path,):
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
                    ans += step
                distance = Levenshtein.distance(ans, log_entry["output"])
                similarity = 1 - distance / max(len(ans), len(log_entry["output"]))  # 转成 0~1 相似度
                print(f"相似度：{similarity:.2f}")
        if similarity == 1:
            ans = ""
            for step in parsed_output:
                ans += step
                ans += "<END_METHOD>"
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--model_path', type=str, default='/home/lijiakun25/models/Qwen3-8b')
    parser.add_argument('--run_name', type=str, default='qwen7b-sft')
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--swandlab_project', type=str, default='testval_sft')

    parser.add_argument('--max_input_tokens', type=int, default=1500)
    parser.add_argument('--max_output_tokens', type=int, default=1024)

    parser.add_argument('--train_dataset_path', type=str, default='data/testval_sft_train.json') #TODO:更换数据集合
    parser.add_argument('--eval_dataset_path', type=str, default='data/testval_sft_eval.json')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3)

    args = parser.parse_args()

    run_name = args.run_name
    model_path = args.model_path
    output_dir = f'output/sft/{run_name}'
    max_input_tokens = args.max_input_tokens
    max_output_tokens = args.max_output_tokens
    max_length = max_input_tokens + max_output_tokens


    if get_local_rank() == '0':
        swandlab.init(
            project=args.swandlab_project,
            name=run_name
        )


    print_log(f'=============== load model ===============')

    model_args = {
        'pretrained_model_name_or_path': model_path,
        'device_map': {'': get_current_device()},
        'trust_remote_code': True
    }

    print_log(f'=============== quantization ===============')
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(**model_args)
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print_log(f'load {model_path}.')

    lora_args = {
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05
    }

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **lora_args,
        bias='none',
        target_modules=['up_proj', 'gate_proj', 'v_proj', 'k_proj', 'q_proj', 'o_proj', 'down_proj'],
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)


    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_trainable_params(model)
    formatted_num_params = '{:,}'.format(num_params)
    print_log(f'Number of trainable parameters: {formatted_num_params}')


    SYSTEM_PROMPT = '''\
You are an expert test programmer. Given a problem description and a pair of test input and output, please verify whether the test pair is correct.
First describe your reasoning steps, then put your final answer (yes or no) in a \\boxed{}.'''

    print_log(f'=============== load dataset ===============')
    # load dataset


    def load_local_dataset(dataset_path: str) -> Dataset:
        df = pd.read_json(dataset_path)
        if args.train_samples > 0:
            df = df[ : args.train_samples]

        ds = Dataset.from_pandas(df)
        def formatting_prompts(row):
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row['result']}
            ]
            row['messages'] = messages
            return row

        ds = ds.map(
            formatting_prompts,
            remove_columns=['prompt', 'result'],
            num_proc=32
        )
        return ds

    train_dataset = load_local_dataset(args.train_dataset_path)
    eval_dataset = load_local_dataset(args.eval_dataset_path)
    

    print_log('==================== example ====================')
    print_log(train_dataset['messages'][0])
    print_log('=================================================')

    training_args = SFTConfig(
        run_name=run_name,

        # args for SFTConfig
        max_seq_length=max_length,
        dataset_num_proc=32,

        # args for TrainingArguments
        output_dir=output_dir,
        overwrite_output_dir=True,

        logging_strategy='steps',
        logging_steps=1,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,

        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        # optim='paged_adamw_8bit',

        do_eval=True,
        eval_strategy='steps',
        eval_steps=50,

        bf16=support_bf16(),
        fp16=not support_bf16(),

        use_liger_kernel=True,
        save_strategy='epoch',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            'use_reentrant': False
        },
        ddp_find_unused_parameters=False,
        deepspeed='deepspeed_config.json',
        report_to='swanlab'
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print_log('=============================== start training ===============================')
    print_log(f'gpus: {trainer.args.world_size}')
    print_log(f'''train batch size: {trainer.args.world_size * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}''')
    print_log(f'''eval batch size: {trainer.args.world_size * training_args.per_device_eval_batch_size * training_args.gradient_accumulation_steps}''')

    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.log_metrics('train', train_result.metrics)
    trainer.save_metrics('train', train_result.metrics)
    trainer.save_state()
