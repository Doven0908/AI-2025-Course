import os
import json
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets

MATH500_DATASET_DIR = "/home/lijiakun25/models/datasets/math500/test.jsonl"
AIME24_DATASET_DIR = "/home/lijiakun25/models/datasets/AIME24/aime_2024_problems.parquet"
AIME25_DATASET_DIR = "/home/lijiakun25/models/datasets/AIME25/aime2025.jsonl"
OLYMPIADBENCH_DATASET_DIR = "/home/lijiakun25/models/datasets/OlympiadBench_OE_TO"

def load_data(data_name,data_path):
    if os.path.exists(data_path) and data_path.endswith(".jsonl"):
        examples = datasets.load_dataset("json", data_files={"test": data_path})["test"]
    elif os.path.exists(data_path) and data_path.endswith(".parquet"):
        examples = datasets.load_dataset("parquet", data_files={"test": data_path})["test"]
    else:
        print(f"data path {data_path} not exists, please check again and we'll load data from default path")
        if data_name == "math":
            examples = datasets.load_dataset("json", data_files={"test": MATH500_DATASET_DIR})["test"]
        elif data_name == "aime24":
            examples = datasets.load_dataset("parquet", data_files={"test": AIME24_DATASET_DIR})["test"]
        elif data_name == "aime25":
            examples = datasets.load_dataset("json", data_files={"test": AIME25_DATASET_DIR})["test"]
        elif data_name.startswith("olympiadbench"):
            name_parts = data_name.split("_")  # e.g. ['olympiadbench', 'physics', 'en']
            filter_subjects = {"physics", "maths"} if "physics" not in name_parts and "maths" not in name_parts else set(name_parts) & {"physics", "maths"}
            filter_langs = {"zh", "en"} if "zh" not in name_parts and "en" not in name_parts else set(name_parts) & {"zh", "en"}


            # 遍历文件筛选
            parquet_files = []
            for fname in os.listdir(OLYMPIADBENCH_DATASET_DIR):
                if not fname.endswith(".parquet"):
                    continue

                parts = fname.replace(".parquet", "").split("_")  # e.g. ['OE', 'TO', 'physics', 'zh', 'CEE']
                subject = parts[2]  # physics or maths
                lang = parts[3]     # zh or en

                if subject in filter_subjects and lang in filter_langs:
                    parquet_files.append(os.path.join(OLYMPIADBENCH_DATASET_DIR, fname))

            if not parquet_files:
                raise ValueError("No matching parquet files found for given filters.")

            # 读取为一个合并的大 Dataset
            datasets_list = [
                load_dataset("parquet", data_files={"test": f})["test"]
                for f in parquet_files
            ]
            examples = concatenate_datasets(datasets_list)

        else:
            raise NotImplementedError(data_name)
    return examples
