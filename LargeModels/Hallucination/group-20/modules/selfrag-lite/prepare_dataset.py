import json 
import argparse
import random

EXP_FILE_NAME = './experiment_file.json'

def main(dataset: str, num: int) -> None:
    if dataset == 'asqa':
        data = json.load(open('./eval_data/asqa_eval_gtr_top100.json'))
        data = random.sample(data, num)

        json.dump(data, open(EXP_FILE_NAME, 'w'), indent=4)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (JSON format).")
    parser.add_argument("--num", type=int, required=True, help="Number of samples to process from the dataset.")
    args = parser.parse_args()
    
    main(dataset=args.dataset, num=args.num)


