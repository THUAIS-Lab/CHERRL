"""
Preprocess the IFEval dataset to parquet format
"""

import re
import os
import json
import datasets
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def load_dataset(data_paths):
    # 获取 verl/utils/reward_score/local_server 的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    verl_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    local_server_path = os.path.join(verl_dir, 'verl/utils/reward_score/local_server')
    local_server_path = os.path.abspath(local_server_path)
    
    def _load_dataset(data_path):
        data = []
        data_path = os.path.expanduser(data_path)  # 展开 ~ 符号
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                item = json.loads(line.strip())
                # 替换 <path here> 为实际路径，需要转义引号
                path_str = f'"{local_server_path}"'
                functions = ["import sys\nsys.path.append(" + path_str + ")\n"+function for function in item["functions"]]

                data.append({
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "checkers": item["checkers"],
                    "functions": functions
                })
        return data
    
    data = []
    for data_path in data_paths:
        data.extend(_load_dataset(data_path))
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/if_prompts')
    parser.add_argument('--data_path', default='data/VerInstruct/data.jsonl',
                        help='Path to the VerInstruct JSONL file')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_paths = [args.data_path]
    assert len(data_paths) != 0, "Please set your data path"

    data_source = data_paths[0].split("/")[-2].lower()
    data_list = load_dataset(data_paths)
    dataset = datasets.Dataset.from_list(data_list)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('prompt')
            checkers = example.pop('checkers')
            functions = example.pop('functions')
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "if",
                "reward_model": {
                    "style": "rm",
                    "ground_truth": json.dumps({
                        "checkers": checkers,
                        "functions": functions
                    })
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)  # 展开 ~ 符号
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
