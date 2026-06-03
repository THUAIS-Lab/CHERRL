"""
Preprocess the WritingBench dataset to parquet format
Similar to if_prompts.py but adapted for WritingBench data structure
"""

import re
import os
import json
import datasets
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def load_dataset(data_paths):
    """Load WritingBench dataset from JSONL files"""
    def _load_dataset(data_path):
        data = []
        data_path = os.path.expanduser(data_path)  # 展开 ~ 符号
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc=f"Loading {data_path}"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line.strip())
                    data.append({
                        "index": item.get("index", 0),
                        "domain1": item.get("domain1", ""),
                        "domain2": item.get("domain2", ""),
                        "lang": item.get("lang", "en"),
                        "query": item.get("query", ""),
                        "checklist": item.get("checklist", [])
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        return data
    
    data = []
    for data_path in data_paths:
        data.extend(_load_dataset(data_path))
    
    return data


def convert_checklist_to_rubrics(checklist):
    """
    Convert WritingBench checklist format to rubrics format for RuscaRL reward
    
    WritingBench format:
    {
        "name": "criteria name",
        "criteria_description": "description",
        "1-2": "low score description",
        "3-4": "medium-low score description",
        "5-6": "medium score description",
        "7-8": "medium-high score description",
        "9-10": "high score description"
    }
    
    Rubrics format (for ruscarl_reward.py):
    {
        "criterion": "criteria name",
        "points": 1.0,
        "tags": {}
    }
    """
    rubrics = []
    for item in checklist:
        rubric = {
            "criterion": item.get("criteria_description", item.get("name", "")),
            "points": 1.0,  # Each criterion has equal weight, can be adjusted
            "tags": {
                "name": item.get("name", ""),
                "description": item.get("criteria_description", ""),
                # Store score descriptions for reference
                "score_1_2": item.get("1-2", ""),
                "score_3_4": item.get("3-4", ""),
                "score_5_6": item.get("5-6", ""),
                "score_7_8": item.get("7-8", ""),
                "score_9_10": item.get("9-10", ""),
            }
        }
        rubrics.append(rubric)
    return rubrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/writingbench_prompts')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # data_paths = ["data/Crab-VerIF/data.jsonl"]
    
    data_paths = ["/data/DATASET/WritingBench/WritingBench.jsonl"]
    assert len(data_paths) != 0, "Please set your data path"
 
    data_source = data_paths[0].split("/")[-2].lower()
    data_list = load_dataset(data_paths)
    dataset = datasets.Dataset.from_list(data_list)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            query = example.pop('query')
            checklist = example.pop('checklist', [])
            domain1 = example.pop('domain1', '')
            domain2 = example.pop('domain2', '')
            lang = example.pop('lang', 'en')
            index = example.pop('index', idx)
            
            # Convert checklist to rubrics format
            rubrics = convert_checklist_to_rubrics(checklist)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "ability": "writing",
                "reward_model": {
                    "style": "rm",
                    "rubrics": rubrics,
                    "ground_truth": "none",
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'domain1': domain1,
                    'domain2': domain2,
                    'lang': lang,
                    'original_checklist': checklist,
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
    
    # Print first example for inspection
    print("\n" + "="*80)
    print("First processed example (complete content):")
    print("="*80)
    first_example = train_dataset[0]
    print(json.dumps(first_example, indent=2, ensure_ascii=False))
    print("="*80 + "\n")
