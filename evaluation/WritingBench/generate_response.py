import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI


def save_output(output, file_name):
    """
    Saves output data to a specified file in JSONL format.
    """
    # 创建目录（如果不存在）
    dir_name = os.path.dirname(file_name)
    if dir_name:  # 只有当目录名不为空时才创建
        os.makedirs(dir_name, exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    """
    Loads JSONL lines from a file into a list of dictionaries.
    """
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

# === Your code here ===

def writer(query, api_key, base_url, model_name):
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"error: {e}")
        raise
    
def process(id_query_map, out_file, api_key, base_url, model_name):
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(id_query_map)
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {id_query_map.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            if existing_count > 0 and i < existing_count: 
                pbar.update()
                continue
            data = {"index": content["index"]}
            query = content["query"]
            data["response"] = writer(query, api_key, base_url, model_name)
            save_output([data], out_file)
            cnt += 1
            pbar.update()

    print(f"CNT: {cnt}")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--query_file", type=str, help="Path to the query file.", default="benchmark_query/benchmark_all.jsonl")
    parser.add_argument("--output_file", type=str, help="Path to the output file.", default=None)
    parser.add_argument("--api_key", type=str, required=True, help="API key for the OpenAI client.")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL for the API endpoint.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to use for generation.")

    args = parser.parse_args()
    
    # 如果没有指定输出文件，使用带时间戳的默认文件名
    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"responses/{timestamp}/response_model_{timestamp}.jsonl"

    process(args.query_file, args.output_file, args.api_key, args.base_url, args.model_name)

