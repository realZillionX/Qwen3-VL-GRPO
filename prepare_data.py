import json
import os
import argparse
from pathlib import Path

def process_eyeballing(data_root, output_path):
    dataset_dir = os.path.join(data_root, 'dataset_eyeballing')
    json_path = os.path.join(dataset_dir, 'data.json')
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping eyeballing.")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        # Construct absolute image path
        image_path = os.path.join(dataset_dir, item['image'])
        
        # Format for ms-swift
        # query: prompt
        # response: correct_option (for reference/SFT, though GRPO uses rewards)
        # images: [image_path]
        # solution: correct_option (specifically for reward function)
        
        entry = {
            "query": item['prompt'],
            "response": item['correct_option'],
            "images": [image_path],
            "solution": item['correct_option'],
            "task_type": "eyeballing"
        }
        processed.append(entry)
    
    return processed

def process_maze(data_root, output_path):
    dataset_dir = os.path.join(data_root, 'dataset_maze')
    json_path = os.path.join(dataset_dir, 'data.json')
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping maze.")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        image_path = os.path.join(dataset_dir, item['image'])
        
        # Maze uses gpt5_prompt
        # solution is the list of cell IDs [1, 2, 3...]
        
        solution_str = json.dumps(item['solution_path_cell_ids'])
        
        entry = {
            "query": item['gpt5_prompt'],
            "response": solution_str,
            "images": [image_path],
            "solution": solution_str,
            "task_type": "maze"
        }
        processed.append(entry)
    
    return processed

def main():
    parser = argparse.ArgumentParser(description="Convert VLMPuzzle datasets to ms-swift JSONL format")
    parser.add_argument("--data_root", type=str, default="/inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset", help="Root directory containing dataset_eyeballing and dataset_maze")
    parser.add_argument("--output_path", type=str, default="train.jsonl", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    all_data = []
    all_data.extend(process_eyeballing(args.data_root, args.output_path))
    all_data.extend(process_maze(args.data_root, args.output_path))
    
    print(f"Total samples: {len(all_data)}")
    
    with open(args.output_path, 'w') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
