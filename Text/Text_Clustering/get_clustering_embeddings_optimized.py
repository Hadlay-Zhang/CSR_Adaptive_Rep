import argparse
import os
import json
import gzip
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.datasets import fetch_20newsgroups

def get_model():
    model_name = "nvidia/NV-Embed-v2"
    print(f"Loading model: {model_name}")
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map="auto",
        torch_dtype=torch_dtype
    )
    
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compilation enabled")
    except:
        print("Model compilation not available")
    
    return model

def get_optimal_batch_size(model, sample_text: str, instruction: str):
    batch_sizes = [32, 16, 8, 4, 2, 1]
    
    for batch_size in batch_sizes:
        try:
            test_texts = [sample_text] * batch_size
            with torch.no_grad():
                embeddings = model.encode(test_texts, instruction=instruction)
                del embeddings
                torch.cuda.empty_cache()
            print(f"Optimal batch size: {batch_size}")
            return max(1, batch_size // 2)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return 1

def open_file(file_path):
    if file_path.endswith('.gz'):
        return gzip.open(file_path, 'rt', encoding='utf-8')
    else:
        return open(file_path, 'r', encoding='utf-8')

def find_input_file(base_dir, filename_template, split):
    possible_files = [
        os.path.join(base_dir, filename_template.format(split=split)),
        os.path.join(base_dir, filename_template.format(split=split) + ".gz")
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path
    return None

def process_biorxiv_optimized(model, input_file, output_file, instruction, batch_size):
    # Build label mapping
    string_to_number = {}
    with open_file(input_file) as f:
        for line in f:
            obj = json.loads(line)
            label = obj['labels']
            if label not in string_to_number:
                string_to_number[label] = len(string_to_number)

    # Read all data
    all_texts, all_labels = [], []
    with open_file(input_file) as f:
        for line in tqdm(f, desc=f"Reading {input_file}"):
            obj = json.loads(line)
            all_texts.append(obj['sentences'])
            all_labels.append(string_to_number[obj['labels']])
    
    print(f"Total samples: {len(all_texts)}")
    
    if batch_size is None:
        batch_size = get_optimal_batch_size(model, all_texts[0], instruction)
    
    all_embeddings = []
    current_batch_size = batch_size
    
    i = 0
    with tqdm(total=len(all_texts), desc="Generating embeddings") as pbar:
        while i < len(all_texts):
            batch_texts = all_texts[i:i+current_batch_size]
            
            try:
                with torch.no_grad():
                    embeddings = model.encode(batch_texts, instruction=instruction)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings)
                
                i += current_batch_size
                pbar.update(len(batch_texts))
                
                if len(all_embeddings) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    raise RuntimeError("Cannot process even with batch size 1")
                
                current_batch_size = max(1, current_batch_size // 2)
                print(f"OOM: reducing batch size to {current_batch_size}")
                continue
    
    final_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    del all_embeddings
    torch.cuda.empty_cache()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, data=final_embeddings, label=np.array(all_labels))

def process_twentynewsgroups_optimized(model, output_file, instruction, batch_size, split):
    data = fetch_20newsgroups(subset="all" if split == "all" else split, remove=('headers', 'footers', 'quotes'))
    all_texts, all_labels = data.data, data.target
    
    print(f"Total samples: {len(all_texts)}")
    
    if batch_size is None:
        batch_size = get_optimal_batch_size(model, all_texts[0], instruction)
    
    all_embeddings = []
    current_batch_size = batch_size
    
    i = 0
    with tqdm(total=len(all_texts), desc="Generating embeddings") as pbar:
        while i < len(all_texts):
            batch_texts = all_texts[i:i+current_batch_size]
            
            try:
                with torch.no_grad():
                    embeddings = model.encode(batch_texts, instruction=instruction)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings)
                
                i += current_batch_size
                pbar.update(len(batch_texts))
                
                if len(all_embeddings) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    raise RuntimeError("Cannot process even with batch size 1")
                
                current_batch_size = max(1, current_batch_size // 2)
                print(f"OOM: reducing batch size to {current_batch_size}")
                continue
    
    final_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    del all_embeddings
    torch.cuda.empty_cache()
    
    output_file_final = f"{output_file}_{split}.npz"
    os.makedirs(os.path.dirname(output_file_final), exist_ok=True)
    np.savez(output_file_final, data=final_embeddings, label=np.array(all_labels))

DATASET_CONFIG = {
    "biorxiv-p2p": {
        "dir": "../datasets/biorxiv-clustering-p2p/",
        "instruction": "Generate a representation that captures the core scientific topic and main subject of the following title and abstract for the purpose of clustering similar research together. Focus on themes, methods, and biological concepts. \n Title and abstract: ",
        "batch_size": 8,
        "file_template": "{split}.jsonl",
        "output": "./biorxiv-clustering-p2p/{split}.npz",
        "processor": process_biorxiv_optimized
    },
    "biorxiv-s2s": {
        "dir": "../datasets/biorxiv-clustering-s2s/",
        "instruction": "Given a title of an essay related to biology, Please describe the research field this essay may belong to. \n Title: ",
        "batch_size": 2,
        "file_template": "{split}.jsonl",
        "output": "./biorxiv-clustering-s2s/{split}.npz",
        "processor": process_biorxiv_optimized
    },
    "twentynewsgroups": {
        "dir": "../datasets/twentynewsgroups-clustering/",
        "instruction": "Identify the topic or theme of the given news articles. \n News article: ",
        "batch_size": 2,
        "file_template": "{split}.jsonl",
        "output": "./twentynewsgroups/my_prompt",
        "processor": process_twentynewsgroups_optimized
    }
}

def main():
    parser = argparse.ArgumentParser(description="Generate optimized clustering embeddings")
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIG.keys()), help="Dataset name")
    parser.add_argument('--split', required=True, choices=["train", "test", "val", "all"], help="Data split")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size (auto-detect if not provided)")
    args = parser.parse_args()

    print("Starting optimized embedding generation...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    config = DATASET_CONFIG[args.dataset]
    model = get_model()

    if args.dataset == "twentynewsgroups" and args.split not in ["all", "train", "test"]:
        raise ValueError("twentynewsgroups only supports all, train and test splits")

    if args.dataset.startswith("biorxiv") and args.split not in ["test"]:
        raise ValueError("biorxiv datasets only support test split")

    print(f"Batch size: {args.batch_size if args.batch_size else 'Auto-detect'}")

    if args.dataset.startswith("biorxiv"):
        input_file = find_input_file(config["dir"], config["file_template"], args.split)
        if not input_file:
            raise FileNotFoundError(f"Input file not found for {args.dataset} {args.split}")
        output_file = config["output"].format(split=args.split)
        config["processor"](model, input_file, output_file, config["instruction"], args.batch_size)
    elif args.dataset == "twentynewsgroups":
        output_file = config["output"]
        config["processor"](model, output_file, config["instruction"], args.batch_size, args.split)
    
    final_output = config["output"].format(split=args.split) if args.dataset != 'twentynewsgroups' else f"{config['output']}_{args.split}.npz"
    print(f"Embeddings saved to {final_output}")

if __name__ == '__main__':
    main()