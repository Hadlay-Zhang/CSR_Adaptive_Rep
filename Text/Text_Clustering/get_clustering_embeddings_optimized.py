import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.datasets import fetch_20newsgroups

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Embedding Generator")
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "biorxiv-p2p", "biorxiv-s2s", "twentynewsgroups"
    ], help="Dataset: biorxiv-p2p, biorxiv-s2s, or twentynewsgroups")
    parser.add_argument("--split", type=str, required=True, choices=[
        "train", "test", "val", "all"
    ], help="Split: train, test, val, all")
    parser.add_argument("--batch_size", type=int, default=None, help="Override default batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    return parser.parse_args()

def get_optimal_batch_size(model, sample_text: str, instruction: str, max_length: int = 512):
    """Find optimal batch size based on available GPU memory"""
    batch_sizes = [64, 32, 16, 8, 4, 2, 1]
    
    for batch_size in batch_sizes:
        try:
            test_texts = [sample_text] * batch_size
            with torch.no_grad():
                embeddings = model.encode(test_texts, instruction=instruction, max_length=max_length)
                del embeddings
                torch.cuda.empty_cache()
            print(f"Optimal batch size: {batch_size}")
            return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Error testing batch size {batch_size}: {e}")
            continue
    return 1

def process_biorxiv_optimized(file_path, embed_path, instruction, batch_size, max_length):
    string_to_number = {}
    all_texts = []
    all_labels = []
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Reading data"):
            obj = json.loads(line)
            label = obj['labels']
            if label not in string_to_number:
                string_to_number[label] = len(string_to_number)
            
            all_texts.append(obj['sentences'])
            all_labels.append(string_to_number[label])
    
    print(f"Total samples: {len(all_texts)}, Labels: {len(string_to_number)}")

    if batch_size is None:
        batch_size = get_optimal_batch_size(model, all_texts[0], instruction, max_length)
    all_embeddings = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Generating embeddings"):
        batch_texts = all_texts[i:i+batch_size]
        
        with torch.no_grad():
            embeddings = model.encode(batch_texts, instruction=instruction, max_length=max_length)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        # cleanup
        if (i // batch_size + 1) % 50 == 0:
            torch.cuda.empty_cache()
    # Concat and save
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_embeddings_cpu = final_embeddings.cpu().numpy()
    del all_embeddings, final_embeddings
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    print(f"Saving embeddings to: {embed_path}")
    np.savez(embed_path, 
             data=final_embeddings_cpu, 
             label=np.array(all_labels),
             batch_size_used=batch_size)

def process_twentynewsgroups_optimized(embed_path, instruction, batch_size, split, max_length):
    data = fetch_20newsgroups(subset="all" if split == "all" else split, remove=('headers', 'footers', 'quotes'))
    texts, labels = data.data, data.target
    print(f"Total samples: {len(texts)}")
    if batch_size is None:
        batch_size = get_optimal_batch_size(model, texts[0], instruction, max_length)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        with torch.no_grad():
            embeddings = model.encode(batch_texts, instruction=instruction, max_length=max_length)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        if (i // batch_size + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    print("Concatenating embeddings...")
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_embeddings_cpu = final_embeddings.cpu().numpy()
    del all_embeddings, final_embeddings
    torch.cuda.empty_cache()
    output_file = f"{embed_path}_{split}.npz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving embeddings to: {output_file}")
    np.savez(output_file, 
             data=final_embeddings_cpu, 
             label=np.array(labels),
             batch_size_used=batch_size)

if __name__ == "__main__":
    args = parse_args()
    
    print("Starting optimized embedding generation...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Load model
    model_name = "nvidia/NV-Embed-v2"
    print(f"Loading model: {model_name}")
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map="auto",
        torch_dtype=torch_dtype
    )
    
    # Try model compilation
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compilation enabled")
    except:
        print("Model compilation not available")

    DATASET_CONFIG = {
        "biorxiv-p2p": {
            "dir": "../datasets/biorxiv-clustering-p2p/",
            "instruction": "Generate a representation that captures the core scientific topic and main subject of the following title and abstract for the purpose of clustering similar research together. Focus on themes, methods, and biological concepts. \n Title and abstract: ",
            "file_template": "{split}.jsonl",
            "output": "./biorxiv-clustering-p2p/{split}.npz",
            "processor": process_biorxiv_optimized
        },
        "biorxiv-s2s": {
            "dir": "../datasets/biorxiv-clustering-s2s/",
            "instruction": "Given a title of an essay related to biology, Please describe the research field this essay may belong to. \n Title: ",
            "file_template": "{split}.jsonl",
            "output": "./biorxiv-clustering-s2s/{split}.npz",
            "processor": process_biorxiv_optimized
        },
        "twentynewsgroups": {
            "dir": "../datasets/twentynewsgroups-clustering/",
            "instruction": "Identify the topic or theme of the given news articles. \n News article: ",
            "file_template": "{split}.jsonl",
            "output": "./twentynewsgroups/my_prompt",
            "processor": process_twentynewsgroups_optimized
        }
    }

    dataset_cfg = DATASET_CONFIG[args.dataset]
    
    if args.dataset == "twentynewsgroups" and args.split not in ["all", "train", "test"]:
        raise ValueError("twentynewsgroups only supports all, train and test splits")

    if (args.dataset == "biorxiv-s2s" or args.dataset == 'biorxiv-p2p') and args.split not in ["test"]:
        raise ValueError("biorxiv-datasets only supports test splits")

    # Use provided batch size or optimal batch size
    batch_size = args.batch_size
    
    input_file = os.path.join(dataset_cfg["dir"], dataset_cfg["file_template"].format(split=args.split))
    output_path = dataset_cfg["output"].format(split=args.split)

    print(f"Batch size: {batch_size}")

    if args.dataset.startswith("biorxiv"):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        dataset_cfg["processor"](input_file, output_path, dataset_cfg["instruction"], 
                                batch_size, args.max_length)
    elif args.dataset == "twentynewsgroups":
        dataset_cfg["processor"](output_path, dataset_cfg["instruction"], 
                                batch_size, args.split, args.max_length)
    else:
        raise ValueError("Unknown dataset type")

    final_output = output_path if args.dataset != 'twentynewsgroups' else f"{output_path}_{args.split}.npz"