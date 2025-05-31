import argparse
import torch.nn.functional as F
from transformers import AutoModel
import json
from tqdm import tqdm
import numpy as np
import os
import torch


def get_model():
    model_name = "nvidia/NV-Embed-v2"
    print(f"Loading model: {model_name}")
    
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map="auto"
    )
    
    # Optional compilation for speed
    try:
        if os.getenv('USE_TORCH_COMPILE', '0') == '1':
            model = torch.compile(model, mode="reduce-overhead")
    except:
        pass
    
    return model


def find_batch_size(model, sample_text, instruction):
    """Find optimal batch size for GPU memory"""
    for batch_size in [64, 32, 16, 8, 4, 2, 1]:
        try:
            test_texts = [sample_text] * batch_size
            with torch.no_grad():
                embeddings = model.encode(test_texts, instruction=instruction)
                del embeddings
                torch.cuda.empty_cache()
            return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return 1


def process_texts(model, input_file, output_file, instruction, batch_size):
    # Read all texts
    all_texts = []
    with open(input_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            all_texts.append(obj['text'])
    
    # Auto-detect batch size if needed
    if batch_size is None:
        batch_size = find_batch_size(model, all_texts[0], instruction)
    
    # Process in batches
    embeddings_list = []
    current_batch = batch_size
    
    i = 0
    with tqdm(total=len(all_texts), desc="Processing") as pbar:
        while i < len(all_texts):
            batch_texts = all_texts[i:i+current_batch]
            
            try:
                with torch.no_grad():
                    embeddings = model.encode(batch_texts, instruction=instruction)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    embeddings_list.append(embeddings.cpu().numpy())
                
                i += current_batch
                pbar.update(len(batch_texts))
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current_batch = max(1, current_batch // 2)
                continue
    
    # Combine and save
    final_embeddings = np.vstack(embeddings_list)
    torch.cuda.empty_cache()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, data=final_embeddings)
    print(f"Saved embeddings to {output_file}")


DATASETS = {
    "nfcorpus": {
        "dir": "../datasets/nfcorpus/",
        "instructions": {
            "corpus": "",
            "queries": "Given a question, retrieve relevant documents that answer the question"
        },
        "output": "./nfcorpus/{split}.npz"
    },
    "scifact": {
        "dir": "../datasets/scifact/",
        "instructions": {
            "corpus": "",
            "queries": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
        "output": "./scifact/{split}.npz"
    },
    "fiqa": {
        "dir": "../datasets/fiqa/",
        "instructions": {
            "corpus": "Instruct: Given a text related to finance, please describe the queries it may answer and intent of these queries. \n text:",
            "queries": "Instruct: Given a query relevant to finance, please describe the intent of this query. \n query: "
        },
        "output": "./fiqa/{split}.npz"
    }
}


def main():
    parser = argparse.ArgumentParser(description='Generate optimized embeddings')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--split', required=True, help='Data split (corpus, queries, etc.)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto if not set)')
    
    args = parser.parse_args()
    
    config = DATASETS[args.dataset]
    model = get_model()
    
    # Get instruction for this split
    if isinstance(config["instructions"], dict):
        instruction = config["instructions"].get(args.split, "")
    else:
        instruction = config["instructions"]
    
    # File paths
    input_file = os.path.join(config["dir"], f"{args.split}.jsonl")
    output_file = config["output"].format(split=args.split)
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Process
    process_texts(model, input_file, output_file, instruction, args.batch_size)


if __name__ == '__main__':
    main() 