import argparse
import os
import torch.nn.functional as F
from transformers import AutoModel
import json
from tqdm import tqdm
import numpy as np
import torch


def get_model():
    model_name = "nvidia/NV-Embed-v2"
    print(f"Loading model: {model_name}")
    
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map="auto"
    )    
    return model


def find_batch_size(model, sample_text, instruction):
    """Find optimal batch size for GPU memory"""
    for batch_size in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
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


def process_file(file_path, embed_path, model, instruction, batch_size):
    # Read all data first
    all_texts = []
    all_labels = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            all_texts.append(obj['text'])
            all_labels.append(obj['label'])
    
    # auto-detect batch size if needed
    if batch_size is None:
        batch_size = find_batch_size(model, all_texts[0], instruction)

    embeddings_list = []
    current_batch = batch_size
    i = 0
    with tqdm(total=len(all_texts), desc=f"Processing {os.path.basename(file_path)}") as pbar:
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
    
    embedding_array = np.vstack(embeddings_list)
    labels_array = np.array(all_labels)
    
    torch.cuda.empty_cache()
    
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    np.savez(embed_path, data=embedding_array, label=labels_array)
    print(f"Saved embeddings to {embed_path}")


DATASETS = {
    "banking77": {
        "instruction": "Instruct: Given a question, please describe the intent of this question. \n Question: ",
        "splits": ["train", "test"],
        "file_template": "../datasets/banking77/{split}.jsonl",
        "embed_template": "./Banking77/{split}.npz",
        "languages": None, 
    },
    "tweet_sentiment_extraction": {
        "instruction": "Instruct: Given a sentence, please describe the sentiment behind it. Whether is it positive, neutral or negative? \n Sentence: ",
        "splits": ["train", "test"],
        "file_template": "../datasets/tweet_sentiment_extraction/{split}.jsonl",
        "embed_template": "./tweet_sentiment_extraction/{split}.npz",
        "languages": None, 
    },
    "mtop_intent": {
        "instruction": "Instruct: Given a question, please describe the intent of this question. \n Question: ",
        "splits": ["train", "validation", "test"],
        "languages": ['de', 'en', 'es', 'fr', 'hi', 'th'],
        "file_template": "../datasets/mtop_intent/{language}/{split}.jsonl",
        "embed_template": "./mtop_intent/{language}/{split}.npz",
    }
}


def main():
    parser = argparse.ArgumentParser(description='Generate optimized embeddings for classification')
    parser.add_argument('--dataset', required=True, choices=DATASETS.keys())
    parser.add_argument('--language', default=None, help='Language (for mtop_intent)')
    parser.add_argument('--split', default=None, help='Data split (default: all splits)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto if not set)')

    args = parser.parse_args()
    config = DATASETS[args.dataset]
    model = get_model()

    if config["languages"]: # only for mtop_intent
        languages = [args.language] if args.language else config["languages"]
        splits = [args.split] if args.split else config["splits"]
        
        for language in languages:
            for split in splits:
                file_path = config["file_template"].format(language=language, split=split)
                embed_path = config["embed_template"].format(language=language, split=split)
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}, skipping.")
                    continue
                
                print(f"Processing {args.dataset} | language={language}, split={split}")
                process_file(file_path, embed_path, model, config["instruction"], args.batch_size)
    else:
        splits = [args.split] if args.split else config["splits"]
        
        for split in splits:
            file_path = config["file_template"].format(split=split)
            embed_path = config["embed_template"].format(split=split)
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}, skipping.")
                continue
            
            print(f"Processing {args.dataset} | split={split}")
            process_file(file_path, embed_path, model, config["instruction"], args.batch_size)


if __name__ == "__main__":
    main() 