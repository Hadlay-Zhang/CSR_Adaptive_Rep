"""
Convert original npz embed files to splitted npy files
"""

import os
import numpy as np
from tqdm import tqdm
import argparse

def convert_npz_to_batch_format(npz_path, output_path, batch_size=256):
    """
    Args:
        npz_path: Path to input npz file
        output_path: Output directory path
        batch_size: Batch size for splitting data
    """
    print(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    
    embeddings = data['data']  # Shape: (N, 2048)
    labels = data['label']     # Shape: (N, 1)
    
    print(f"Data shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    img_dir = os.path.join(output_path, 'img')
    label_dir = os.path.join(output_path, 'label')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    num_samples = embeddings.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Splitting {num_samples} samples into {num_batches} batches of size {batch_size}")
    
    for i in tqdm(range(num_batches), desc="Converting batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Save batch files
        np.save(os.path.join(img_dir, f'emb_{i}.npy'), batch_embeddings)
        np.save(os.path.join(label_dir, f'emb_{i}.npy'), batch_labels.squeeze())  # Remove extra dimension
    
    print(f"Conversion completed. Files saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert npz embed files to batch format')
    parser.add_argument('--train_npz', 
                        default='pretrained_embed/SoTA_RN50_Embeds/1K_train_sota.npz',
                        help='Path to training npz file')
    parser.add_argument('--val_npz', 
                        default='pretrained_embed/SoTA_RN50_Embeds/1K_val_sota.npz',
                        help='Path to validation npz file')
    parser.add_argument('--output_dir', 
                        default='retrieval/pretrained_emb',
                        help='Output directory for converted files')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256,
                        help='Batch size for splitting data')
    
    args = parser.parse_args()
    
    train_output = os.path.join(args.output_dir, 'train_emb')
    val_output = os.path.join(args.output_dir, 'val_emb')
    
    # convert data
    if os.path.exists(args.train_npz):
        print("Converting training data...")
        convert_npz_to_batch_format(args.train_npz, train_output, args.batch_size)
    
    if os.path.exists(args.val_npz):
        print("\nConverting validation data...")
        convert_npz_to_batch_format(args.val_npz, val_output, args.batch_size)

if __name__ == '__main__':
    main() 