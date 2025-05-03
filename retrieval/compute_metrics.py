'''
Code to evaluate 1NN accuracies on ImageNet-1K dataset.
'''
import pandas as pd
import numpy as np
import time
import os
from argparse import ArgumentParser
parser=ArgumentParser()
parser.add_argument('--topk', default=8, type=int,help='the number of topk used for CSR encoding dir')
parser.add_argument('--eval_k', type=int, default=4, help='Value of K to calculate Top-K accuracy')
args = parser.parse_args()
topk = args.topk
root = f"./CSR_topk_{topk}/"
dataset = 'V1'
index_type = 'exactl2' # ['exactl2', 'hnsw_8', 'hnsw_32']
k = args.eval_k
k_for_loading = 2048

def compute_topk_accuracy(val_classes, db_classes, neighbors, k):
    """Computes Top-K accuracy"""
    num_queries = neighbors.shape[0]
    num_neighbors_found = neighbors.shape[1]
    actual_k = min(k, num_neighbors_found)
    
    if actual_k != k:
        print(f"Warning: Requested Top-{k}, but only {num_neighbors_found} neighbors found. Using Top-{actual_k}.")
    if actual_k == 0:
        return 0.0
    
    # top-k predictions
    topk_indices = neighbors[:, :actual_k]
    topk_labels = db_classes[topk_indices]
    correct_in_topk = np.zeros(num_queries, dtype=bool)
    for i in range(num_queries):
        target = val_classes[i]
        correct_in_topk[i] = target in topk_labels[i]

    return np.mean(correct_in_topk)

# Load database and query set for nested models
db_labels = np.load(root + f"V1_train_topk_{topk}-y.npy")
query_labels = np.load(root + f"V1_val_topk_{topk}-y.npy")

start = time.time()

# Load neighbors array
neighbors_path = root + "neighbors/" + index_type + "_" \
+ f"{k_for_loading}shortlist_" + dataset + ".csv"
print("Loading neighbors file: " + neighbors_path)
neighbors = pd.read_csv(neighbors_path, header=None).to_numpy()

# Top1 accuracy
top1_indices = neighbors[:, 0]
top1_labels = db_labels[top1_indices]
print("Top1= ", np.sum(top1_labels == query_labels) / query_labels.shape[0])
end = time.time()
print("Eval time for Top1 = %0.3f sec\n" % (end - start))

# Top-K accuracy
if k > 0:
    start_topk = time.time()
    topk_accuracy = compute_topk_accuracy(query_labels, db_labels, neighbors, k)
    print(f"Top-{k} accuracy = {topk_accuracy:.5f}")
    end_topk = time.time()
    print(f"Eval time for Top-{k} = {end_topk - start_topk:.3f} sec\n")