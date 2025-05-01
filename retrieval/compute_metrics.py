import pandas as pd
import numpy as np
import time
import os
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('--topk', default=8, type=int,help='the number of topk used for CSR encoding dir')
parser.add_argument('--eval_k', type=int, default=1, help='Value of K to calculate MAP@K')
args = parser.parse_args()
topk = args.topk
root = f"./CSR_topk_{topk}/"
dataset = 'V1'
index_type = 'exactl2' # ['exactl2', 'hnsw_8', 'hnsw_32']
k = args.eval_k
k_for_loading = 2048

def compute_map_at_k(val_classes, db_classes, neighbors, k):
    """
    Computes Mean Average Precision @ k (MAP@k).

    Args:
        val_classes (np.ndarray): Array of shape (num_queries,) containing the true class labels for each query.
        db_classes (np.ndarray): Array of shape (num_db_items,) containing the class labels for each item in the database.
        neighbors (np.ndarray): Array of shape (num_queries, num_neighbors_found) containing the indices
                                 of retrieved neighbors for each query, ordered by rank (closest first).
                                 num_neighbors_found should be >= k.
        k (int): The cutoff rank K for calculating MAP@K.

    Returns:
        float: The Mean Average Precision @ k score.
    """
    num_queries = neighbors.shape[0]
    num_neighbors_found = neighbors.shape[1]

    actual_k = min(k, num_neighbors_found)
    if actual_k != k:
        print(f"Warning: Requested MAP@{k}, but only {num_neighbors_found} neighbors were found in the file. Calculating MAP@{actual_k}.")
    if actual_k == 0:
        return 0.0

    average_precisions = []
    for i in range(num_queries):
        target = val_classes[i]
        indices = neighbors[i, :actual_k]
        labels = db_classes[indices]
        matches = (labels == target)
        if not np.any(matches):
            average_precisions.append(0.0)
            continue

        precision_values = []
        hits = 0
        for j in range(actual_k):
            if matches[j]:
                hits += 1
                precision_at_j = hits / (j + 1.0) # Precision@j+1
                precision_values.append(precision_at_j)

        # Average Precision for this query is the mean of precision values calculated at the ranks of relevant items.
        ap = np.mean(precision_values) if precision_values else 0.0
        average_precisions.append(ap)

    # MAP@k
    map_at_k = np.mean(average_precisions)
    return map_at_k


# Load database and query set for nested models

# Database: 1.2M x 1 for imagenet1k
db_labels = np.load(root + f"V1_train_topk_{topk}-y.npy")

# Query set: 50K x 1 for imagenet1k
query_labels = np.load(root + f"V1_val_topk_{topk}-y.npy")


start = time.time()
# Load database and query set for fixed feature models
# Load neighbors array and compute metrics

neighbors_path = root + "neighbors/" + index_type + "_" \
                 + f"{k_for_loading}shortlist_" + dataset + ".csv"
print("Loading neighbors file: " + neighbors_path)
neighbors = pd.read_csv(neighbors_path, header=None).to_numpy()

# Top1
top1_indices = neighbors[:, 0]
top1_labels = db_labels[top1_indices]
print("Top1= ", np.sum(top1_labels == query_labels) / query_labels.shape[0])

end = time.time()
print("Eval time for Top1 = %0.3f sec\n" % (end - start))

# MAP@k
if k > 0:
    start_map = time.time()
    map_at_k_value = compute_map_at_k(query_labels, db_labels, neighbors, k)
    print(f"MAP@{k} = {map_at_k_value:.5f}")
    end_map = time.time()
    print(f"Eval time for MAP@{k} = {end_map - start_map:.3f} sec\n")