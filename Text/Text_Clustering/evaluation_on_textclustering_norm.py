import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """
    Calculate Top-1 clustering accuracy using the Hungarian algorithm.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        confusion[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    total_correct = confusion[row_ind, col_ind].sum()
    return total_correct / len(y_pred)

def main(args):
    data = np.load(args.embedding_path)
    X = data['data']
    y = data['label']

    # Check if embeddings need normalization
    norms = np.linalg.norm(X, axis=1)
    is_normalized = np.allclose(norms, 1.0, atol=1e-3)
    
    if args.normalize or (not is_normalized and args.normalize is None):
        X = normalize(X, norm='l2')
        print(f"Applied L2 normalization to embeddings")
    elif is_normalized:
        print(f"Embeddings are already L2 normalized")
    
    print(f"Embedding shape: {X.shape}")

    # Auto-detect number of clusters from ground truth labels
    n_clusters_actual = len(np.unique(y))
    if args.n_clusters is None:
        n_clusters = n_clusters_actual
        print(f"Auto-detected {n_clusters} clusters from data labels")
    else:
        n_clusters = args.n_clusters
        print(f"Using specified {n_clusters} clusters (data has {n_clusters_actual} ground truth classes)")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=args.batch_size,
        n_init=args.n_init
    )
    kmeans.fit(X)
    y_pred = kmeans.labels_

    acc = cluster_accuracy(y, y_pred)
    print(f"Top-1 Clustering Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniBatchKMeans clustering and cluster accuracy calculation.")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the embedding .npz file")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters (auto-detect if not specified)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--n_init", type=str, default="auto", help="Number of initializations ('auto' or int)")
    parser.add_argument("--normalize", action='store_true', help="Force L2 normalization of embeddings")
    args = parser.parse_args()
    main(args)
