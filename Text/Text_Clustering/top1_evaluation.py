import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        confusion[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    return confusion[row_ind, col_ind].sum() / len(y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    args = parser.parse_args()
    
    data = np.load(args.embedding_path)
    X = normalize(data['data'], norm='l2')
    y = data['label']
    
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    acc = cluster_accuracy(y, y_pred)
    print(f"Top-1 Clustering Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main() 