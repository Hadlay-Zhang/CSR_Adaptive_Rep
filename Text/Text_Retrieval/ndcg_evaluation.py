"""
NDCG@10 Evaluation for Text Retrieval
"""

import argparse
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict


def load_embeddings(npz_path):
    """Load embeddings from npz file"""
    print(f"Loading embeddings from {npz_path}")
    data = np.load(npz_path)
    
    # trycommon keys
    if 'embeddings' in data:
        embeddings = data['embeddings']
    elif 'data' in data:
        embeddings = data['data']
    else:
        embeddings = data[list(data.keys())[0]]
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings


def load_queries(queries_path):
    """Load queries from jsonl file"""
    print(f"Loading queries from {queries_path}")
    queries = {}
    query_ids = []
    with open(queries_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            queries[data['_id']] = data['text']
            query_ids.append(data['_id'])
    return queries, query_ids


def load_corpus(corpus_path):
    """Load corpus from jsonl file"""
    print(f"Loading corpus from {corpus_path}")
    corpus = {}
    corpus_ids = []
    with open(corpus_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get('title', '') + ' ' + data.get('text', '')
            corpus[data['_id']] = text.strip()
            corpus_ids.append(data['_id'])
    return corpus, corpus_ids


def load_qrels(qrels_path):
    """Load qrels with graded relevance scores"""
    print(f"Loading qrels from {qrels_path}")
    qrels = defaultdict(dict)
    
    with open(qrels_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            query_id, corpus_id, score = parts[0], parts[1], int(parts[2])
            qrels[query_id][corpus_id] = score
    
    return dict(qrels)


def dcg(scores, k):
    """Calculate DCG@k"""
    scores = np.array(scores)[:k]
    if scores.size:
        return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def ndcg_at_k(retrieved_docs, qrels_dict, k):
    """Calculate NDCG@k using graded relevance"""
    # Get relevance scores for retrieved docs
    scores = [qrels_dict.get(doc_id, 0) for doc_id in retrieved_docs[:k]]
    
    # DCG@k
    dcg_k = dcg(scores, k)
    # ideal DCG@k
    ideal_scores = sorted(qrels_dict.values(), reverse=True)
    idcg_k = dcg(ideal_scores, k)
    
    return dcg_k / idcg_k if idcg_k > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='NDCG@k Evaluation for Text Retrieval')
    parser.add_argument('--corpus_embed', required=True, help='Corpus embeddings npz file')
    parser.add_argument('--queries_embed', required=True, help='Query embeddings npz file')
    parser.add_argument('--dataset_path', required=True, help='Dataset directory path')
    parser.add_argument('--k', type=int, default=10, help='Cut-off rank (default: 10)')
    parser.add_argument('--qrels_file', default='test.tsv', help='Qrels filename (default: test.tsv)')
    
    args = parser.parse_args()
    

    corpus_embeddings = load_embeddings(args.corpus_embed)
    query_embeddings = load_embeddings(args.queries_embed)
    queries_file = os.path.join(args.dataset_path, 'queries.jsonl')
    corpus_file = os.path.join(args.dataset_path, 'corpus.jsonl')
    qrels_file = os.path.join(args.dataset_path, 'qrels', args.qrels_file)
    
    queries, query_ids = load_queries(queries_file)
    corpus, corpus_ids = load_corpus(corpus_file)
    qrels = load_qrels(qrels_file)
    
    # Normalize embeddings for cosine similarity (L2)
    corpus_norm = normalize(corpus_embeddings, norm='l2')
    query_norm = normalize(query_embeddings, norm='l2')
    
    ndcg_scores = []
    for i in range(len(query_norm)):
        query_id = query_ids[i]
        if query_id not in qrels:
            continue
        # top-k most similar docs
        similarities = cosine_similarity(query_norm[i:i+1], corpus_norm).flatten()
        top_indices = np.argsort(similarities)[::-1][:args.k]
        retrieved_docs = [corpus_ids[idx] for idx in top_indices]
        # NDCG@k
        score = ndcg_at_k(retrieved_docs, qrels[query_id], args.k)
        ndcg_scores.append(score)
    
    final_ndcg = np.mean(ndcg_scores) * 100
    print(f"\nNDCG@{args.k}: {final_ndcg:.2f}%")
    print(f"Evaluated {len(ndcg_scores)} queries")


if __name__ == "__main__":
    main() 