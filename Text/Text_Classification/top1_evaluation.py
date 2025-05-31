"""
Top-1 accuracy evaluation for text classification
"""

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_embeddings(npz_path):
    """Load embeddings and labels from npz file"""
    print(f"Loading {npz_path}")
    data = np.load(npz_path)
    embeddings = data['data']
    labels = data['label']
    print(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]}D embeddings")
    return embeddings, labels


def train_classifier(train_embeddings, train_labels):
    """Train a simple linear classifier"""    
    # encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(train_labels)
    # train logistic regression classifier
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, encoded_labels)
    
    print(f"Trained on {len(np.unique(encoded_labels))} classes")
    return classifier, label_encoder


def evaluate_classifier(classifier, label_encoder, test_embeddings, test_labels):
    """Evaluate classifier and compute top-1 accuracy"""
    encoded_test_labels = label_encoder.transform(test_labels)
    # predict
    predictions = classifier.predict(test_embeddings)
    # top-1 acc
    accuracy = accuracy_score(encoded_test_labels, predictions)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Top-1 accuracy evaluation for classification')
    parser.add_argument('--train_embedding_path', required=True, help='Path to train embeddings npz file')
    parser.add_argument('--test_embedding_path', required=True, help='Path to test embeddings npz file')
    args = parser.parse_args()

    train_embeddings, train_labels = load_embeddings(args.train_embedding_path)
    test_embeddings, test_labels = load_embeddings(args.test_embedding_path)
    
    classifier, label_encoder = train_classifier(train_embeddings, train_labels)
    
    accuracy = evaluate_classifier(classifier, label_encoder, test_embeddings, test_labels)
    
    print(f"\nTop-1 Accuracy: {accuracy * 100:.2f}%")
    print(f"Evaluated on {len(test_embeddings)} test samples")


if __name__ == "__main__":
    main() 