import numpy as np
import hdbscan
import umap
import matplotlib.pyplot as plt
import json
from pathlib import Path


def run_hdbscan(vectors, min_cluster_size=5):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(vectors)
    return labels


def reduce_umap(vectors, n_components=2, random_state=42):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(vectors)
    return embedding


def save_cluster_plot(embedding, labels, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10)
    plt.title("UI Cluster via HDBSCAN + UMAP")
    plt.savefig(save_path)
    print(f"✅ Saved cluster plot to {save_path}")


def save_cluster_config(labels, save_path, method="hdbscan"):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    config = {
        "method": method,
        "n_clusters": int(n_clusters),
        "label_set": [int(x) for x in set(labels)]
    }
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved cluster config to {save_path}")
