import numpy as np
from pathlib import Path
import hdbscan
import umap
import matplotlib.pyplot as plt
import json

VEC_PATH = Path("../data/encoded/vectors.npy")
NAME_PATH = Path("../data/encoded/filenames.json")
LABEL_SAVE = Path("../data/cluster/labels.npy")
PLOT_SAVE = Path("../data/cluster/cluster_plot.png")
CONFIG_SAVE = Path("../data/cluster/config.json")

def zmain():
    print("🚀 Đang load vectors...")
    vectors = np.load(VEC_PATH)
    with open(NAME_PATH) as f:
        filenames = json.load(f)

    print("🔍 Clustering với HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(vectors)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ Đã phát hiện {n_clusters} cụm (UI states)")

    # Lưu nhãn
    LABEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    np.save(LABEL_SAVE, labels)

    config = {
        "method": "hdbscan",
        "n_clusters": int(n_clusters),
        "label_set": list(set(labels))
    }
    with open(CONFIG_SAVE, "w") as f:
        json.dump(config, f)

    print("📊 Đang reduce dimension bằng UMAP...")
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=8)
    plt.title("UI Cluster via HDBSCAN + UMAP")
    plt.savefig(PLOT_SAVE)
    print(f"🖼️ Plot saved to: {PLOT_SAVE}")

if __name__ == "__main__":
    main()
