import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.clustering import run_hdbscan, reduce_umap, save_cluster_plot, save_cluster_config

ENCODED_PATH = Path("../data/encoded")
CLUSTER_PATH = Path("../data/cluster")
CLUSTER_PATH.mkdir(parents=True, exist_ok=True)

VEC_PATH = ENCODED_PATH / "vectors.npy"
NAME_PATH = ENCODED_PATH / "filenames.json"
LABEL_PATH = CLUSTER_PATH / "labels.npy"
PLOT_PATH = CLUSTER_PATH / "cluster_plot.png"
CONFIG_PATH = CLUSTER_PATH / "config.json"

vectors = np.load(VEC_PATH)
with open(NAME_PATH) as f:
    filenames = json.load(f)

print("🔍 Clustering với HDBSCAN...")
labels = run_hdbscan(vectors)
np.save(LABEL_PATH, labels)
print("✅ Đã lưu labels")

print("📊 Đang UMAP để visualize...")
embedding = reduce_umap(vectors)
save_cluster_plot(embedding, labels, PLOT_PATH)
save_cluster_config(labels, CONFIG_PATH)
