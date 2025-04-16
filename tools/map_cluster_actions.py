import numpy as np
import json
import cv2
from pathlib import Path

RAW_DIR = Path("../data/raw")
ENCODED_PATH = Path("../data/encoded")
CLUSTER_PATH = Path("../data/cluster")

LABELS_PATH = CLUSTER_PATH / "labels.npy"
FILENAMES_PATH = ENCODED_PATH / "filenames.json"
ACTIONS_PATH = CLUSTER_PATH / "cluster_actions.json"

labels = np.load(LABELS_PATH)
with open(FILENAMES_PATH) as f:
    filenames = json.load(f)

unique_clusters = sorted(set(labels))
action_map = {}

for cluster_id in unique_clusters:
    if cluster_id == -1:
        print(f"⚠️ Cluster -1 (noise) → tự động gán 'ignore'")
        action_map[str(cluster_id)] = "ignore"
        continue

    print(f"\n🔍 Gán hành động cho Cluster {cluster_id}:")
    sample_idxs = [i for i, l in enumerate(labels) if l == cluster_id]

    for idx in sample_idxs[:5]:  # hiển thị 5 ảnh mẫu
        img_path = RAW_DIR / filenames[idx]
        img = cv2.imread(str(img_path))
        if img is not None:
            cv2.imshow(f"Cluster {cluster_id}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("👉 Gõ hành động cho cụm này:")
    print("  - tap:x,y      → vd: tap:800,600")
    print("  - wait         → chờ (không làm gì)")
    print("  - done         → kết thúc luồng")
    print("  - ignore       → bỏ qua cụm này")
    action = input("⤵ Nhập action: ").strip()
    action_map[str(cluster_id)] = action

with open(ACTIONS_PATH, "w") as f:
    json.dump(action_map, f, indent=2)

print(f"✅ Đã lưu vào {ACTIONS_PATH}")