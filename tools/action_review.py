import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
PENDING_DIR = Path("data/cluster/pending_review")
PENDING_DIR.mkdir(parents=True, exist_ok=True)

def save_for_review(img, vec, cluster_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = PENDING_DIR / f"ui_{timestamp}.png"
    vec_path = PENDING_DIR / f"ui_{timestamp}.npy"

    cv2.imwrite(str(img_path), img)
    np.save(str(vec_path), vec)

    print(f"📥 Đã lưu UI chưa rõ tại: {img_path}")
    print("❓ Bạn muốn làm gì với UI này?")
    print("  a) Gán hành động ngay")
    print("  b) Bỏ qua, xử lý sau")
    print("  c) Gán là 'ignore' luôn")
    choice = input("👉 Chọn [a/b/c]: ").strip().lower()

    if choice == "a":
        action = input("⤵ Gõ hành động (vd: tap:800,600 / wait / done): ").strip()
        update_action_map(cluster_id, action)
        print("✅ Đã gán hành động cho cluster này.")
    elif choice == "c":
        update_action_map(cluster_id, "ignore")
        print("✅ Đã đánh dấu cụm này là 'ignore'")
    else:
        print("💤 Đã lưu, xử lý sau trong pending_review/")


def update_action_map(cluster_id, action):
    ACTION_PATH = Path("data/cluster/cluster_actions.json")
    if ACTION_PATH.exists():
        with open(ACTION_PATH) as f:
            actions = json.load(f)
    else:
        actions = {}

    actions[str(cluster_id)] = action

    with open(ACTION_PATH, "w") as f:
        json.dump(actions, f, indent=2)


if __name__ == "__main__":
    print("🧪 Đây là module dùng trong agent. Chạy riêng để debug nếu cần.")