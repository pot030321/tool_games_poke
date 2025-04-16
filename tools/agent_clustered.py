import json
import numpy as np
import cv2
import time
from pathlib import Path
from core.encoder import get_encoder, encode_image
from core.adb import adb_screenshot, adb_tap, adb_start_game
from core.agent import ClusteredAgent
from action_review import save_for_review

# Paths
VECTOR_PATH = Path("../data/encoded/vectors.npy")
LABEL_PATH = Path("../data/cluster/labels.npy")
ACTION_PATH = Path("../data/cluster/cluster_actions.json")


def run_agent():
    print("🤖 AgentClustered đang chạy...")

    print("🚀 Đang mở game...")
    adb_start_game()
    print("⏳ Đợi game load...")
    time.sleep(7)

    print("🔍 Load data...")
    all_vecs = np.load(VECTOR_PATH)
    all_labels = np.load(LABEL_PATH)
    with open(ACTION_PATH) as f:
        action_map = json.load(f)

    encoder = get_encoder()
    agent = ClusteredAgent(encoder, all_vecs, all_labels, action_map, confidence_threshold=0.90)

    while True:
        img = adb_screenshot()
        if img is None:
            print("❌ Không chụp được màn hình")
            continue

        vec = encode_image(img, encoder)
        cluster, confidence = agent.predict_cluster(vec)
        print(f"🧠 Cluster nhận dạng: {cluster} (confidence={confidence:.4f})")

        if agent.should_review(cluster, confidence):
            print("⚠️ UI chưa rõ hoặc chưa gán → chuyển sang review mode")
            save_for_review(img, vec, cluster)
            continue

        action = agent.get_action(cluster)
        if action.startswith("tap:"):
            parts = action.split(":")
            x, y = map(int, parts[1].split(","))
            print(f"👆 TAP tại ({x}, {y})")
            adb_tap(x, y)
        elif action == "wait":
            print("⏳ Đang chờ...")
        elif action == "done":
            print("✅ Flow kết thúc")
            break
        elif action == "ignore":
            print("🚫 Bỏ qua cụm này.")
            continue


if __name__ == "__main__":
    run_agent()
