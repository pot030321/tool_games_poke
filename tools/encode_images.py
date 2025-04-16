import os
import json
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.encoder import get_encoder, encode_image

RAW_DIR = Path("../data/review")
ENCODED_DIR = Path("../data/encoded")
ENCODED_DIR.mkdir(parents=True, exist_ok=True)

vectors = []
filenames = []

model = get_encoder()

for img_name in sorted(os.listdir(RAW_DIR)):
    if not img_name.lower().endswith(".png"):
        continue
    img_path = RAW_DIR / img_name
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Không đọc được ảnh: {img_path}")
        continue
    vec = encode_image(img, model)
    vectors.append(vec)
    filenames.append(img_name)

vectors = np.array(vectors)
np.save(ENCODED_DIR / "vectors.npy", vectors)

with open(ENCODED_DIR / "filenames.json", "w") as f:
    json.dump(filenames, f)

print(f"✅ Đã encode {len(vectors)} ảnh và lưu vào {ENCODED_DIR}")