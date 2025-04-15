import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

RAW_DIR = Path("../data/raw/")
SAVE_VEC = Path("../data/encoded/vectors.npy")
SAVE_LIST = Path("../data/encoded/filenames.json")


def load_images(image_dir):
    image_paths = sorted(image_dir.glob("*.png"))
    return image_paths


def get_resnet50_encoder():
    model = models.resnet50(pretrained=True)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC
    return model


def encode_images(model, image_paths, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    vectors = []
    for path in tqdm(image_paths):
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).squeeze().cpu().numpy()
        vectors.append(feat)

    return np.stack(vectors), [str(p.name) for p in image_paths]


def main():
    os.makedirs(SAVE_VEC.parent, exist_ok=True)
    image_paths = load_images(RAW_DIR)
    model = get_resnet50_encoder()

    print(f"🚀 Đang encode {len(image_paths)} ảnh bằng ResNet50...")
    vectors, filenames = encode_images(model, image_paths)

    np.save(SAVE_VEC, vectors)
    with open(SAVE_LIST, "w") as f:
        json.dump(filenames, f)

    print(f"✅ Đã lưu vector vào: {SAVE_VEC}")
    print(f"✅ Danh sách file: {SAVE_LIST}")


if __name__ == "__main__":
    main()
