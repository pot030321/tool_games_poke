import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def get_encoder():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


def encode_image(img, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(x).squeeze().cpu().numpy()
    return vec