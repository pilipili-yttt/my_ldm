import torch
import numpy as np
import pickle
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
from scipy.linalg import sqrtm


def load_resnet(device):
    model = resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def normalize_images(images):
    """
    Apply ImageNet normalization to a batch of images (B, 3, H, W)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize(images)


def calculate_fid(model, real_images, fake_images):
    with torch.no_grad():
        real_feats = model(real_images).view(real_images.shape[0], -1)
        fake_feats = model(fake_images).view(fake_images.shape[0], -1)

    mu1 = real_feats.mean(dim=0)
    mu2 = fake_feats.mean(dim=0)

    sigma1 = np.cov(real_feats.cpu().numpy().T)
    sigma2 = np.cov(fake_feats.cpu().numpy().T)

    diff = mu1.cpu().numpy() - mu2.cpu().numpy()
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# === 0. 设置设备 ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 1. 加载生成图像 ===
gen_npz = np.load("models/ldm/cifar5_vae/cifar5_vae/samples/00054740/2025-07-12-15-27-36/numpy/9984x32x32x3-samples.npz")
gen_images = torch.tensor(gen_npz["arr_0"]).permute(0, 3, 1, 2).float() / 255.0

# === 2. 加载真实图像 ===
with open("/data/cifar-100/cifar-100-python/test", "rb") as f:
    data = pickle.load(f, encoding="latin1")
real_images = data["data"].reshape(-1, 3, 32, 32)
real_images = torch.tensor(real_images).float() / 255.0

# === 3. 对齐数量，标准化，并移到设备上 ===
N = min(real_images.shape[0], gen_images.shape[0])
real_images = real_images[:N]
gen_images = gen_images[:N]

real_images = normalize_images(real_images)
gen_images = normalize_images(gen_images)

real_images = real_images.to(device)
gen_images = gen_images.to(device)

# === 4. 计算 FID ===
model = load_resnet(device)
fid = calculate_fid(model, real_images, gen_images)
print(f"FID Score: {fid:.4f}")
