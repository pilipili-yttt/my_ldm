import os
import random
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets
from PIL import Image


def load_inception_model(device):
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(device)
    return inception


def get_activations_from_dataloader(dataloader, model, device, limit=None):
    model.eval()
    activations = []
    total_seen = 0
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Extracting real image activations"):
            if limit is not None and total_seen >= limit:
                break
            remaining = None if limit is None else limit - total_seen
            if remaining is not None and x.size(0) > remaining:
                x = x[:remaining]
            x = x.to(device)
            pred = model(x)
            activations.append(pred.cpu().numpy())
            total_seen += x.size(0)
    activations = np.concatenate(activations, axis=0)
    if limit is not None:
        activations = activations[:limit]
    return activations


def get_activations_from_folder(folder, transform, model, device, batch_size=32, limit=None, seed=42):
    image_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if limit is not None and limit < len(image_files):
        random.seed(seed)
        image_files = random.sample(image_files, limit)

    activations = []
    batch = []
    with torch.no_grad():
        for path in tqdm(image_files, desc=f"Processing {len(image_files)} generated images"):
            img = Image.open(path).convert('RGB')
            img = transform(img).unsqueeze(0)
            batch.append(img)
            if len(batch) == batch_size:
                images = torch.cat(batch, dim=0).to(device)
                pred = model(images)
                activations.append(pred.cpu().numpy())
                batch = []
        if batch:
            images = torch.cat(batch, dim=0).to(device)
            pred = model(images)
            activations.append(pred.cpu().numpy())

    activations = np.concatenate(activations, axis=0)
    if limit is not None:
        activations = activations[:limit]
    return activations


def compute_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return (fid)


def main():
    # ---- Config ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cifar_root = "/data/cifar-100"
    gen_folder = "models/ldm/cifar5_12/cifar5_12/samples/00041055/2025-07-13-22-55-19/img"
    sample_count = 50000         # How many real samples to use
    gen_sample_count = 10000     # How many generated samples to use
    batch_size = 128

    # ---- Transform ----
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # ---- Model ----
    inception = load_inception_model(device)

    # ---- Real images ----
    print("\nLoading and extracting real CIFAR100 test images...")
    real_dataset = datasets.CIFAR100(root=cifar_root, train=True, download=False, transform=transform)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    act_real = get_activations_from_dataloader(real_loader, inception, device, limit=sample_count)
    print(f"Real activations shape: {act_real.shape}")

    # ---- Generated images ----
    print("\nExtracting generated images activations...")
    act_gen = get_activations_from_folder(gen_folder, transform, inception, device,
                                           batch_size=batch_size, limit=gen_sample_count)
    print(f"Generated activations shape: {act_gen.shape}")

    # ---- FID ----
    mu_real, sigma_real = compute_statistics(act_real)
    mu_gen, sigma_gen = compute_statistics(act_gen)

    print("\nCalculating FID...")
    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"\nFID score = {fid:.4f}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

 
