import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def rescale(x):
    return (x + 1.) / 2.


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    return Image.fromarray(x)


@torch.no_grad()
def sample_class_conditional(model, sampler, class_label, batch_size, steps, scale, eta, device):
    print(f"Sampling class {class_label} | steps={steps}, scale={scale}, eta={eta}")

    # 条件embedding
    xc = torch.tensor([class_label] * batch_size, device=device)
    c = model.get_learned_conditioning({model.cond_stage_key: xc})

    # 自动获取最后一个 embedding ID（作为无条件 token）
    n_total_classes = model.cond_stage_model.embedding.num_embeddings

    uncond_id = n_total_classes - 1
    uc = model.get_learned_conditioning(
        {model.cond_stage_key: torch.tensor([uncond_id]*batch_size, device=device)}
    )

    # DDIM采样
    samples, _ = sampler.sample(
        S=steps,
        conditioning=c,
        batch_size=batch_size,
        shape=[3, 32, 32],
        verbose=False,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=eta
    )
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp(rescale(x_samples), 0.0, 1.0)
    return x_samples


def save_images(images, outdir, start_idx, class_label):
    n_saved = start_idx
    for img in images:
        pil = custom_to_pil(img)
        pil.save(os.path.join(outdir, f"class{class_label}_{n_saved:06}.png"))
        n_saved += 1
    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", type=str, required=True,
                        help="path to model checkpoint or its log directory")
    parser.add_argument("-l", "--logdir", type=str, default="samples",
                        help="where to save samples")
    parser.add_argument("-c", "--custom_steps", type=int, default=50,
                        help="DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="eta for DDIM")
    parser.add_argument("--scale", type=float, default=3.0,
                        help="classifier-free guidance scale")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="batch size for sampling")
    parser.add_argument("--per_class", type=int, default=50,
                        help="how many images per class to sample")
    parser.add_argument("--classes", type=int, nargs="+",default=list(range(100)),
                        help="class labels to generate (e.g. 0 12 87)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use (default: 0)")
    args = parser.parse_args()

    # 设定设备
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # 加载配置和权重
    if os.path.isdir(args.resume):
        logdir = args.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")
        config_file = os.path.join(logdir, "config.yaml")
    else:
        ckpt = args.resume
        logdir = os.path.dirname(ckpt)
        config_file = os.path.join(logdir, "config.yaml")

    print(f"Using checkpoint: {ckpt}")
    print(f"Using config: {config_file}")

    config = OmegaConf.load(config_file)
    pl_sd = torch.load(ckpt, map_location="cpu")

    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outdir = os.path.join(args.logdir, f"samples_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving images to: {outdir}")

    # 类别采样循环
    for class_label in args.classes:
        print(f"Sampling class {class_label}")
        n_done = 0
        with torch.no_grad():
            while n_done < args.per_class:
                bs = min(args.batch_size, args.per_class - n_done)
                samples = sample_class_conditional(
                    model, sampler,
                    class_label=class_label,
                    batch_size=bs,
                    steps=args.custom_steps,
                    scale=args.scale,
                    eta=args.eta,
                    device=device
                )
                n_done = save_images(samples, outdir, n_done, class_label)
        print(f"Class {class_label}: finished {n_done} images")


if __name__ == "__main__":
    main()
