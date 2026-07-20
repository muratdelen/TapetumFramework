"""Evaluate PSNR, SSIM, and LPIPS for generated RetinexTapetum outputs."""

import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config


def list_images(folder):
    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(config.IMG_EXTS)
    )


def gaussian_window(window_size, channel, device):
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window = kernel.unsqueeze(1) @ kernel.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).item()


def calculate_ssim(pred, target, window_size=11):
    channel = pred.size(1)
    window = gaussian_window(window_size, channel, pred.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    score = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return score.mean().item()


def main():
    device = torch.device(config.DEVICE)
    to_tensor = transforms.ToTensor()
    names = [
        name for name in list_images(config.RESULT_DIR)
        if os.path.exists(os.path.join(config.TEST_HIGH_DIR, name))
    ]
    if not names:
        raise RuntimeError(
            f"No matched output/reference images found in {config.RESULT_DIR} and {config.TEST_HIGH_DIR}"
        )

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net=config.LPIPS_NET).to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}")

    rows = []
    for index, name in enumerate(names, start=1):
        pred = to_tensor(Image.open(os.path.join(config.RESULT_DIR, name)).convert("RGB")).unsqueeze(0).to(device)
        target = to_tensor(Image.open(os.path.join(config.TEST_HIGH_DIR, name)).convert("RGB")).unsqueeze(0).to(device)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

        psnr = calculate_psnr(pred, target)
        ssim = calculate_ssim(pred, target)
        lpips_value = float("nan")
        if lpips_fn is not None:
            with torch.no_grad():
                lpips_value = lpips_fn(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()
        rows.append({"image": name, "psnr": psnr, "ssim": ssim, "lpips": lpips_value})
        print(f"[{index}/{len(names)}] {name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_value:.4f}")

    output_csv = Path(config.RESULT_DIR) / "metrics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nDataset summary")
    print(f"Images : {len(rows)}")
    print(f"PSNR   : {np.mean([row['psnr'] for row in rows]):.4f}")
    print(f"SSIM   : {np.mean([row['ssim'] for row in rows]):.4f}")
    valid_lpips = [row["lpips"] for row in rows if not np.isnan(row["lpips"])]
    if valid_lpips:
        print(f"LPIPS  : {np.mean(valid_lpips):.4f}")
    print(f"CSV    : {output_csv}")


if __name__ == "__main__":
    main()
