"""Loss functions for RetinexTapetum."""

import torch
import torch.nn.functional as F
from config import (
    W_L1, W_SSIM, W_COLOR, W_CHROMA, W_ATTN, W_EDGE, W_DARK_NOISE,
    W_LPIPS, LPIPS_LOSS_RESIZE, W_RECON_LOW, W_RECON_HIGH, W_REFLECT,
    W_SMOOTH_LOW, W_SMOOTH_HIGH, W_SMOOTH_ENH,
)


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def create_gaussian_window(window_size, channel, device):
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()


def ssim_loss(pred, target, window_size=11):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    channel = pred.size(1)
    window = create_gaussian_window(window_size, channel, pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return 1 - ssim_map.mean()


def color_consistency_loss(pred, target):
    return F.l1_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))


def rgb_to_luminance(x):
    if x.size(1) == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def chroma_consistency_loss(pred, target, eps=1e-3):
    pred_chroma = torch.clamp(pred / (rgb_to_luminance(pred) + eps), 0.0, 4.0)
    target_chroma = torch.clamp(target / (rgb_to_luminance(target) + eps), 0.0, 4.0)
    return F.l1_loss(pred_chroma, target_chroma)


def attention_regularization(t):
    return torch.mean(torch.abs(t)) + 0.03 * torch.mean(t ** 2)


def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def edge_detail_loss(pred, target):
    pred_gray, target_gray = rgb_to_luminance(pred), rgb_to_luminance(target)
    return charbonnier_loss(gradient_x(pred_gray), gradient_x(target_gray)) + charbonnier_loss(
        gradient_y(pred_gray), gradient_y(target_gray)
    )


def dark_region_chroma_noise_loss(pred, target, low, threshold=0.35):
    dark_weight = torch.clamp((threshold - rgb_to_luminance(low)) / threshold, 0.0, 1.0)
    pred_chroma = pred - rgb_to_luminance(pred)
    target_chroma = target - rgb_to_luminance(target)
    wx, wy = dark_weight[:, :, :, :-1], dark_weight[:, :, :-1, :]
    return charbonnier_loss(gradient_x(pred_chroma) * wx, gradient_x(target_chroma) * wx) + charbonnier_loss(
        gradient_y(pred_chroma) * wy, gradient_y(target_chroma) * wy
    )


def lpips_perceptual_loss(pred, target, perceptual_fn):
    if perceptual_fn is None:
        return pred.new_zeros(())
    pred_lpips, target_lpips = pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)
    if LPIPS_LOSS_RESIZE and LPIPS_LOSS_RESIZE > 0:
        size = (LPIPS_LOSS_RESIZE, LPIPS_LOSS_RESIZE)
        pred_lpips = F.interpolate(pred_lpips, size=size, mode="bilinear", align_corners=False)
        target_lpips = F.interpolate(target_lpips, size=size, mode="bilinear", align_corners=False)
    return perceptual_fn(pred_lpips * 2.0 - 1.0, target_lpips * 2.0 - 1.0).mean()


def illumination_smoothness_loss(l_map, guidance):
    gray_x, gray_l = rgb_to_luminance(guidance), rgb_to_luminance(l_map)
    weight_x = torch.exp(-10.0 * torch.abs(gradient_x(gray_x)))
    weight_y = torch.exp(-10.0 * torch.abs(gradient_y(gray_x)))
    return torch.mean(torch.abs(gradient_x(gray_l)) * weight_x) + torch.mean(
        torch.abs(gradient_y(gray_l)) * weight_y
    )


def decomposition_loss(output, low, high):
    loss_recon_low = charbonnier_loss(output["recon_low"], low)
    loss_recon_high = charbonnier_loss(output["recon_high"], high)
    loss_reflect = F.l1_loss(output["reflectance_low"], output["reflectance_high"])
    loss_smooth_low = illumination_smoothness_loss(output["illumination_low"], low)
    loss_smooth_high = illumination_smoothness_loss(output["illumination_high"], high)
    loss_smooth_enh = illumination_smoothness_loss(output["illumination_t"], low)
    total = (
        W_RECON_LOW * loss_recon_low + W_RECON_HIGH * loss_recon_high + W_REFLECT * loss_reflect
        + W_SMOOTH_LOW * loss_smooth_low + W_SMOOTH_HIGH * loss_smooth_high
        + W_SMOOTH_ENH * loss_smooth_enh
    )
    logs = {
        "decomp": total.item(), "recon_low": loss_recon_low.item(),
        "recon_high": loss_recon_high.item(), "reflect": loss_reflect.item(),
        "smooth_low": loss_smooth_low.item(), "smooth_high": loss_smooth_high.item(),
        "smooth_enh": loss_smooth_enh.item(),
    }
    return total, logs


def total_loss_fn(output, low, gt, perceptual_fn=None, lpips_weight=None):
    pred, t_map = output["enhanced"], output["tapetum_attention"]
    effective_lpips_weight = W_LPIPS if lpips_weight is None else lpips_weight
    l1 = charbonnier_loss(pred, gt)
    ssim_l = ssim_loss(pred, gt)
    color_l = color_consistency_loss(pred, gt)
    chroma_l = chroma_consistency_loss(pred, gt)
    attn_l = attention_regularization(t_map)
    edge_l = edge_detail_loss(pred, gt)
    dark_noise_l = dark_region_chroma_noise_loss(pred, gt, low)
    lpips_l = lpips_perceptual_loss(pred, gt, perceptual_fn)
    decomp_l, decomp_logs = decomposition_loss(output, low, gt)
    total = (
        W_L1 * l1 + W_SSIM * ssim_l + W_COLOR * color_l + W_CHROMA * chroma_l
        + W_ATTN * attn_l + W_EDGE * edge_l + W_DARK_NOISE * dark_noise_l
        + effective_lpips_weight * lpips_l + decomp_l
    )
    logs = {
        "total": total.item(), "l1": l1.item(), "ssim": ssim_l.item(),
        "color": color_l.item(), "chroma": chroma_l.item(), "attn": attn_l.item(),
        "edge": edge_l.item(), "dark_noise": dark_noise_l.item(), "lpips": lpips_l.item(),
        **decomp_logs,
    }
    return total, logs
