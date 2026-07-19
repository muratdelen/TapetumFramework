"""Inference script for RetinexTapetum."""

import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from config import (
    DEVICE,
    TEST_LOW_DIR,
    CKPT_DIR,
    RESULT_DIR,
    BASE_CHANNELS,
    LAMBDA_INIT,
    LAMBDA_MAX,
    COLOR_RESTORE,
    COLOR_RESTORE_STRENGTH,
    COLOR_RESTORE_EPS,
)
from model import RetinexTapetum

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
os.makedirs(RESULT_DIR, exist_ok=True)


def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])


def infer_base_channels(checkpoint):
    state = checkpoint.get("model", checkpoint)
    head_weight = state.get("decom_net.head.weight")
    if head_weight is None:
        return BASE_CHANNELS
    return int(head_weight.shape[0])


def load_model():
    ckpt_path = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    checkpoint_base = infer_base_channels(checkpoint)
    model = RetinexTapetum(
        base=checkpoint_base,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def tensor_to_pil(x):
    x = x.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(x)


def rgb_to_luminance(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def restore_input_chroma(enhanced, low, strength=0.75, eps=1e-4):
    y_low = rgb_to_luminance(low)
    y_enh = rgb_to_luminance(enhanced)
    input_chroma = low / (y_low + eps)
    chroma_restored = torch.clamp(y_enh * input_chroma, 0.0, 1.0)
    return torch.clamp((1.0 - strength) * enhanced + strength * chroma_restored, 0.0, 1.0)


@torch.no_grad()
def run_test():
    model = load_model()
    to_tensor = transforms.ToTensor()
    files = list_images(TEST_LOW_DIR)

    for fname in tqdm(files, desc="Testing"):
        in_path = os.path.join(TEST_LOW_DIR, fname)
        out_path = os.path.join(RESULT_DIR, fname)
        img = Image.open(in_path).convert("RGB")
        inp = to_tensor(img).unsqueeze(0).to(DEVICE)
        enh = model(inp)["enhanced"]
        if COLOR_RESTORE:
            enh = restore_input_chroma(
                enh,
                inp,
                strength=COLOR_RESTORE_STRENGTH,
                eps=COLOR_RESTORE_EPS,
            )
        tensor_to_pil(enh).save(out_path)

    print("Results saved to:", RESULT_DIR)


if __name__ == "__main__":
    run_test()
