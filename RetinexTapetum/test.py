"""Run RetinexTapetum inference on the configured test split."""

import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import config
from model import RetinexTapetum


def list_images(folder):
    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(config.IMG_EXTS)
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = int(checkpoint.get("base_channels", config.BASE_CHANNELS))
    lambda_max = float(checkpoint.get("lambda_max", config.LAMBDA_MAX))
    model = RetinexTapetum(
        base=base_channels,
        lambda_init=config.LAMBDA_INIT,
        lambda_max=lambda_max,
    ).to(device)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    device = torch.device(config.DEVICE)
    checkpoint_path = os.environ.get(
        "RETINEX_TAPETUM_CHECKPOINT",
        os.path.join(config.CKPT_DIR, "best.pth"),
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model or set "
            "RETINEX_TAPETUM_CHECKPOINT."
        )

    Path(config.RESULT_DIR).mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint_path, device)
    to_tensor = transforms.ToTensor()

    names = list_images(config.TEST_LOW_DIR)
    if not names:
        raise RuntimeError(f"No test images found in {config.TEST_LOW_DIR}")

    with torch.no_grad():
        for index, name in enumerate(names, start=1):
            image = Image.open(os.path.join(config.TEST_LOW_DIR, name)).convert("RGB")
            low = to_tensor(image).unsqueeze(0).to(device)
            enhanced = model(low)["enhanced"].clamp(0.0, 1.0)
            output_path = os.path.join(config.RESULT_DIR, name)
            save_image(enhanced.cpu(), output_path)
            print(f"[{index}/{len(names)}] {output_path}")


if __name__ == "__main__":
    main()
