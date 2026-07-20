"""Train RetinexTapetum on a paired low-light dataset."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import LOLPairDataset, list_images
from losses import total_loss_fn
from model import RetinexTapetum


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_files(files, val_ratio, seed):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_ratio)))
    return sorted(files[val_count:]), sorted(files[:val_count])


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_psnr):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "base_channels": config.BASE_CHANNELS,
            "lambda_max": config.LAMBDA_MAX,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_psnr", -1.0))


def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    Path(config.CKPT_DIR).mkdir(parents=True, exist_ok=True)

    paired = [
        name for name in list_images(config.TRAIN_LOW_DIR)
        if os.path.exists(os.path.join(config.TRAIN_HIGH_DIR, name))
    ]
    if not paired:
        raise RuntimeError(f"No paired training images found under {config.DATA_ROOT}")

    train_files, val_files = split_files(paired, config.VAL_RATIO, config.SPLIT_SEED)
    train_set = LOLPairDataset(
        config.TRAIN_LOW_DIR,
        config.TRAIN_HIGH_DIR,
        crop_size=config.CROP_SIZE,
        training=True,
        file_list=train_files,
    )
    val_set = LOLPairDataset(
        config.TRAIN_LOW_DIR,
        config.TRAIN_HIGH_DIR,
        crop_size=config.CROP_SIZE,
        training=False,
        file_list=val_files,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = RetinexTapetum(
        base=config.BASE_CHANNELS,
        lambda_init=config.LAMBDA_INIT,
        lambda_max=config.LAMBDA_MAX,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.MIN_LR,
    )

    perceptual_fn = None
    if config.USE_LPIPS_LOSS:
        try:
            import lpips
            perceptual_fn = lpips.LPIPS(net=config.LPIPS_NET).to(device).eval()
            for parameter in perceptual_fn.parameters():
                parameter.requires_grad = False
        except Exception as exc:
            print(f"LPIPS disabled: {exc}")

    start_epoch, best_psnr = 1, -1.0
    resume_path = os.path.join(config.CKPT_DIR, config.RESUME_CKPT_NAME)
    if config.RESUME_TRAINING and os.path.exists(resume_path):
        start_epoch, best_psnr = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        print(f"Resuming at epoch {start_epoch}; best PSNR={best_psnr:.4f}")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(low, high)
            loss, _ = total_loss_fn(output, low, high, perceptual_fn=perceptual_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        model.eval()
        validation_scores = []
        with torch.no_grad():
            for batch in val_loader:
                low = batch["low"].to(device)
                high = batch["high"].to(device)
                pred = model(low)["enhanced"]
                validation_scores.append(psnr(pred, high).item())

        mean_loss = running_loss / max(len(train_loader), 1)
        mean_psnr = float(np.mean(validation_scores))
        print(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"loss={mean_loss:.6f} | val_psnr={mean_psnr:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )

        save_checkpoint(
            os.path.join(config.CKPT_DIR, "last.pth"),
            model,
            optimizer,
            scheduler,
            epoch,
            max(best_psnr, mean_psnr),
        )
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            save_checkpoint(
                os.path.join(config.CKPT_DIR, "best.pth"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
            )
            print(f"Saved new best checkpoint: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
