# RetinexTapetum

This directory contains the executable RetinexTapetum implementation used for training, inference, and full-reference evaluation.

## Source files

- `model.py` — RetinexTapetum architecture
- `config.py` — dataset selection, paths, hyperparameters, and checkpoint settings
- `dataset.py` — paired low-light/normal-light dataset loader
- `losses.py` — enhancement, perceptual, structural, chromatic, and Retinex decomposition losses
- `train.py` — dataset-specific training and validation-based checkpoint selection
- `test.py` — checkpoint inference on the configured test split
- `evaluate_metrics.py` — PSNR, SSIM, and LPIPS evaluation
- `requirements.txt` — Python dependencies

## Install

```bash
git clone https://github.com/muratdelen/TapetumFramework.git
cd TapetumFramework/RetinexTapetum
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version when GPU execution is required.

## Dataset layout

The default local dataset location is `TapetumFramework/datasets/`. Images must be paired by filename.

```text
datasets/
├── LOL-v1/
│   ├── Train/Low
│   ├── Train/Normal
│   ├── Test/Low
│   └── Test/Normal
├── LOL-v2/
│   ├── Real_captured/
│   │   ├── Train/Low
│   │   ├── Train/Normal
│   │   ├── Test/Low
│   │   └── Test/Normal
│   └── Synthetic/
│       └── ...
└── UHD-LL down4/
    └── ...
```

Datasets and experiment artifacts are available from the TAPETUM archive where redistribution is permitted:

- RetinexTapetum project folder: https://drive.google.com/drive/folders/1kkPLkUoDK_Zvo9jOW7GSt0xoiFatF9Hb?usp=drive_link
- Main experiment archive: https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT?usp=drive_link

Third-party datasets and baseline materials remain subject to their original licenses.

## Select a dataset

Configuration can be controlled with environment variables.

### LOL-v2 Real-Captured

```bash
export RETINEX_DATA_NAME="LOL-v2"
export RETINEX_DATA_VARIANT="Real_captured"
```

### LOL-v2 Synthetic

```bash
export RETINEX_DATA_NAME="LOL-v2"
export RETINEX_DATA_VARIANT="Synthetic"
```

### LOL-v1

```bash
export RETINEX_DATA_NAME="LOL-v1"
export RETINEX_DATA_VARIANT="none"
```

### UHD-LL down4

```bash
export RETINEX_DATA_NAME="UHD-LL down4"
export RETINEX_DATA_VARIANT="none"
```

On Windows PowerShell, use `$env:RETINEX_DATA_NAME="LOL-v2"` and equivalent commands.

## Train

```bash
python train.py
```

The default manuscript-oriented configuration uses:

- separate training for each dataset
- 120 epochs
- batch size 2
- 256 × 256 aligned random crops
- seed 42
- Adam optimization
- cosine annealing
- validation PSNR checkpoint selection
- bounded amplification with `lambda_max = 1.35`

Checkpoints are written to the dataset-specific `checkpoints/` directory. `last.pth` is updated every epoch and `best.pth` stores the checkpoint with the highest validation PSNR.

### Resume training

```bash
export RETINEX_TAPETUM_RESUME=true
export RETINEX_TAPETUM_RESUME_CKPT=last.pth
python train.py
```

## Test

```bash
python test.py
```

By default, inference loads `best.pth`. A different checkpoint can be selected with:

```bash
export RETINEX_TAPETUM_CHECKPOINT="/absolute/path/to/checkpoint.pth"
python test.py
```

Enhanced images are written to the configured dataset-specific `results/Test/` directory.

## Evaluate

```bash
python evaluate_metrics.py
```

The evaluator matches generated outputs and normal-light references by filename and reports per-image and mean PSNR, SSIM, and LPIPS. Results are also written to `metrics.csv`.

## Reproducibility scope

The GitHub directory contains the runnable implementation. Google Drive stores large checkpoints, generated images, diagnostic outputs, notebooks, and experiment archives that should not be committed to a normal Git repository.
