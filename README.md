# RetinexTapetum

Official implementation and reproducibility materials for **RetinexTapetum: A Bio-Inspired Darkness-Aware Retinex Framework for Low-Light Image Enhancement**.

RetinexTapetum is a compact Retinex-based low-light image enhancement framework with darkness-aware spatial amplification, tapetum-guided modulation, and bounded residual refinement.

## Repository contents

- `src/model.py` — RetinexTapetum architecture
- `src/losses.py` — training objectives
- `src/dataset.py` — paired low-/normal-light dataset loader
- `src/train.py` — training and checkpoint selection
- `src/test.py` — inference on test images
- `src/evaluate_metrics.py` — PSNR, SSIM, and LPIPS evaluation
- `src/config.py` — dataset paths and hyperparameters
- `src/utils.py` — reproducibility and metric helpers
- `diagnostics/RetinexTapetum_Ablation_Diagnostics_Dataset_Aware.py` — checkpoint-based inference-time interventions
- `paper/` — manuscript LaTeX sources included with the repository snapshot

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a PyTorch build appropriate for the local CUDA version when GPU acceleration is required.

## Dataset layout

The code expects paired low-light and normal-light images matched by filename. Configure paths in `src/config.py`.

Typical structure:

```text
dataset_root/
├── Train/
│   ├── Low/
│   └── Normal/
└── Test/
    ├── Low/
    └── Normal/
```

The paper evaluates LOL-v1, LOL-v2 Real-Captured, LOL-v2 Synthetic, and UHD-LL down4. Datasets and third-party checkpoints are not redistributed here; obtain them from their original sources and comply with their licenses.

## Training

```bash
cd src
python train.py
```

Edit `config.py` before execution. The manuscript configuration uses dataset-specific training from scratch, a maximum of 120 epochs, deterministic seed 42, 256 x 256 aligned crops, and validation-based checkpoint selection.

## Testing and metrics

```bash
cd src
python test.py
python evaluate_metrics.py
```

Generated outputs, checkpoints, datasets, caches, and local Drive mirrors are intentionally excluded from version control.

## Diagnostic interventions

The script in `diagnostics/` suppresses selected pathways only at inference time using an already trained checkpoint. These experiments are diagnostic interventions, not retrained architectural ablations.

## Reproducibility notes

- Update all local/Google Drive paths in `src/config.py`.
- Use the checkpoint corresponding to each dataset.
- Keep RGB inputs normalized to `[0, 1]`.
- The common speed protocol in the manuscript uses an FP32 input of `1 x 3 x 256 x 256`, batch size 1, 50 warm-up passes, and 500 timed passes.
- Large datasets, generated result images, and model checkpoints are not committed to this repository.

## Citation

```bibtex
@article{delen2026retinextapetum,
  title   = {RetinexTapetum: A Bio-Inspired Darkness-Aware Retinex Framework for Low-Light Image Enhancement},
  author  = {Delen, Murat and Ciftci, Serdar},
  year    = {2026},
  note    = {Manuscript}
}
```

Repository: `https://github.com/muratdelen/TapetumFramework`

## License

No software license has yet been declared. Until a license file is added, copyright is retained by the authors and reuse is not automatically granted.
