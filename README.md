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

## Google Drive experiment archive

Large experiment artifacts and auxiliary files that are not suitable for direct storage in GitHub are available in the project Google Drive archive:

**[TAPETUM Google Drive experiment archive](https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT?usp=drive_link)**

The archive contains the broader experimental workspace, including:

- RetinexTapetum source snapshots and Colab workflows
- dataset-specific checkpoints and generated outputs
- LOL-v1, LOL-v2, and UHD-LL down4 experiment folders
- baseline method folders for RetinexFormer, URetinex-Net++, KinD++, RetinexNet, RUAS, Zero-DCE, and LIME
- speed and complexity measurements under `SpeedMetrics`
- hyperparameter-search outputs under `HyperparameterSearch`
- manuscript figures, qualitative comparisons, and supporting paper materials under `paper_retinextapetum`
- mobile-oriented experiments under `RetinexTapetumMobile`

The GitHub repository is the canonical location for version-controlled source code. Google Drive is used for large checkpoints, datasets, experiment outputs, notebooks, and supporting artifacts. Availability of third-party datasets and baseline files remains subject to their original licenses and distribution terms.

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

The paper evaluates LOL-v1, LOL-v2 Real-Captured, LOL-v2 Synthetic, and UHD-LL down4. Datasets and third-party checkpoints are not redistributed through GitHub; use the linked archive only where redistribution is permitted and comply with the original licenses.

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
- Large datasets, generated result images, model checkpoints, and complete experiment archives are hosted in the linked Google Drive folder rather than committed to GitHub.

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

Experiment archive: `https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT?usp=drive_link`

## License

No software license has yet been declared. Until a license file is added, copyright is retained by the authors and reuse is not automatically granted.
