# SG-CMN: Structural-Guided Causal Modulation Network

Official implementation for multi-modal medical image segmentation with missing modality handling.

## Overview

SG-CMN addresses the clinical challenge of missing MRI modalities during inference by learning to predict the absent modality's features from available ones using a structural causal model. The architecture consists of:

1. **Three Independent Encoders**: extract multi-scale features from each modality independently.
2. **Causal Predictor**: predicts missing modality bottleneck features from available modalities (T1 + T2 → Missing).
3. **Edge Expert**: a pretrained lightweight network that extracts boundary information (τ) to guide the decoder.
4. **SE-Attention Fusion**: fuses three modality features with channel attention.
5. **Edge-Guided Decoder**: uses boundary attention for refined segmentation.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (GPU with >= 8 GB VRAM recommended)

```bash
pip install -r requirements.txt
```

## Project Structure

```
SGCMN/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── models/
│   └── sgcmn/
│       ├── __init__.py
│       ├── network.py             # Main SG-CMN model
│       ├── encoder.py             # Multi-scale 3D encoder
│       ├── decoder.py             # Edge-guided decoder
│       ├── experts.py             # Edge Expert + Causal Predictor
│       ├── modules.py             # Basic building blocks (SE, Conv, Residual)
│       └── losses.py              # CombinedLoss, StyleLoss, QualitySentinel
├── train.py                       # Training script (BraTS & Parotid)
├── evaluate.py                    # Evaluation script (5 metrics)
├── pretrain_edge_expert.py        # Edge Expert pretraining
├── utils.py                       # Metrics, TTA, post-processing utilities
├── data/
│   └── README.md                  # Data preparation instructions
└── pretrained_models/             # Place pretrained weights here
    └── (edge_expert.pth)
```

## Reproduction Commands

### Step 0: Prepare Data

See `data/README.md` for dataset organization and preprocessing details.

### Step 1: Pretrain Edge Expert

```bash
# BraTS 2024
python3 pretrain_edge_expert.py \
    --dataset brats \
    --data_dir /path/to/BraTS2024/train \
    --splits_file /path/to/splits_brats.json \
    --output_path pretrained_models/edge_expert_brats.pth \
    --epochs 80 --batch_size 2 --lr 5e-4 --patience 20 \
    --gpu 0

# Parotid
python3 pretrain_edge_expert.py \
    --dataset parotid \
    --data_dir /path/to/imagesTr \
    --label_dir /path/to/labelsTr \
    --splits_file /path/to/splits_5fold.json \
    --output_path pretrained_models/edge_expert_parotid.pth \
    --target_shape 64 128 128 \
    --epochs 80 --batch_size 2 --lr 5e-4 --patience 20 \
    --gpu 0
```

### Step 2: Train SG-CMN

```bash
# BraTS 2024 (single split)
python3 train.py \
    --dataset brats \
    --data_dir /path/to/BraTS2024/train \
    --splits_file /path/to/splits_brats.json \
    --edge_expert_ckpt pretrained_models/edge_expert_brats.pth \
    --output_dir results/brats \
    --epochs 50 --batch_size 1 --lr 1e-4 --patience 15 \
    --mensa_dropout 0.3 \
    --lambda_causal 0.5 --lambda_style 0.1 --lambda_sentinel 0.5 \
    --num_folds 1 \
    --gpu 0

# Parotid (5-fold cross-validation)
python3 train.py \
    --dataset parotid \
    --data_dir /path/to/imagesTr \
    --label_dir /path/to/labelsTr \
    --splits_file /path/to/splits_5fold.json \
    --edge_expert_ckpt pretrained_models/edge_expert_parotid.pth \
    --output_dir results/parotid_5fold \
    --epochs 60 --batch_size 2 --lr 1e-4 --patience 20 \
    --mensa_dropout 0.3 --freeze_edge_expert \
    --lambda_causal 0.5 --lambda_style 0.1 --lambda_sentinel 0.5 \
    --num_folds 5 \
    --gpu 0
```

### Step 3: Evaluate

```bash
# BraTS 2024
python3 evaluate.py \
    --dataset brats \
    --data_dir /path/to/BraTS2024/train \
    --splits_file /path/to/splits_brats.json \
    --checkpoint results/brats/best_model.pth \
    --edge_expert_ckpt pretrained_models/edge_expert_brats.pth \
    --output_dir results/brats_eval \
    --tta 4flip \
    --gpu 0

# Parotid (5-fold ensemble)
python3 evaluate.py \
    --dataset parotid \
    --data_dir /path/to/imagesTr \
    --label_dir /path/to/labelsTr \
    --splits_file /path/to/splits_5fold.json \
    --checkpoint_dir results/parotid_5fold \
    --edge_expert_ckpt pretrained_models/edge_expert_parotid.pth \
    --output_dir results/parotid_eval \
    --tta 4flip --num_folds 5 \
    --gpu 0
```

## Hyperparameters

| Parameter         | BraTS 2024  | Parotid     |
|:------------------|:-----------:|:-----------:|
| Target shape      | 96×96×96    | 64×128×128  |
| Base channels     | 64          | 64          |
| Batch size        | 1           | 2           |
| Learning rate     | 1e-4        | 1e-4        |
| Optimizer         | AdamW       | AdamW       |
| Weight decay      | 1e-5        | 1e-5        |
| Scheduler         | CosineAnnealing | CosineAnnealing |
| Epochs            | 50          | 60          |
| Early stopping    | 15          | 20          |
| Mensa dropout     | 0.3         | 0.3         |
| λ_causal          | 0.5         | 0.5         |
| λ_style           | 0.1         | 0.1         |
| λ_sentinel        | 0.5         | 0.5         |
| Edge Expert       | Unfrozen    | Frozen      |
| TTA (evaluation)  | 4-flip      | 4-flip      |
| Post-processing   | LCC         | LCC         |

## Software & Hardware

- **Framework**: PyTorch 2.0+
- **GPU**: NVIDIA A100 / RTX 3090 (8+ GB VRAM)
- **Training time**: ~2h per fold (Parotid), ~3h (BraTS) on single GPU
- **Inference time**: ~1.5s per case (with 4-flip TTA)

## License

This code is provided for academic research purposes only.
