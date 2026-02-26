# SG-CMN: Structural-Guided Causal Modulation Network

Official implementation for multi-modal medical image segmentation with missing modality handling.

## Overview

SG-CMN addresses the clinical challenge of missing MRI modalities during inference by learning to predict the absent modality's features from available ones using a structural causal model. The architecture consists of:

1. **Three Independent Encoders**: extract multi-scale features from each modality independently.
2. **Causal Predictor**: predicts missing modality bottleneck features from available modalities (T1 + T2 → Missing).
3. **Edge Expert**: a pretrained lightweight network that extracts boundary information (τ) to guide the decoder.
4. **SE-Attention Fusion**: fuses three modality features with channel attention.
5. **Edge-Guided Decoder**: uses boundary attention for refined segmentation.
6. **Quality Sentinel**: predicts segmentation quality (Dice) for failure detection.

## Results

### Dataset 1: Parotid Gland Segmentation (5-fold CV, 11 test cases)

| Method       | Dice (%)       | HD95 (mm)     | ASSD (mm)     | Sensitivity | Precision |
|:-------------|:--------------:|:-------------:|:-------------:|:-----------:|:---------:|
| HVED         | 62.49 ± 13.65  | 8.55 ± 9.34   | 4.63 ± 5.61   | 0.733       | 0.675     |
| HeMIS        | 63.54 ± 20.20  | 7.83 ± 9.83   | 4.98 ± 8.08   | 0.653       | 0.691     |
| RFNet        | 69.81 ± 11.28  | 4.77 ± 5.52   | 2.19 ± 2.80   | 0.886       | 0.628     |
| CKMD         | 73.11 ± 11.45  | 4.27 ± 6.92   | 3.09 ± 6.22   | 0.951       | 0.607     |
| DC-Seg       | 75.29 ± 27.64  | 5.50 ± 10.48  | 3.67 ± 8.25   | 0.788       | 0.771     |
| MGD-KD       | 83.33 ± 4.82   | 1.85 ± 0.76   | 0.77 ± 0.25   | **0.962**   | 0.744     |
| **SG-CMN (Ours)** | **90.52 ± 4.24** | **1.39 ± 0.78** | **0.27 ± 0.14** | 0.927 | **0.887** |

### Dataset 2: BraTS 2024 (Fixed split, 32 test cases, missing T1ce)

| Method       | Dice (%)       | HD95 (mm)      | ASSD (mm)      | Sensitivity | Precision |
|:-------------|:--------------:|:--------------:|:--------------:|:-----------:|:---------:|
| HeMIS        | 84.62 ± 12.77  | 9.66 ± 12.69   | 2.03 ± 3.05    | 0.867       | 0.862     |
| HVED         | 62.13 ± 17.61  | 23.11 ± 17.84  | 7.63 ± 7.65    | 0.565       | 0.829     |
| RFNet        | 70.25 ± 15.74  | 17.58 ± 16.47  | 4.82 ± 5.50    | 0.704       | 0.812     |
| M2FTrans     | 61.97 ± 18.59  | 22.41 ± 16.95  | 8.60 ± 8.55    | 0.569       | 0.857     |
| MMCFormer    | 70.89 ± 17.98  | 15.99 ± 16.08  | 5.20 ± 6.76    | 0.741       | 0.769     |
| MGD-KD       | 85.01 ± 11.14  | 8.70 ± 11.71   | 2.08 ± 2.87    | 0.873       | 0.866     |
| CKMD         | 79.75 ± 14.88  | 11.15 ± 12.75  | 3.31 ± 5.08    | 0.840       | 0.808     |
| DCSeg        | 86.40 ± 10.27  | 7.59 ± 11.36   | 1.67 ± 2.68    | 0.873       | 0.892     |
| **SG-CMN (Ours)** | **89.45 ± 5.63** | **5.01 ± 5.78** | **1.10 ± 1.25** | **0.898** | **0.912** |

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

## Model Parameters

| Component           | Parameters |
|:--------------------|:----------:|
| Encoder (×3)        | 28.8M      |
| Edge Expert         | 0.02M      |
| Causal Predictor    | 5.0M       |
| Fusion + SE         | 0.8M       |
| Decoder             | 8.9M       |
| Quality Sentinel    | 0.1M       |
| **Total**           | **43.6M**  |

## Software & Hardware

- **Framework**: PyTorch 2.0+
- **GPU**: NVIDIA A100 / RTX 3090 (8+ GB VRAM)
- **Training time**: ~2h per fold (Parotid), ~3h (BraTS) on single GPU
- **Inference time**: ~1.5s per case (with 4-flip TTA)

## License

This code is provided for academic research purposes only.
