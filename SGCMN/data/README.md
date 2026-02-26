# Data Preparation

## Dataset 1: BraTS 2024 (Brain Tumor Segmentation)

- **Source**: [BraTS 2024 Challenge](https://www.synapse.org/brats2024)
- **Modalities**: T2w, FLAIR (T2f), T1ce (missing modality target), T1n
- **Task**: Whole tumor binary segmentation with T1ce as the missing modality
- **Split**: Fixed hold-out 7:1:2 (train:val:test, seed=2026)

### Directory Structure
```
BraTS2024/train/
├── BraTS-GLI-XXXXX-XXX/
│   ├── BraTS-GLI-XXXXX-XXX-t2w.nii
│   ├── BraTS-GLI-XXXXX-XXX-t2f.nii
│   ├── BraTS-GLI-XXXXX-XXX-t1c.nii
│   ├── BraTS-GLI-XXXXX-XXX-t1n.nii
│   └── BraTS-GLI-XXXXX-XXX-seg.nii
└── ...
```

### Modality Mapping
| MRI Modality | Model Input | Role |
|:-------------|:------------|:-----|
| T2w          | T1          | Available modality 1 |
| FLAIR (T2f)  | T2          | Available modality 2 |
| T1ce (T1c)   | Mensa       | Missing modality (dropped at test time) |

## Dataset 2: Parotid Gland Segmentation (Private)

- **Modalities**: 3 MRI sequences (`_0000`, `_0001`, `_0002`)
- **Task**: Parotid gland binary segmentation with `_0000` as the missing modality
- **Split**: 5-fold cross-validation with 11 held-out test cases
- **Ethics**: Approved by the institutional ethics committee (anonymized for review)

### Directory Structure
```
imagesTr/
├── case_XXXXX_0000.nii.gz   # Modality 0 (missing target)
├── case_XXXXX_0001.nii.gz   # Modality 1 (T1)
├── case_XXXXX_0002.nii.gz   # Modality 2 (T2)
└── ...

labelsTr/
├── case_XXXXX.nii.gz
└── ...
```

## Splits File Format

### Single Split (BraTS)
```json
{
  "train": ["case_id_1", "case_id_2", ...],
  "val": ["case_id_3", ...],
  "test": ["case_id_4", ...]
}
```

### 5-Fold Cross-Validation (Parotid)
```json
[
  {"train": ["case_00001", ...], "val": ["case_00010", ...]},
  {"train": ["case_00002", ...], "val": ["case_00020", ...]},
  ...
]
```

## Preprocessing

All modalities are preprocessed identically during training and evaluation:
1. **Resize** to target shape using trilinear interpolation (labels: nearest neighbor)
2. **Z-score normalization** on non-zero voxels:
   ```
   mean = volume[mask].mean()
   std  = volume[mask].std()
   volume -= mean
   volume /= (std + 1e-8)
   ```
3. **Mensa dropout**: during training, the missing modality is randomly zeroed out
   with probability `p = 0.3` to simulate missing modality at test time.
