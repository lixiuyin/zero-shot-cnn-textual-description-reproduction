# Reproduction scripts (Ba et al. ICCV 2015)

One script per table or figure. Run from the **`Code`** directory.

**Note:** Train models first using `scripts/train.py` before running reproduction scripts.

## Checkpoint File Naming

All checkpoints are auto-saved to `checkpoints/` with descriptive names (no timestamp):

**Format:** `{model_type}_{loss}_{dataset}_{layer}_{n_unseen}[_tr{ratio}].pt`

**Examples:**
- `fc_bce_cub_fc_40.pt` - FC model, BCE loss, CUB, 40 unseen classes
- `conv_bce_cub_conv5_3_40.pt` - Conv model, BCE loss, CUB, conv5_3 layer, 40 unseen
- `fc_conv_bce_cub_conv5_3_40.pt` - FC+Conv joint model, BCE loss, CUB, 40 unseen
- `fc_bce_cub_fc_0_tr0.5.pt` - FC model, 50/50 split (n_unseen=0, train_ratio=0.5)

## Checkpoint to Table Mapping

| Table | Model Type | Checkpoint File | Notes |
|-------|-----------|-----------------|-------|
| **Table 1 - CUB** | fc | `fc_bce_cub_fc_40.pt` | 40 unseen |
| | conv | `conv_bce_cub_conv5_3_40.pt` | conv5_3 layer |
| | fc+conv | `fc_conv_bce_cub_conv5_3_40.pt` | Joint model |
| **Table 1 - Flowers** | fc | `fc_bce_flowers_fc_20.pt` | 20 unseen |
| | conv | `conv_bce_flowers_conv5_3_20.pt` | conv5_3 layer |
| | fc+conv | `fc_conv_bce_flowers_conv5_3_20.pt` | Joint model |
| **Table 2** | fc (BCE) | `fc_bce_cub_fc_40.pt` | Reuses Table 1 |
| | fc (Hinge) | `fc_hinge_cub_fc_40.pt` | Hinge loss |
| | fc (Euclidean) | `fc_euclidean_cub_fc_40.pt` | Euclidean loss |
| **Table 3** | fc+conv (conv4_3) | `fc_conv_bce_cub_conv4_3_40.pt` | conv4_3 layer |
| | fc+conv (conv5_3) | `fc_conv_bce_cub_conv5_3_40.pt` | Reuses Table 1 |
| | fc+conv (pool5) | `fc_conv_bce_cub_pool5_40.pt` | pool5 layer |
| **Table 4** | fc (50/50) | `fc_bce_cub_fc_0_tr0.5.pt` | 50/50 split |
| | fc+conv (50/50) | `fc_conv_bce_cub_conv5_3_0_tr0.5.pt` | 50/50 split |
| **Figure 2** | fc | `fc_bce_cub_fc_40.pt` | Reuses Table 1 |

## Quick Start

**Recommended:** Use explicit checkpoint paths for reliable results.

```bash
# Step 1: Train models
python scripts/train.py --model_type fc --dataset cub --epochs 200
python scripts/train.py --model_type conv --dataset cub --epochs 200
python scripts/train.py --model_type fc+conv --dataset cub --epochs 200
python scripts/train.py --model_type fc --dataset cub --loss hinge --epochs 200
python scripts/train.py --model_type fc --dataset cub --loss euclidean --epochs 200
python scripts/train.py --model_type fc+conv --dataset cub --conv_feature_layer conv4_3 --epochs 200
python scripts/train.py --model_type fc+conv --dataset cub --conv_feature_layer pool5 --epochs 200
python scripts/train.py --model_type fc --dataset cub --n_unseen 0 --train_ratio 0.5 --epochs 200
python scripts/train.py --model_type fc+conv --dataset cub --n_unseen 0 --train_ratio 0.5 --epochs 200

# Step 2: Reproduce results with explicit checkpoint paths
python scripts/reproduce/table1.py --cub_root data/images/birds --checkpoint_fc checkpoints/fc_bce_cub_fc_40.pt --checkpoint_conv checkpoints/conv_bce_cub_conv5_3_40.pt --checkpoint_fc_conv checkpoints/fc_conv_bce_cub_conv5_3_40.pt --out_dir results
python scripts/reproduce/table2.py --cub_root data/images/birds --checkpoint_bce checkpoints/fc_bce_cub_fc_40.pt --checkpoint_hinge checkpoints/fc_hinge_cub_fc_40.pt --checkpoint_euclidean checkpoints/fc_euclidean_cub_fc_40.pt --out_dir results
python scripts/reproduce/table3.py --cub_root data/images/birds --checkpoint_conv4_3 checkpoints/fc_conv_bce_cub_conv4_3_40.pt --checkpoint_conv5_3 checkpoints/fc_conv_bce_cub_conv5_3_40.pt --checkpoint_pool5 checkpoints/fc_conv_bce_cub_pool5_40.pt --out_dir results
python scripts/reproduce/table4.py --cub_root data/images/birds --checkpoint_fc checkpoints/fc_bce_cub_fc_0_tr0.5.pt --checkpoint_fc_conv checkpoints/fc_conv_bce_cub_conv5_3_0_tr0.5.pt --out_dir results
python scripts/reproduce/figure2.py --cub_root data/images/birds --checkpoint_fc checkpoints/fc_bce_cub_fc_40.pt --out_dir results
```

**Alternative:** Use `--checkpoint_dir checkpoints` for auto-detection (may have ambiguity with multiple matching files).

## Incremental Updates

**Table 1 and Table 4** support incremental updates - you can run them multiple times with different datasets, and they will preserve existing results:

```bash
# First run: CUB dataset only
python scripts/reproduce/table1.py --cub_root data/images/birds --checkpoint_fc checkpoints/fc_bce_cub_fc_40.pt

# Second run: Flowers dataset (preserves CUB results)
python scripts/reproduce/table1.py --flowers_root data/images/flowers --checkpoint_fc checkpoints/fc_bce_flowers_fc_20.pt

# Re-run: Update CUB results (preserves Flowers results)
python scripts/reproduce/table1.py --cub_root data/images/birds --checkpoint_fc checkpoints/fc_bce_cub_fc_40.pt
```

The script automatically:
- Reads existing `TableN.csv` if it exists
- Preserves "Ours" results for datasets **not** being evaluated
- Updates "Ours" results for the current dataset
- Never overwrites Paper values

## Scripts Overview

| Script      | Output              | Paper caption |
|------------|---------------------|---------------|
| `table1.py`  | `results/tables/Table1.csv`  | Table 1. ROC-AUC and PR-AUC(AP) performance compared to other methods. |
| `table2.py`  | `results/tables/Table2.csv`  | Table 2. Model performance using various objective functions on CUB-200-2010 dataset. |
| `table3.py`  | `results/tables/Table3.csv`  | Table 3. Performance comparison using different intermediate ConvLayers from VGG net on CUB-200-2010 dataset. |
| `table4.py`  | `results/tables/Table4.csv`  | Table 4. Performance of our model trained on the full dataset, a 50/50 split is used for each class. |
| `figure2.py` | `results/figures/Figure2.png` | Figure 2. [LEFT] Word sensitivities of unseen classes (fc model, CUB200-2010). [RIGHT] Text features describing visual features. |
| `compile_all_tables.py` | `results/AllTables.pdf` | Compiles all `TableN.tex` files to a single PDF with **xelatex**. Run after generating the tables. |

Shared helpers: `common.py`, `eval_utils.py`.

## Data splits (paper-aligned)

- **CUB zero-shot (Table 1–3, Figure 2)**
  `prepare_birds_zero_shot`: 40 unseen / 160 seen classes; within 160 seen, **80% train / 20% test** (`train_ratio=0.8`). Seeds: `unseen_seed=42`, `split_seed=42` (see `data/dataset.py`).
- **CUB full-dataset 50/50 (Table 4)**
  `prepare_birds_50_50`: all 200 classes, **50% train / 50% test per class** (seed 42).
- **Oxford Flowers zero-shot (Table 1)**
  `prepare_flowers_zero_shot`: **82 seen / 20 unseen** (last 20 in sorted order). You provide `--flowers_root` with all class directories.
- **Oxford Flowers 50/50 (Table 4)**
  `--flowers_root` (single root with 102 class dirs) → `prepare_flowers_50_50`.

## Using checkpoint_dir

Use `--checkpoint_dir` to **auto-detect checkpoints by pattern matching**:

```bash
# Train models (checkpoints auto-saved with detailed names)
# Then run reproduction scripts with auto-detection:
python scripts/reproduce/table1.py --cub_root data/images/birds --checkpoint_dir checkpoints
```

**Auto-detection logic** (from `common.py`):
1. Scans checkpoint_dir for files matching the new detailed naming format
2. Returns the most recently modified matching file
3. Falls back to old fixed names (`fc.pt`, `conv.pt`) for backward compatibility
4. Supports both CUB and Flowers datasets
5. Correctly distinguishes between `fc`, `conv`, and `fc_conv` models

**Pattern examples**:
- `"fc"` key → matches `fc_*_cub_fc_*_*.pt` or `fc_*_flowers_fc_*_*.pt`
- `"conv"` key → matches `conv_*_cub_conv5_3_*.pt` or `conv_*_flowers_pool5_*.pt`
- `"fc_conv"` key → matches `fc_conv_*_cub_conv*_*.pt` (joint models only, excludes `fc` or `conv`)
- `"bce"` key → matches `fc_bce_cub_*_*.pt` or `fc_bce_flowers_*_*.pt`

**Alternative**: Specify checkpoints manually with individual arguments (overrides auto-detection).

## Table / figure format

Tables organize results with **Paper and Ours data side-by-side** for easy comparison:
- **Table 1**: 14 columns - ROC-AUC and PR-AUC (unseen, seen, mean) for each metric, with Paper and Ours adjacent
- **Table 2**: 6 columns - Each loss function (BCE, Hinge, Euclidean) has Paper and Ours columns
- **Table 3**: 6 columns - Each conv layer (Conv5_3, Conv4_3, Pool5) has Paper and Ours columns
- **Table 4**: 4 columns - Each dataset (CUB, Oxford Flowers) has Paper and Ours columns
- **Figure 2**: Left = word sensitivities (fc, CUB200-2010), right = nearest-neighbor table

All numbers are produced by **evaluating the loaded checkpoints** (no hardcoded values).
