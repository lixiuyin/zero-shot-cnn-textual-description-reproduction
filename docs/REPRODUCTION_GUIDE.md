# Reproduction Guide — Ba et al. ICCV 2015

This guide provides detailed instructions for reproducing all paper experiments and generating comparison tables. For a high-level overview and quick start, see the [README](../README.md).

## Paper Experiments Overview

| Table/Figure | Paper Section | Paper Dataset | Our Dataset | Model | Key Variable |
|---|---|---|---|---|---|
| Table 1 | Sec 5.4 | CUB-2010, CUB-2011, Flowers | CUB-2011, Flowers | fc, conv, fc+conv | model type |
| Table 2 | Sec 5.5 | **CUB-2010** | CUB-2011 | fc | loss function (BCE / Hinge / Euclidean) |
| Table 3 | Sec 5.6 | **CUB-2010** | CUB-2011 | fc+conv | conv layer (conv4\_3 / conv5\_3 / pool5) |
| Table 4 | Sec 5.7 | CUB-2010, CUB-2011, Flowers | CUB-2011, Flowers | fc, fc+conv | full-dataset 50/50 split |
| Figure 2 | Sec 5.8 | **CUB-2010** | CUB-2011 | fc | word sensitivity + NN retrieval |
| Figure 5 | Appendix | CUB, Flowers | CUB-2011, Flowers | fc+conv (conv) | conv filter visualization |

> **Bold** = our dataset differs from the paper's. CUB-200-2010 (6,033 images) is no longer publicly available; we use CUB-200-2011 (11,788 images, same 200 bird classes).

**Data splits (Paper Sec 5.2, 5.3):**
- CUB: 40 unseen / 160 seen classes; seen classes use 80% train / 20% test; 5-fold cross-validation
- Flowers: 20 unseen / 82 seen classes; seen classes use 80% train / 20% test

**Paper Table 1 note:** "For both ROC-AUC and PR-AUC, we report the best numbers obtained among the models trained on different objective functions." We reproduce only the "Ours (fc/conv/fc+conv)" rows, not the DA/GPR baseline methods from Elhoseiny et al. [5] and Kulis et al. [15].

---

## Recommended: One-command Full Pipeline

```bash
# Step 1: Train all paper models (single-run, ~12 h on one RTX 5090)
bash train.sh

# Step 1 (alt): Train with 5-fold CV (paper default, ~60 h on one GPU)
bash train.sh --n-folds 5

# Step 2: Generate all tables and figures
bash reproduce.sh
```

- `train.sh` trains every required checkpoint. By default it uses `--n-folds 1` (single run, checkpoints in `checkpoints/`). Pass `--n-folds 5` for 5-fold CV (checkpoints in `checkpoints/fold{i}/`).
- `reproduce.sh` automatically detects fold directories and averages CV folds when computing Table 1 metrics. Results are written to `results/`.

For innovation experiments (beyond the paper):

```bash
bash innovate.sh
```

Checkpoints saved under `checkpoints/innov/`.

---

## Environment Setup

```bash
# Using uv (recommended)
uv sync
source .venv/bin/activate

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download datasets
cd data && python download_dataset.py && cd ..
```

---

## Step 1: Training

### Training Loop Details

**File:** `scripts/train.py`

**Epoch loop with early stopping:**
```python
# Early stopping uses three phases:
#   1. Warmup (epoch < min_save_epoch): track best_metric only
#   2. Observation (min_save_epoch <= epoch < min_epochs): save best, don't count patience
#   3. Patience (epoch >= min_epochs): save best AND count patience

min_save_epoch = max(5, args.min_epochs // 10)  # e.g., 5 for min_epochs=50

for epoch in range(1, args.epochs + 1):
    # Training
    model.train()
    for batch in train_loader:
        images = batch["image"]
        labels_seen = seen_label_map_tensor[labels]  # Remap to seen subset indices
        loss = train_step(model, optimizer, criterion, images,
                         text_features_seen, labels_seen, device)

    # Evaluation (seen + unseen)
    seen_top1, seen_top5 = _evaluate(test_seen_loader, text_features_seen, ...)
    unseen_top1, unseen_top5 = _evaluate(test_unseen_loader, text_features_unseen, ...)

    # Early stopping check
    current_metric = unseen_top1 if n_unseen > 0 else seen_top1
    if current_metric > best_metric:
        best_metric = current_metric
        best_epoch = epoch
        if epoch >= min_save_epoch:
            best_model_state = copy.deepcopy(model.state_dict())
        if epoch >= args.min_epochs:
            patience_counter = 0
    elif epoch >= args.min_epochs:
        patience_counter += 1

    if patience_counter >= args.patience:
        break  # Early stopping

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
```

**Loss computation with auxiliary losses:**
```python
# Base loss (BCE/Hinge/Euclidean)
loss = criterion(batch_scores, targets)

# CLIP contrastive loss (fc/fc+conv only)
if use_clip_loss:
    pos_text_emb = text_emb[labels]  # [B, k] ground-truth class per image
    loss = loss + clip_weight * clip_contrastive_loss(image_emb, pos_text_emb,
                                                         temperature=0.07)

# Center alignment loss
if use_alignment:
    loss = loss + align_weight * center_alignment_loss(image_emb, pos_text_emb)

# Embedding MSE loss
if use_embedding_loss:
    loss = loss + embedding_weight * embedding_mse_loss(image_emb, pos_text_emb,
                                                          reduction="sum")
```

**Optimizer learning rate selection:**
```python
# Paper uses lr=1e-4 for all models, but we use higher LR for conv models
effective_lr = args.lr  # 1e-4
if args.model_type in ("conv", "fc+conv"):
    effective_lr = LR_CONV  # 5e-4 (empirical: improves convergence)

optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `fc` | `fc`, `conv`, or `fc+conv` |
| `--dataset` | `cub` | `cub` or `flowers` |
| `--epochs` | `50` | Maximum training epochs |
| `--loss` | `bce` | `bce`, `hinge`, or `euclidean` |
| `--n_unseen` | 40 (CUB) / 20 (Flowers) | Unseen classes (0 = full dataset) |
| `--train_ratio` | `0.8` | Train split for seen classes |
| `--conv_feature_layer` | `conv5_3` | `conv4_3`, `conv5_3`, or `pool5` (VGG-19 only) |
| `--batch_size` | `200` | Batch size (paper default) |
| `--lr` | `1e-4` | Learning rate for fc (conv/fc+conv always use 5e-4) |
| `--k` | `50` | Joint embedding dimension |
| `--ft_hidden` | `300` | Text encoder hidden dim |
| `--gv_hidden` | `300` | Image fc-branch hidden dim |
| `--conv_channels` | `5` | K' predicted conv filters |
| `--text_encoder` | `tfidf` | `tfidf`, `sbert`, `sbert_multi`, `clip`, or `clip_multi` |
| `--text_dim` | auto | Text feature dim (auto-detected from encoder when -1) |
| `--image_backbone` | `vgg19` | `vgg19`, `densenet121`, or `resnet50` |
| `--fc_mode` | `default` | `default` or `penultimate` (DenseNet/ResNet only) |
| `--save` | auto | Custom checkpoint path (without .pt suffix) |
| `--log_file` | auto | Custom log path |
| `--no_early_stopping` | — | Disable early stopping (enabled by default) |
| `--patience` | `20` | Early stopping patience |
| `--min_epochs` | `50` | Minimum epochs before early stopping |
| `--seed` | `42` | Base random seed (fold_seed = seed + fold_idx) |
| `--deterministic` | — | Deterministic cuDNN (exact GPU reproducibility) |
| `--n_folds` | `5` | CV folds (>1 = multi-fold into `fold{i}/`; 1 = single run at root) |
| `--standard_sampler` | — | Use RandomSampler (paper method); default is ClassAwareSampler |
| `--data_root` | `data` | Root directory for images/ and wikipedia/ |
| `--wikipedia_jsonl` | auto | Optional explicit path to Wikipedia JSONL |
| `--use_clip_loss` | — | Add auxiliary CLIP contrastive loss (fc/fc+conv only) |
| `--clip_weight` | `0.1` | CLIP loss weight λ |
| `--clip_temperature` | `0.07` | CLIP softmax temperature |
| `--use_center_align` | — | Add center alignment loss (fc/fc+conv only) |
| `--center_align_weight` | `0.1` | Center alignment loss weight |
| `--use_embedding_loss` | — | Add embedding MSE loss (fc/fc+conv only) |
| `--embedding_weight` | `1.0` | Embedding MSE loss weight |

> **Note:** `train.sh` overrides several defaults: `--epochs 200`, `--n-folds 1`. The `train.py` script defaults are `--epochs 50`, `--n_folds 5`.

### Early Stopping (enabled by default)

- **Zero-shot mode** (`n_unseen > 0`): monitors **Unseen Top-1 accuracy**
- **Full-dataset mode** (`n_unseen = 0`): monitors **Test Top-1 accuracy**
- Stops after `--patience` epochs without improvement (default: 20)
- Requires at least `--min_epochs` before stopping (default: 50)
- Saves the **best checkpoint** (not the final epoch)

### Training Output Format

Zero-shot mode (Tables 1–3):
```
Epoch   1/200 | Loss: 46.8649 | Seen:  35.2%/ 62.8% | Unseen:  12.3%/ 28.5% | ETA: 15:32
```

Full-dataset mode (Table 4):
```
Epoch   1/200 | Loss: 45.123 | Test:  12.3%/ 45.6% | ETA: 15:32
```

### Checkpoint Naming Convention

```
{model_type}_{loss}_{dataset}_{layer}_{n_unseen}[_extensions].pt
```

- `fc+conv` is mapped to `fc_conv` (avoids `+` in filenames)
- Extension suffixes (only when non-default):
  - `_te{encoder}` if text_encoder ≠ `tfidf` (e.g. `_tesbert`, `_teclip`)
  - `_bb{backbone}` if image_backbone ≠ `vgg19` (e.g. `_bbdensenet121`)
  - `_fc{mode}` if fc_mode ≠ `default` (e.g. `_fcpenultimate`)
  - `_clip{weight}` if use_clip_loss (e.g. `_clip0.1`)
  - `_tr{ratio}` if train_ratio ≠ 0.8 (e.g. `_tr0.5`)

Examples:
- `fc_bce_cub_fc_40.pt` — FC, BCE, CUB, 40 unseen
- `conv_hinge_cub_conv5_3_40.pt` — Conv, Hinge, CUB, conv5_3, 40 unseen
- `fc_conv_bce_flowers_conv5_3_20.pt` — FC+Conv, BCE, Flowers, conv5_3, 20 unseen
- `fc_bce_cub_fc_0_tr0.5.pt` — FC, BCE, CUB, full dataset, 50/50 split
- `fc_conv_bce_cub_conv5_3_40_tesbert_bbdensenet121.pt` — FC+Conv with SBERT + DenseNet

### Table 1 Models — Cross-validation (Paper Sec 5.2: "5-fold cross-validation is used")

```bash
# CUB (5-fold CV)
python scripts/train.py --model_type fc      --dataset cub     --epochs 200
# -> checkpoints/fold{0-4}/fc_bce_cub_fc_40.pt

python scripts/train.py --model_type conv    --dataset cub     --epochs 200
# -> checkpoints/fold{0-4}/conv_bce_cub_conv5_3_40.pt

python scripts/train.py --model_type fc+conv --dataset cub     --epochs 200
# -> checkpoints/fold{0-4}/fc_conv_bce_cub_conv5_3_40.pt

# Flowers (5-fold CV)
python scripts/train.py --model_type fc      --dataset flowers --epochs 200
# -> checkpoints/fold{0-4}/fc_bce_flowers_fc_20.pt

python scripts/train.py --model_type conv    --dataset flowers --epochs 200
# -> checkpoints/fold{0-4}/conv_bce_flowers_conv5_3_20.pt

python scripts/train.py --model_type fc+conv --dataset flowers --epochs 200
# -> checkpoints/fold{0-4}/fc_conv_bce_flowers_conv5_3_20.pt
```

> **Note:** `train.sh` also trains hinge and euclidean variants for all model types on both datasets (single runs) for Table 1's "best among loss functions" comparison. The conv-only model does not support euclidean loss.

### Table 2 Models — Loss function ablation (Paper Sec 5.5)

BCE reuses Table 1 fold checkpoints (auto-detected). Only hinge and euclidean need explicit training:

```bash
python scripts/train.py --model_type fc --dataset cub --loss hinge     --epochs 200 --n_folds 1
# -> checkpoints/fc_hinge_cub_fc_40.pt

python scripts/train.py --model_type fc --dataset cub --loss euclidean --epochs 200 --n_folds 1
# -> checkpoints/fc_euclidean_cub_fc_40.pt
```

### Table 3 Models — Conv layer ablation (Paper Sec 5.6)

conv5_3 reuses Table 1 fold checkpoints. Only conv4_3 and pool5 need explicit training:

```bash
python scripts/train.py --model_type fc+conv --dataset cub --conv_feature_layer conv4_3 --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_cub_conv4_3_40.pt

python scripts/train.py --model_type fc+conv --dataset cub --conv_feature_layer pool5   --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_cub_pool5_40.pt
```

### Table 4 Models — Full dataset supervised (Paper Sec 5.7, 50/50 split)

```bash
python scripts/train.py --model_type fc      --dataset cub     --n_unseen 0 --train_ratio 0.5 --epochs 200 --n_folds 1
# -> checkpoints/fc_bce_cub_fc_0_tr0.5.pt

python scripts/train.py --model_type fc+conv --dataset cub     --n_unseen 0 --train_ratio 0.5 --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_cub_conv5_3_0_tr0.5.pt

python scripts/train.py --model_type fc      --dataset flowers --n_unseen 0 --train_ratio 0.5 --epochs 400 --n_folds 1
# -> checkpoints/fc_bce_flowers_fc_0_tr0.5.pt

python scripts/train.py --model_type fc+conv --dataset flowers --n_unseen 0 --train_ratio 0.5 --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_flowers_conv5_3_0_tr0.5.pt
```

> **Note:** Flowers FC model uses 400 epochs instead of 200 for full-dataset convergence.

---

## Step 2: Generating Results

All reproduction scripts use **checkpoint auto-detection** by default — they search `checkpoints/` (and `fold{i}/` subdirectories) using pattern matching. You can also pass explicit checkpoint paths via `--checkpoint_*` arguments.

### Table 1 — Model Type Comparison (Paper Sec 5.4)

```bash
python scripts/reproduce/table1.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir results
```

The script auto-detects fc, conv, fc+conv checkpoints for both CUB and Flowers. It evaluates all model types and selects the best loss function variant for each (ROC-AUC and PR-AUC).

### Table 2 — Loss Function Comparison (Paper Sec 5.5)

```bash
python scripts/reproduce/table2.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir results
```

Optional explicit overrides: `--checkpoint_bce`, `--checkpoint_hinge`, `--checkpoint_euclidean`.

### Table 3 — Conv Layer Ablation (Paper Sec 5.6)

```bash
python scripts/reproduce/table3.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir results
```

Optional explicit overrides: `--checkpoint_conv4`, `--checkpoint_conv5`, `--checkpoint_pool5`.

### Table 4 — Supervised 50/50 Baseline (Paper Sec 5.7)

```bash
python scripts/reproduce/table4.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir results
```

Optional explicit overrides: `--checkpoint_fc`, `--checkpoint_fc_conv`.

### Figure 2 — Word Sensitivity + Nearest-Neighbour Retrieval (Paper Sec 5.8)

```bash
python scripts/reproduce/figure2.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir results
```

Optional explicit override: `--checkpoint_fc`. Additional options: `--n_unseen_show` (default 3), `--classes` (override class selection), `--max_words_ablate` (0 = all non-zero TF-IDF dims).

### Figure 5 — Conv Filter Visualization (Appendix)

```bash
python scripts/reproduce/figure5.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir results
```

Optional explicit override: `--checkpoint_conv`. Additional options: `--n_unseen_show` (default 3), `--top_k` (default 5), `--classes`.

### Innovation Summary Table

```bash
python scripts/reproduce/table_innov.py \
    --cub_root data/images/birds \
    --innov_dir checkpoints/innov \
    --out_dir results
```

### Shared Evaluation Script Arguments

| Argument | Description |
|----------|-------------|
| `--cub_root` | CUB image root directory |
| `--flowers_root` | Oxford Flowers-102 image root directory |
| `--wikipedia_birds` | CUB Wikipedia text path (default: `data/wikipedia/birds.jsonl`) |
| `--wikipedia_flowers` | Flowers Wikipedia text path (default: `data/wikipedia/flowers.jsonl`) |
| `--checkpoint_dir` | Checkpoint directory for auto-detection (default: `checkpoints/`) |
| `--n_folds` | CV fold count (0 = auto-detect from `fold*/` dirs) |
| `--out_dir` | Output directory (default: `results/`) |
| `--device` | `cuda` or `cpu` |
| `--batch_size` | Evaluation batch size (default: 64) |
| `--seed` | Random seed for data splits (default: 42) |

### Output Files

| Type | Location | Use |
|------|----------|-----|
| CSV | `results/tables/Table*.csv` | Spreadsheet viewing |
| LaTeX | `results/tex/Table*.tex` | Paper tables |
| PDF | `results/Table*.pdf` | Per-table compiled output |
| Combined PDF | `results/AllTables.pdf` | All tables combined |
| Figures | `results/figures/Figure*.png` | Paper figures |

---

## Checkpoint Auto-detection

When no explicit `--checkpoint_*` path is provided, scripts search `checkpoint_dir` using named patterns defined in `scripts/reproduce/common.py`.

**Resolution priority:**
1. Explicit path (if provided and exists)
2. Pattern match in `checkpoint_dir` using the `CHECKPOINT_PATTERNS` table

**Pattern table (selected examples):**

| Key | Pattern |
|-----|---------|
| `fc_bce_cub` | `fc_bce_cub_fc_*.pt` |
| `conv_bce_cub` | `conv_bce_cub_conv5_3_*.pt` |
| `fc_conv_bce_cub` | `fc_conv_bce_cub_conv5_3_*.pt` |
| `fc_conv_bce_cub_conv4_3` | `fc_conv_bce_cub_conv4_3_*.pt` |
| `fc_bce_cub_5050` | `fc_bce_cub_fc_0_tr0.5.pt` |

**CV fold aggregation:**
When a key matches files in multiple `fold{i}/` directories, `resolve_cv_checkpoints` collects all folds and the evaluation script averages results across folds.

**Dataset-specific keys** prevent cross-dataset contamination:
- `fc_bce_cub` → `fold{i}/fc_bce_cub_fc_*.pt` (CUB only)
- `fc_bce_flowers` → `fold{i}/fc_bce_flowers_fc_*.pt` (Flowers only)
- `fc_conv_bce_cub_conv4_3` → `fc_conv_bce_cub_conv4_3_*.pt` (Table 3 CUB conv4_3 only)

---

## Implementation Details

### Paper Alignment

| Component | Paper Spec (Section) | Implementation |
|-----------|---------------------|----------------|
| VGG-19 | ImageNet pretrained, frozen (Sec 5.1) | torchvision VGG19, no fine-tuning |
| Image preprocessing | Shortest side → 224px, center crop 224×224 (Sec 5.1) | torchvision transforms |
| ft(·) text encoder | p → 300 → k, k=50 (Sec 3.2, 5.1) | Linear(p, 300) → ReLU → Linear(300, 50) |
| gv(·) image encoder | 4096 → 300 → k (Sec 3.2, 5.1) | Linear(4096, 300) → ReLU → Linear(300, 50) |
| Conv branch g'v(·) | K'=5 filters 3×3, conv5_3 512×14×14 (Sec 3.3, 5.1) | Conv2d(512→5, 3×3) + predicted K'×3×3 |
| Joint model | ŷ = w^T gv(x) + o(conv(w', g'v(a))) (Eq. 5, Sec 3.4) | `_forward_fc() + _forward_conv()` |
| Initialization | Small init (Sec 5.1) | std=0.01 for weight-prediction layers (fc2, ConvWeightPredictor) |
| Loss: BCE | Eq. 6, sum reduction (Sec 4.1) | `F.binary_cross_entropy_with_logits(reduction="sum")` |
| Loss: Hinge | Eq. 7, margin=1 (Sec 4.2) | `F.relu(margin - targets * scores).sum()` |
| Loss: Euclidean | Pairwise distance with margin (Sec 4.2.1) | Positive: minimize ‖g-f‖²; Negative: max(0, margin-‖g-f‖)² |
| Minibatch | Sum only over classes in batch, O(B×U) (Sec 4.1) | `torch.unique` dynamic class selection |
| Optimizer | Adam (Sec 5.1) | Adam(lr=1e-4) fc; Adam(lr=5e-4) conv/fc+conv* |
| Batch size | 200 (Sec 5.1) | Default: 200 |
| TF-IDF | 9763-d, log normalization (Sec 5.2) | `sublinear_tf=True`, `max_features=9763` |
| Pooling | Global average pooling for conv (Sec 3.3) | `out.flatten(2).mean(2)` |
| CV | 5-fold cross-validation (Sec 5.2) | `--n_folds 5` (train.py default) |

*lr=5e-4 for conv/fc+conv is empirical; the paper uses lr=1e-4 for all models.

### Extension Components (beyond paper)

| Component | Details |
|-----------|---------|
| DenseNet-121 | Conv branch: denseblock3 output (1024×14×14); FC: classifier 1000-d (default) or avgpool 1024-d (penultimate) |
| ResNet-50 | Conv branch: layer3 output (1024×14×14); FC: fc 1000-d (default) or avgpool 2048-d (penultimate) |
| SBERT | sentence-transformers/all-MiniLM-L6-v2, 384-d, document-level (SentenceTransformer.encode), L2-normalized |
| SBERT-multi | sentence-transformers/all-MiniLM-L6-v2, 384-d, sentence-level mean pooling (split on `.!?;\n`, truncate at 300 chars per sentence) |
| CLIP text | openai/clip-vit-base-patch32, 512-d, document-level (EOS token pooling via text_model → text_projection → pooler_output), L2-normalized |
| CLIP-multi | openai/clip-vit-base-patch32, 512-d, sentence-level mean pooling (split on `.!?;\n`, batch_size=64 for encoding) |
| CLIP contrastive loss | Symmetric InfoNCE with temperature τ=0.07: (loss_i2t + loss_t2i) / 2 × batch_size |
| Center alignment loss | MSE between mean image and text embeddings: ‖mean(g) - mean(f)‖² |
| Embedding MSE loss | Direct MSE(image_emb, text_emb) with sum reduction: Σ‖g - f‖² |

### Minibatch Loss (Paper Sec. 4, Eq. 6–7)

The model scores all classes, but loss is computed only over classes present in the batch (O(B×U), U ≤ B):

```python
all_scores = model(images, text_features)          # [B, C_all]
unique_classes, inverse = torch.unique(labels, return_inverse=True)
batch_scores = all_scores[:, unique_classes]        # [B, U]
```

### Loss Functions (Complete Code Specification)

**File:** `utils/losses.py`

#### BCE Loss (Paper Eq. 6, Sec 4.1)

```python
def bce_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy loss (Paper Eq. 6).

    Paper Eq. 6:
        L = Σ_ij [ I_ij log σ(yhat_ij) + (1-I_ij) log(1-σ(yhat_ij)) ]

    where σ is the sigmoid function and I_ij ∈ {0,1}.

    Note: Paper uses SUM (not mean) over all i,j in batch.
    """
    return F.binary_cross_entropy_with_logits(scores, targets.float(), reduction="sum")
```

**Key specifications:**
- Targets: Binary {0, 1} for negative/positive classes
- Reduction: `sum` (not mean) — per paper specification
- Sigmoid applied internally via `binary_cross_entropy_with_logits`

#### Hinge Loss (Paper Eq. 7, Sec 4.2)

```python
def hinge_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge loss (Paper Eq. 7).

    Paper Eq. 7:
        L = Σ_ij max(0, margin - I_ij * y_ij)

    where I_ij ∈ {+1,-1} and margin = 1.
    """
    return F.relu(margin - targets * scores).sum()
```

**Key specifications:**
- Targets: Binary {+1, -1} for positive/negative classes
- Margin: Default 1.0 (paper specification)
- Reduction: `sum` over all elements

#### Euclidean Loss (Paper Sec 4.2.1)

```python
def euclidean_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Direct Euclidean distance loss (Paper Sec 4.2.1).

    Contrastive formulation:
        Positive pairs (I=+1): L = ||g_i - f_j||²  (pull together)
        Negative pairs (I=-1): L = max(0, margin - ||g_i - f_j||)²  (push apart)

    Pairwise squared Euclidean distance:
        ||g_i - f_j||² = ||g_i||² - 2*g_i^T*f_j + ||f_j||²
    """
    # Pairwise squared Euclidean distances [B, U]
    dist_sq = (image_emb.pow(2).sum(dim=1, keepdim=True)    # [B, 1]
               - 2 * (image_emb @ text_emb.T)                # [B, U]
               + text_emb.pow(2).sum(dim=1).unsqueeze(0))    # [1, U]
    dist_sq = dist_sq.clamp(min=0.0)

    pos_mask = (targets > 0).float()
    neg_mask = (targets < 0).float()

    # Positive: minimize squared distance
    loss_pos = (dist_sq * pos_mask).sum()

    # Negative: push apart if closer than margin
    dist = dist_sq.sqrt()
    loss_neg = (F.relu(margin - dist).pow(2) * neg_mask).sum()

    return loss_pos + loss_neg
```

**Key specifications:**
- Positive pairs: Minimize squared Euclidean distance ‖g - f‖²
- Negative pairs: Hinge on distance margin — max(0, margin - ‖g - f‖)²
- Avoids parasitic L2 penalty issue (paper's hinge+L2 decomposition pushes embeddings to zero)

#### CLIP Contrastive Loss (Extension)

```python
def clip_contrastive_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """CLIP-style symmetric contrastive loss.

    Pulls together matched (image, text) pairs and pushes apart mismatched
    pairs within the mini-batch using L2-normalized embeddings.

    Returns: (loss_i2t + loss_t2i) / 2 × batch_size
    """
    image_emb = F.normalize(image_emb, dim=1)
    text_emb = F.normalize(text_emb, dim=1)

    logits = image_emb @ text_emb.T / temperature  # [B, B]
    labels = torch.arange(image_emb.size(0), device=image_emb.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2 * image_emb.size(0)
```

**Key specifications:**
- L2-normalization applied to both embeddings
- Temperature τ = 0.07 (CLIP default)
- Symmetric: image-to-text + text-to-image directions
- Scaled by batch_size for sum reduction consistency

#### Center Alignment Loss (Extension)

```python
def center_alignment_loss(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Center alignment loss between image and text embeddings.

    L_align = ||μ_g - μ_f||²

    where μ_g = mean(image_emb, dim=0) and μ_f = mean(text_emb, dim=0).
    """
    return F.mse_loss(image_emb.mean(0), text_emb.mean(0))
```

**Key specifications:**
- Computes mean of embeddings across batch dimension
- Aligns global centers of visual and textual distributions
- Standard MSE loss between mean vectors

#### Embedding MSE Loss (Extension)

```python
def embedding_mse_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Direct embedding MSE loss.

    L_emb = MSE(g, f) where g is image embedding and f is text embedding.

    This is a regression/embedding alignment objective, not a classification
    objective. Unlike euclidean_loss(), this operates on embeddings directly.
    """
    return F.mse_loss(image_emb, text_emb, reduction=reduction)
```

**Key specifications:**
- Direct MSE between paired embeddings (not scores vs targets)
- Regression objective for embedding alignment
- Supports `sum` reduction (λ=1.0 default in training)

### Model Architecture (Complete Code Specification)

#### Text Encoder ft(·): p → 300 → k

**File:** `models/text_encoder.py`

```python
class TextEncoder(nn.Module):
    def __init__(self, input_dim: int = 9763, hidden_dim: int = 300, output_dim: int = 50):
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # p → 300
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 300 → k
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)  # Small init
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)           # [B, p] → [B, 300]
        x = F.relu(x)             # [B, 300]
        x = self.fc2(x)           # [B, 300] → [B, k]
        return x                  # [B, k]

    def forward_with_hidden(self, x: torch.Tensor) -> tuple:
        h = F.relu(self.fc1(x))   # [C, p] → [C, 300] (hidden for conv predictor)
        out = self.fc2(h)         # [C, 300] → [C, k]
        return out, h             # ([C, k], [C, 300])
```

**Key specifications:**
- Input dimensions: TF-IDF=9763, SBERT=384, CLIP=512
- Hidden dimension: 300 (paper Sec 5.1)
- Output dimension k: 50 (paper Sec 5.1)
- Small initialization: std=0.01 on fc2.weight (paper Sec 5.1)

#### Image Encoder gv(·): Frozen Backbone → Projection

**File:** `models/image_encoder.py`

**VGG-19 (paper default):**
```python
# fc branch: VGG fc2 activation (4096-d)
fc_branch = vgg.classifier[0:5]  # Linear(25088→4096) → ReLU → Identity → Linear(4096→4096) → ReLU
projection = Sequential(
    nn.Linear(4096, 300),         # gv_hidden
    nn.ReLU(inplace=True),
    nn.Linear(300, k),             # k = 50
)
# conv branch: conv5_3 (512×14×14) → reduce → K' channels
conv_reduce = nn.Conv2d(512, 5, kernel_size=3, padding=1)  # K' = 5
conv_reduce_act = nn.ReLU(inplace=True)
```

**DenseNet-121:**
```python
# Conv branch: after denseblock3 → 1024×14×14
features_shared = nn.Sequential(*dense.features[:9])     # through denseblock3
# FC branch (default): through classifier
fc_branch = dense.classifier                              # Linear(1024→1000)
# FC branch (penultimate): skip classifier
fc_branch = nn.Identity()                                 # avgpool → 1024-d
conv_reduce = nn.Conv2d(1024, 5, kernel_size=3, padding=1)
```

**ResNet-50:**
```python
# Conv branch: after layer3 → 1024×14×14
features_shared = nn.Sequential(*resnet.children()[:7])   # through layer3
# FC branch (default): through fc
fc_branch = resnet.fc                                     # Linear(2048→1000)
# FC branch (penultimate): skip fc
fc_branch = nn.Identity()                                 # avgpool → 2048-d
conv_reduce = nn.Conv2d(1024, 5, kernel_size=3, padding=1)
```

**All backbones:**
- Pretrained weights frozen (no fine-tuning)
- BatchNorm layers forced to eval mode for deterministic features

#### Conv Weight Predictor: 300 → K'×3×3 Filters

**File:** `models/weight_predictor.py`

```python
class ConvWeightPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 300, k_prime: int = 5, filter_size: int = 3):
        self.fc = nn.Linear(hidden_dim, k_prime * filter_size * filter_size)  # 300 → 45
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)  # Small init
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, text_hidden: torch.Tensor) -> torch.Tensor:
        out = self.fc(text_hidden)    # [C, 300] → [C, 45]
        return out.view(*out.shape[:-1], k_prime, filter_size, filter_size)  # [C, 5, 3, 3]
```

#### Forward Pass by Model Type

**FC model (Sec 3.2):**
```python
# Score: y_c = w_c^T gv(x)
g = image_encoder(images)           # [B, k]
f = text_encoder(text_features)     # [C, k]
scores = g @ f.T                     # [B, C]
```

**Conv model (Sec 3.3):**
```python
conv_feat = image_encoder.forward_conv_feature(images)  # [B, K', H, W]
_, hidden = text_encoder.forward_with_hidden(text_features)  # [C, 300]
filters = conv_weight_predictor(hidden)                  # [C, K', 3, 3]
out = F.conv2d(conv_feat, filters, padding=1)            # [B, C, H, W]
scores = out.flatten(2).mean(2)                          # [B, C] (global avg pool)
```

**FC+Conv model (Sec 3.4):**
```python
# Shared prefix computed once
fc_emb, conv_feat = image_encoder.forward_both(images)
f, hidden = text_encoder.forward_with_hidden(text_features)

# FC branch
fc_scores = fc_emb @ f.T

# Conv branch
filters = conv_weight_predictor(hidden)
conv_scores = F.conv2d(conv_feat, filters, padding=1).flatten(2).mean(2)

# Joint
scores = fc_scores + conv_scores
```

### Text Encoder Pooling Mechanisms

**SBERT (document-level):**
```python
# data/text_sbert.py
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,  # L2-normalize
)
```

**SBERT-multi (sentence-level):**
```python
# data/text_sbert_multi.py
for text in texts:
    sentences = _split_into_sentences(text)  # split on `.!?;\n`
    sent_embeds = model.encode(sentences, normalize_embeddings=True)
    class_embed = sent_embeds.mean(axis=0)    # mean-pool sentences
    class_embed = class_embed / np.linalg.norm(class_embed)  # L2-normalize
```

**CLIP-text (document-level, EOS pooling):**
```python
# data/text_clip.py
inputs = tokenizer(texts, padding=True, truncation=True, max_length=77)
text_out = model.text_model(**inputs)
features = model.text_projection(text_out.pooler_output)  # [N, 512] EOS token
features = features / features.norm(dim=-1, keepdim=True)  # L2-normalize
```

**CLIP-multi (sentence-level):**
```python
# data/text_clip_multi.py
for text in texts:
    sentences = _split_into_sentences(text)
    # Encode in batches (batch_size=64) to avoid OOM
    sent_embeds = [_encode_sentences(batch) for batch in batches]
    class_embed = np.concatenate(sent_embeds).mean(axis=0)  # mean-pool
    class_embed = class_embed / np.linalg.norm(class_embed)  # L2-normalize
```

### Known Deviations from Paper

| Aspect | Paper | Our Implementation | Reason |
|--------|-------|-------------------|--------|
| Dataset (Tables 2, 3, Fig 2) | CUB-200-2010 (6,033 images) | CUB-200-2011 (11,788 images) | CUB-2010 no longer available |
| Learning rate (conv/fc+conv) | lr=1e-4 for all models | lr=5e-4 for conv/fc+conv | Empirical: improves convergence |
| Cross-validation (Tables 2–4) | 5-fold CV for all CUB experiments | Single run (`--n_folds 1`) in train.sh; train.py defaults to 5-fold | Training cost; Table 1 can use 5-fold |
| Framework | Torch (Lua) | PyTorch | Original Torch is deprecated |
| Sampler | Not specified (random) | ClassAwareSampler (default) with 50 classes/batch; RandomSampler with `--standard_sampler` | Better class diversity per batch |
| Early stopping | Not mentioned | Enabled by default (patience=20, min_epochs=50); disable with `--no_early_stopping` | Prevents overfitting |
| Fine-tuning | Table 4 models are fine-tuned on full dataset | No fine-tuning (all weights frozen) | Matches paper spirit for feature extraction |
| Wikipedia text | Texts collected ~10 years ago | Texts collected by us from current Wikipedia | Original repository not publicly available |
| Dropout in VGG classifier | Present in original VGG | Replaced with `nn.Identity()` for determinism | Frozen features must be deterministic |

---

## Data Preparation

The project expects data under `data/`:
- `images/birds/`: CUB-200-2011 images (200 classes, 11,788 images)
- `images/flowers/`: Oxford Flowers-102 images (102 classes, 8,189 images)
- `wikipedia/birds.jsonl`: Wikipedia texts for bird classes (included in repository)
- `wikipedia/flowers.jsonl`: Wikipedia texts for flower classes (included in repository)

Images are downloaded via `data/download_dataset.py` (or automatically by `train.sh`). Alternatively, download manually from the [Google Drive link](https://drive.google.com/file/d/1ki7MEb_LcPpqWF3HNN9S1UJ9hYzpr5mz/view) and unzip to `data/images/`.

Wikipedia texts were collected by us (the original paper's CUB-200-2010 texts are not publicly available).

VGG-19 features are extracted on-the-fly (frozen weights, no fine-tuning) following the paper protocol.

### Data Preprocessing Pipeline

**File:** `data/image_preprocessor.py`

```python
# Image preprocessing (paper Sec 5.1: shortest side → 224px, center crop 224×224)
transform = transforms.Compose([
    transforms.Resize(224),              # Shortest side → 224px
    transforms.CenterCrop(224),          # Center crop 224×224
    transforms.ToTensor(),
    transforms.Normalize(                # ImageNet mean/std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**TF-IDF Processing** (`data/text_processor.py`):

```python
# Paper Sec 5.2: 9763-d, log normalization
vectorizer = TfidfVectorizer(
    max_features=9763,
    sublinear_tf=True,     # log normalization: tf → 1 + log(tf)
)
# Output padded to exact 9763 dimensions if vocabulary is smaller
```

### Zero-Shot Data Split

**File:** `data/dataset.py`

**Seen/Unseen class partition:**
```python
# Dual-seed support:
#   unseen_seed — controls the seen/unseen class partition
#   split_seed  — controls the train/test image split within seen classes
# CUB: 40 unseen / 160 seen classes
# Flowers: 20 unseen / 82 seen classes

positions = list(range(n_classes))
random.seed(unseen_seed)
random.shuffle(positions)
unseen_classes = sorted(all_class_ids[p] for p in positions[:n_unseen])
seen_classes = sorted(all_class_ids[p] for p in positions[n_unseen:])
```

**Train/test split within seen classes (80/20):**
```python
random.seed(split_seed)
for cls in seen_classes:
    imgs = class_to_images[cls][:]  # sorted for determinism
    random.shuffle(imgs)
    split_idx = int(len(imgs) * train_ratio)  # 0.8 for paper
    train_samples = imgs[:split_idx]
    test_seen_samples = imgs[split_idx:]

# Unseen classes: all images go to test_unseen
for cls in unseen_classes:
    test_unseen_samples = class_to_images[cls]
```

### ClassAwareSampler

**File:** `data/sampler.py`

```python
# Groups samples by class, then uses round-robin sampling
# to ensure each batch contains diverse classes
# Default: 50 unique classes per batch (for 160 seen classes)

class ClassAwareSampler(Sampler):
    def __init__(self, dataset, batch_size=200, classes_per_batch=50, seed=42):
        # Group indices by class_id
        # Shuffle within each class pool
        # Round-robin across classes to form batches
```

**Benefits:**
- Better class diversity per batch
- More stable training for zero-shot learning
- Reduces class imbalance effects

---

## Reproducibility

The code is **reproducible**: with the same command you get the same results across runs.

- **Seed**: Training and evaluation use a fixed random seed (default `42`). Data splits, model initialization, and batch order are deterministic. Use `--seed` to change it. For multi-fold CV, fold_seed = seed + fold_idx.
- **GPU**: By default cuDNN uses non-deterministic algorithms for speed. For bit-exact GPU reproducibility, run with `--deterministic` (may be slower).
- **Evaluation**: `scripts/evaluate.py` sets the same seed (42) so evaluation is deterministic.

### Evaluate a single checkpoint

```bash
python scripts/evaluate.py --checkpoint checkpoints/fc_bce_cub_fc_40.pt --dataset cub
```

**Metrics computed:** ROC-AUC, PR-AUC, Top-1 accuracy, Top-5 accuracy (per-class mean, separately for seen and unseen splits).

**Evaluation pipeline:**
```python
# 1. Load dataset (test_seen and test_unseen splits)
test_seen_dataset = ZeroShotDataset(..., mode="test_seen", ...)
test_unseen_dataset = ZeroShotDataset(..., mode="test_unseen", ...)

# 2. Subset text features for evaluation split
text_features_all = dataset.text_features  # [C_total, text_dim]
seen_indices = [dataset.label_to_idx[c] for c in seen_classes]
text_features_seen = text_features_all[seen_indices]  # [C_seen, text_dim]

# 3. Label remapping for subset evaluation
seen_label_map_tensor = torch.full((num_classes,), -1)
for global_idx, subset_idx in seen_label_map.items():
    seen_label_map_tensor[global_idx] = subset_idx

# 4. Compute scores using subset text features
scores = model(images, text_features_seen)  # [B, C_seen]
remapped_labels = seen_label_map_tensor[labels]  # Global → subset indices
```

**Evaluation arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Model checkpoint path |
| `--model_type` | `fc` | `fc`, `conv`, or `fc+conv` |
| `--dataset` | `cub` | `cub` or `flowers` |
| `--k` | `50` | Joint embedding dimension (must match training) |
| `--ft_hidden` | `300` | Text encoder hidden dim (must match training) |
| `--gv_hidden` | `300` | Image fc-branch hidden dim (must match training) |
| `--conv_feature_layer` | `conv5_3` | Conv feature layer (must match training) |
| `--text_encoder` | `tfidf` | `tfidf`, `sbert`, `sbert_multi`, `clip`, or `clip_multi` |
| `--image_backbone` | `vgg19` | `vgg19`, `densenet121`, or `resnet50` |
| `--fc_mode` | `default` | `default` or `penultimate` |
| `--n_unseen` | 40/20 | Number of unseen classes (must match training) |
| `--train_ratio` | `0.8` | Train ratio (must match training) |

Use the same `--dataset`, `--n_unseen`, `--train_ratio`, `--text_encoder`, `--image_backbone`, and `--fc_mode` as training so the data split and model architecture match.

**Metrics computed:** ROC-AUC, PR-AUC, Top-1 accuracy, Top-5 accuracy (per-class mean, separately for seen and unseen splits).

### Evaluation Metrics and Computation

**File:** `scripts/evaluate.py`

#### ROC-AUC and PR-AUC (Per-Class Mean)

```python
def compute_metrics(
    scores: np.ndarray,  # [N, C] logits
    labels: np.ndarray,  # [N] class indices in [0, C-1]
    num_classes: int,
) -> dict[str, float]:
    """For each class c, compute binary ROC-AUC and PR-AUC, then mean across classes.

    For each class c:
    - y_true = (labels == c)  # Binary labels: 1 if sample belongs to class c
    - y_score = scores[:, c]  # Logits for class c
    - roc_auc = roc_auc_score(y_true, y_score)
    - pr_auc = average_precision_score(y_true, y_score)

    Skip classes with no positive samples or all positive samples.
    """
    roc_aucs, pr_aucs = [], []
    for c in range(num_classes):
        y_true = (labels == c).astype(np.float64)
        if y_true.sum() == 0 or y_true.sum() == n:  # Skip empty/full classes
            continue
        y_score = scores[:, c]
        roc_aucs.append(roc_auc_score(y_true, y_score))
        pr_aucs.append(average_precision_score(y_true, y_score))

    return {
        'roc_auc_mean': np.mean(roc_aucs),
        'pr_auc_mean': np.mean(pr_aucs),
    }
```

**Key specifications:**
- **Per-class binary classification**: Each class c is treated as a binary problem (positive=c, negative≠c)
- **ROC-AUC**: Area under ROC curve — TPR vs FPR at various thresholds
- **PR-AUC**: Area under precision-recall curve — more informative for imbalanced data
- **Mean aggregation**: Average across all classes (seen or unseen subset)
- **Skip criteria**: Classes with 0 or N positive samples are excluded (undefined AUC)

#### Top-k Accuracy

```python
def topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Top-k accuracy: correct if true label is in top-k predicted classes.

    Args:
        scores: [N, C] logits for N samples and C classes
        labels: [N] ground truth class indices
        k: Consider top-k predictions (default: 1)

    Returns:
        Fraction of samples where the true label is in top-k predictions.
    """
    pred = np.argsort(-scores, axis=1)[:, :k]  # [N, k] top-k indices per sample
    return (pred == labels.reshape(-1, 1)).any(axis=1).mean()
```

**Key specifications:**
- **Top-1 accuracy**: Correct if highest-scoring class matches ground truth
- **Top-5 accuracy**: Correct if ground truth is in top-5 scoring classes
- **Per-class mean**: Reported metrics are averaged across classes, not samples
- **Subset evaluation**: Computed separately for seen and unseen splits

**Evaluation output example:**
```
[test_seen] ROC-AUC (mean): 0.9810
[test_seen] PR-AUC (mean): 0.4920
[test_seen] Top-1 acc: 0.4960
[test_seen] Top-5 acc: 0.8260
[test_unseen] ROC-AUC (mean): 0.7120
[test_unseen] PR-AUC (mean): 0.0660
[test_unseen] Top-1 acc: 0.1120
[test_unseen] Top-5 acc: 0.4090
```

### Performance Optimizations (do not affect correctness)

- Multi-worker data loading (`NUM_WORKERS=8`)
- `pin_memory` and `persistent_workers` for faster GPU transfer
- `cudnn.benchmark=True` for optimized conv algorithms
- Vectorized label mapping and cached transform objects
- ClassAwareSampler (pre-computed batches for better class diversity)

---

## Reference

- **Paper**: Ba et al., "Predicting deep zero-shot convolutional neural networks using textual descriptions", ICCV 2015
- **Data**: `data/` directory (auto-loaded)
- **VGG-19 weights**: downloaded automatically on first run
