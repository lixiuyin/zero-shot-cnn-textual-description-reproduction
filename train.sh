#!/usr/bin/env bash
# =============================================================================
# train.sh -- Reproduce Ba et al. ICCV 2015 (paper experiments only)
#
# All runs use paper defaults:
#   text encoder  : TF-IDF (9763-d)
#   image backbone: VGG-19 (frozen)
#   loss          : BCE (unless noted)
#   batch size    : 200
#   optimizer     : Adam, lr=1e-4 (conv/fc+conv use 5e-4)
#   early stopping: enabled (patience=20, min_epochs=50)
#   CV folds      : 5 (default, seeds 42..46; checkpoints under fold{i}/)
#
# Checkpoint naming: {model}_{loss}_{dataset}_{layer}_{n_unseen}.pt
#   e.g. checkpoints/fold0/fc_bce_cub_fc_40.pt
#
# Expected total runtime: ~60 h on a single GPU (5x single-run)
# =============================================================================
set -euo pipefail

# -- Options ------------------------------------------------------------------
UPLOAD_HF=0
EPOCHS=200
N_FOLDS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload-hf)
            UPLOAD_HF=1
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --n-folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash train.sh [--upload-hf] [--epochs N] [--n-folds N]"
            echo "  --upload-hf   Upload checkpoints to HuggingFace when finished"
            echo "  --epochs N    Override default epochs (default: 200)"
            echo "  --n-folds N   Override CV folds (default: 1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash train.sh [--upload-hf] [--epochs N] [--n-folds N]"
            exit 1
            ;;
    esac
done

# -- Environment --------------------------------------------------------------
echo "=== Setting up environment ==="
uv sync && source .venv/bin/activate

echo "=== Downloading datasets (skipped if already present) ==="
cd data && python download_dataset.py && cd ..

# =============================================================================
# Part 1 -- Table 1: Model type comparison, CUB zero-shot
#           fc | conv | fc+conv  x  CUB  (40 unseen, BCE)
# =============================================================================
echo ""
echo "=== Part 1: CUB -- Model type ablation (Table 1) ==="

python scripts/train.py \
    --model_type fc \
    --dataset cub \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type conv \
    --dataset cub \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset cub \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# fc hinge and euclidean loss for comparison
python scripts/train.py \
    --model_type fc \
    --dataset cub \
    --loss hinge \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --no_early_stopping

python scripts/train.py \
    --model_type fc \
    --dataset cub \
    --loss euclidean \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# conv hinge loss for comparison (euclidean not supported for conv-only)
python scripts/train.py \
    --model_type conv \
    --dataset cub \
    --loss hinge \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# fc+conv hinge and euclidean loss for comparison
python scripts/train.py \
    --model_type fc+conv \
    --dataset cub \
    --loss hinge \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset cub \
    --loss euclidean \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS
# =============================================================================
# Part 2 -- Table 1: Model type comparison, Flowers zero-shot
#           fc | conv | fc+conv  x  Flowers  (20 unseen, BCE)
# =============================================================================
echo ""
echo "=== Part 2: Flowers -- Model type ablation (Table 1) ==="

python scripts/train.py \
    --model_type fc \
    --dataset flowers \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type conv \
    --dataset flowers \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset flowers \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# fc hinge and euclidean loss for comparison
python scripts/train.py \
    --model_type fc \
    --dataset flowers \
    --loss hinge \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc \
    --dataset flowers \
    --loss euclidean \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# conv hinge loss for comparison (euclidean not supported for conv-only)
python scripts/train.py \
    --model_type conv \
    --dataset flowers \
    --loss hinge \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# fc+conv hinge and euclidean loss for comparison
python scripts/train.py \
    --model_type fc+conv \
    --dataset flowers \
    --loss hinge \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset flowers \
    --loss euclidean \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS
# =============================================================================
# Part 3 -- Table 2: Loss function comparison (CUB, FC model)
#           bce (already trained above) | hinge | euclidean
# =============================================================================
echo ""
echo "=== Part 3: CUB -- Loss function ablation (Table 2) ==="
echo "Skipping: all models on both datasets (except the conv-only model with Euclidean loss) have already been trained."

# =============================================================================
# Part 4 -- Table 3: Conv feature layer ablation (CUB, FC+Conv)
#           conv5_3 (already trained) | conv4_3 | pool5
# =============================================================================
echo ""
echo "=== Part 4: CUB -- Conv feature layer ablation (Table 3) ==="

python scripts/train.py \
    --model_type fc+conv \
    --dataset cub \
    --conv_feature_layer conv4_3 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset cub \
    --conv_feature_layer pool5 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# =============================================================================
# Part 5 -- Table 4: Supervised baseline (50/50 split, n_unseen=0)
#           All classes seen; 50% train / 50% test within each class
# =============================================================================
echo ""
echo "=== Part 5: Supervised baseline -- 50/50 split (Table 4) ==="

python scripts/train.py \
    --model_type fc \
    --dataset cub \
    --n_unseen 0 \
    --train_ratio 0.5 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset cub \
    --n_unseen 0 \
    --train_ratio 0.5 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# Flowers needs more epochs for full-dataset convergence
python scripts/train.py \
    --model_type fc \
    --dataset flowers \
    --n_unseen 0 \
    --train_ratio 0.5 \
    --epochs 400 \
    --n_folds $N_FOLDS

python scripts/train.py \
    --model_type fc+conv \
    --dataset flowers \
    --n_unseen 0 \
    --train_ratio 0.5 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS

# =============================================================================
# Upload checkpoints to HuggingFace
# =============================================================================
echo ""
echo "=== Uploading paper checkpoints to HuggingFace ==="
if [ "$UPLOAD_HF" -eq 1 ]; then
    hf upload LiXiuyin/zero-shot-cnn-comp7404-group17 checkpoints/ . --repo-type model
    echo "Upload completed"
else
    printf "Do you want to upload checkpoints to HuggingFace? (y/n): "
    read -r confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        hf upload LiXiuyin/zero-shot-cnn-comp7404-group17 checkpoints/ . --repo-type model
        echo "Upload completed"
    else
        echo "Upload skipped"
    fi
fi

echo ""
echo "=== train.sh complete ==="