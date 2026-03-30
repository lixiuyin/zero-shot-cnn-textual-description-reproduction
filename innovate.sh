#!/usr/bin/env bash
# =============================================================================
# innovate.sh -- Extension experiments beyond Ba et al. ICCV 2015
#
# All checkpoints are saved under checkpoints/innov/.
# For N_FOLDS > 1, folds go to checkpoints/innov/fold{i}/<name>.pt automatically.
#
# Primary dataset: CUB-200-2011 (40 unseen classes, 80/20 seen split).
# Base model: fc+conv for all sections.
#
# Section A -- Loss function ablation (fc+conv, CUB, VGG-19, TF-IDF)
#   A1  clip_contrastive_loss          (--use_clip_loss)
#   A2  center_alignment_loss          (--use_center_align)
#   A3  embedding_mse_loss             (--use_embedding_loss)
#
# Section B -- Text encoder ablation (fc+conv, CUB, VGG-19)
#   B1  SBERT (all-MiniLM-L6-v2, 384-d)
#   B2  SBERT-multi (sentence-level pooling, 384-d)
#   B3  CLIP Text Encoder (ViT-B/32, 512-d)
#   B4  CLIP-multi (sentence-level pooling, ViT-B/32, 512-d)
#
# Section C -- Image backbone ablation (fc+conv, CUB, TF-IDF)
#   NOTE: conv branch for DenseNet-121 uses denseblock3 output (1024×14×14).
#         conv branch for ResNet-50 uses layer3 output (1024×14×14).
#         Both match VGG conv5_3 spatial resolution (14×14).
#   C1  fc+conv + DenseNet-121
#   C2  fc+conv + ResNet-50
#   C3  fc+conv + DenseNet-121 (penultimate: skip classifier, 1024-d)
#   C4  fc+conv + ResNet-50 (penultimate: skip fc, 2048-d)
#   C5  fc-only + DenseNet-121 (penultimate: avgpool→1024-d)
#   C6  fc-only + ResNet-50 (penultimate: avgpool→2048-d)
# =============================================================================
set -euo pipefail

# -- Options ------------------------------------------------------------------
UPLOAD_HF=0
for arg in "$@"; do
    case "$arg" in
        --upload-hf)
            UPLOAD_HF=1
            ;;
        --help|-h)
            echo "Usage: bash innovate.sh [--upload-hf]"
            echo "  --upload-hf   Upload checkpoints to HuggingFace when finished"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash innovate.sh [--upload-hf]"
            exit 1
            ;;
    esac
done

# -- Environment --------------------------------------------------------------
echo "=== Setting up environment ==="
uv sync && source .venv/bin/activate

echo "=== Downloading datasets (skipped if already present) ==="
cd data && python download_dataset.py && cd ..

EPOCHS=200
DATASET=cub
N_FOLDS=1          # set to e.g. 5 for cross-validation
INNOV_DIR="checkpoints/innov"

mkdir -p "$INNOV_DIR"

# -- Shared evaluation helper -------------------------------------------------
# Usage: run_eval <save_base> [extra evaluate.py args...]
# save_base: path WITHOUT .pt suffix (e.g. checkpoints/innov/fc_conv_bce)
# For N_FOLDS=1 evaluates save_base.pt; for N_FOLDS>1 evaluates each fold.
run_eval() {
    local save_base=$1; shift
    if [ "$N_FOLDS" -gt 1 ]; then
        local evaluated=0
        for i in $(seq 0 $((N_FOLDS - 1))); do
            local ckpt
            ckpt="$(dirname "$save_base")/fold${i}/$(basename "$save_base").pt"
            if [ -f "$ckpt" ]; then
                echo "  [evaluate] fold${i}: $ckpt"
                python scripts/evaluate.py \
                    --checkpoint "$ckpt" \
                    --dataset "$DATASET" \
                    "$@"
                evaluated=$((evaluated + 1))
            else
                echo "  [warn] missing checkpoint for fold${i}: $ckpt"
            fi
        done
        if [ "$evaluated" -eq 0 ]; then
            echo "ERROR: no fold checkpoints found for base: $save_base"
            exit 1
        fi
        if [ "$evaluated" -lt "$N_FOLDS" ]; then
            echo "ERROR: evaluated $evaluated/$N_FOLDS folds for base: $save_base"
            exit 1
        fi
    else
        echo "  [evaluate] ${save_base}.pt"
        if [ ! -f "${save_base}.pt" ]; then
            echo "ERROR: missing checkpoint: ${save_base}.pt"
            exit 1
        fi
        python scripts/evaluate.py \
            --checkpoint "${save_base}.pt" \
            --dataset "$DATASET" \
            "$@"
    fi
}

# =============================================================================
# Section A -- Loss function ablation (fc+conv, CUB, VGG-19, TF-IDF)
# =============================================================================
echo ""
echo "======================================================================="
echo "=== Section A: Loss function ablation (fc+conv, CUB) ==="
echo "======================================================================="

# -- A1: clip_contrastive_loss ------------------------------------------------
echo ""
echo "--- A1: fc+conv + BCE + clip_contrastive_loss (weight=0.1, temp=0.07) ---"
A1_SAVE="$INNOV_DIR/fc_conv_clip"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --loss bce \
    --use_clip_loss \
    --clip_weight 0.1 \
    --clip_temperature 0.07 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$A1_SAVE"
run_eval "$A1_SAVE" --model_type fc+conv

# -- A2: center_alignment_loss ------------------------------------------------
echo ""
echo "--- A2: fc+conv + BCE + center_alignment_loss (weight=0.1) ---"
A2_SAVE="$INNOV_DIR/fc_conv_center_align"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --loss bce \
    --use_center_align \
    --center_align_weight 0.1 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$A2_SAVE"
run_eval "$A2_SAVE" --model_type fc+conv

# -- A3: embedding_mse_loss ---------------------------------------------------
echo ""
echo "--- A3: fc+conv + BCE + embedding_mse_loss (weight=1.0) ---"
A3_SAVE="$INNOV_DIR/fc_conv_embedding_mse"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --loss bce \
    --use_embedding_loss \
    --embedding_weight 1.0 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$A3_SAVE"
run_eval "$A3_SAVE" --model_type fc+conv

# =============================================================================
# Section B -- Text encoder ablation (fc+conv, CUB, VGG-19)
# =============================================================================
echo ""
echo "======================================================================="
echo "=== Section B: Text encoder ablation (fc+conv, CUB, VGG-19) ==="
echo "======================================================================="

# -- B1: SBERT (all-MiniLM-L6-v2, 384-d) -------------------------------------
echo ""
echo "--- B1: fc+conv + SBERT text encoder (384-d) ---"
B1_SAVE="$INNOV_DIR/fc_conv_sbert"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --text_encoder sbert \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$B1_SAVE"
run_eval "$B1_SAVE" --model_type fc+conv --text_encoder sbert

# -- B2: SBERT-multi (sentence-level pooling, 384-d) --------------------------
echo ""
echo "--- B2: fc+conv + SBERT-multi text encoder (384-d) ---"
B2_SAVE="$INNOV_DIR/fc_conv_sbert_multi"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --text_encoder sbert_multi \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$B2_SAVE"
run_eval "$B2_SAVE" --model_type fc+conv --text_encoder sbert_multi

# -- B3: CLIP Text Encoder (ViT-B/32, 512-d) ----------------------------------
echo ""
echo "--- B3: fc+conv + CLIP text encoder (512-d) ---"
B3_SAVE="$INNOV_DIR/fc_conv_clip_text"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --text_encoder clip \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$B3_SAVE"
run_eval "$B3_SAVE" --model_type fc+conv --text_encoder clip

# -- B4: CLIP-multi (sentence-level pooling, 512-d) ---------------------------
echo ""
echo "--- B4: fc+conv + CLIP-multi text encoder (512-d, sentence-level pooling) ---"
B4_SAVE="$INNOV_DIR/fc_conv_clip_multi"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --text_encoder clip_multi \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$B4_SAVE"
run_eval "$B4_SAVE" --model_type fc+conv --text_encoder clip_multi

# =============================================================================
# Section C -- Image backbone ablation (fc+conv, CUB, TF-IDF)
#
# Both backbones now support the conv branch:
#   DenseNet-121: conv branch taps after denseblock3 → [B, 1024, 14×14]
#   ResNet-50:    conv branch taps after layer3      → [B, 1024, 14×14]
# Both match VGG conv5_3 spatial resolution (14×14), enabling fair comparison.
# =============================================================================
echo ""
echo "======================================================================="
echo "=== Section C: Image backbone ablation (fc+conv, CUB, TF-IDF) ==="
echo "======================================================================="

# -- C1: fc+conv + DenseNet-121 (default: through classifier 1000-d) ----------
echo ""
echo "--- C1: fc+conv + DenseNet-121 (default fc: avgpool→classifier→1000-d) ---"
C1_SAVE="$INNOV_DIR/fc_conv_densenet121"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --image_backbone densenet121 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$C1_SAVE"
run_eval "$C1_SAVE" --model_type fc+conv --image_backbone densenet121

# -- C2: fc+conv + ResNet-50 (default: through fc 1000-d) --------------------
echo ""
echo "--- C2: fc+conv + ResNet-50 (default fc: avgpool→fc→1000-d) ---"
C2_SAVE="$INNOV_DIR/fc_conv_resnet50"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --image_backbone resnet50 \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$C2_SAVE"
run_eval "$C2_SAVE" --model_type fc+conv --image_backbone resnet50

# -- C3: fc+conv + DenseNet-121 (penultimate: skip classifier, 1024-d) -------
echo ""
echo "--- C3: fc+conv + DenseNet-121 (penultimate: avgpool→1024-d, skip classifier) ---"
C3_SAVE="$INNOV_DIR/fc_conv_densenet121_penult"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --image_backbone densenet121 \
    --fc_mode penultimate \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$C3_SAVE"
run_eval "$C3_SAVE" --model_type fc+conv --image_backbone densenet121 --fc_mode penultimate

# -- C4: fc+conv + ResNet-50 (penultimate: skip fc, 2048-d) ------------------
echo ""
echo "--- C4: fc+conv + ResNet-50 (penultimate: avgpool→2048-d, skip fc) ---"
C4_SAVE="$INNOV_DIR/fc_conv_resnet50_penult"
python scripts/train.py \
    --model_type fc+conv \
    --dataset $DATASET \
    --image_backbone resnet50 \
    --fc_mode penultimate \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$C4_SAVE"
run_eval "$C4_SAVE" --model_type fc+conv --image_backbone resnet50 --fc_mode penultimate

# -- C5: fc-only + DenseNet-121 (penultimate) --------------------------------
echo ""
echo "--- C5: fc-only + DenseNet-121 (penultimate: avgpool→1024-d) ---"
C5_SAVE="$INNOV_DIR/fc_densenet121_penult"
python scripts/train.py \
    --model_type fc \
    --dataset $DATASET \
    --image_backbone densenet121 \
    --fc_mode penultimate \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$C5_SAVE"
run_eval "$C5_SAVE" --model_type fc --image_backbone densenet121 --fc_mode penultimate

# -- C6: fc-only + ResNet-50 (penultimate) -----------------------------------
echo ""
echo "--- C6: fc-only + ResNet-50 (penultimate: avgpool→2048-d) ---"
C6_SAVE="$INNOV_DIR/fc_resnet50_penult"
python scripts/train.py \
    --model_type fc \
    --dataset $DATASET \
    --image_backbone resnet50 \
    --fc_mode penultimate \
    --epochs $EPOCHS \
    --n_folds $N_FOLDS \
    --save "$C6_SAVE"
run_eval "$C6_SAVE" --model_type fc --image_backbone resnet50 --fc_mode penultimate

# =============================================================================
# Summary table
# =============================================================================
echo ""
echo "======================================================================="
echo "=== Generating innovation summary table ==="
echo "======================================================================="
python scripts/reproduce/table_innov.py \
    --cub_root data/images/birds \
    --innov_dir "$INNOV_DIR" \
    --n_folds "$N_FOLDS"

# =============================================================================
# Upload checkpoints to HuggingFace
# =============================================================================
echo ""
echo "=== Uploading innovation checkpoints to HuggingFace ==="
if [ "$UPLOAD_HF" -eq 1 ]; then
    hf upload LiXiuyin/zero-shot-cnn-comp7404-group17 checkpoints/ . --repo-type model
    echo "Upload completed"
else
    read -r -p "Do you want to upload checkpoints to HuggingFace? (y/n): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        hf upload LiXiuyin/zero-shot-cnn-comp7404-group17 checkpoints/ . --repo-type model
        echo "Upload completed"
    else
        echo "Upload skipped"
    fi
fi

echo ""
echo "=== innovate.sh complete ==="
echo ""
echo "Checkpoint summary (all under $INNOV_DIR/):"
echo "  A1 (CLIP loss)               : fc_conv_clip.pt"
echo "  A2 (center align)            : fc_conv_center_align.pt"
echo "  A3 (embedding MSE)           : fc_conv_embedding_mse.pt"
echo "  B1 (SBERT)                   : fc_conv_sbert.pt"
echo "  B2 (SBERT-multi)             : fc_conv_sbert_multi.pt"
echo "  B3 (CLIP text)               : fc_conv_clip_text.pt"
echo "  B4 (CLIP-multi)              : fc_conv_clip_multi.pt"
echo "  C1 (DenseNet default)        : fc_conv_densenet121.pt"
echo "  C2 (ResNet default)          : fc_conv_resnet50.pt"
echo "  C3 (DenseNet penultimate)    : fc_conv_densenet121_penult.pt"
echo "  C4 (ResNet penultimate)      : fc_conv_resnet50_penult.pt"
echo "  C5 (DenseNet fc-only penult) : fc_densenet121_penult.pt"
echo "  C6 (ResNet fc-only penult)   : fc_resnet50_penult.pt"
echo ""
echo "For CV (N_FOLDS>1), checkpoints live in $INNOV_DIR/fold{i}/<name>.pt"
echo "Summary table: results/tables/TableInnov.csv + results/tex/TableInnov.tex"
