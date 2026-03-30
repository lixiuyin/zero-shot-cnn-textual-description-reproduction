#!/usr/bin/env bash
# =============================================================================
# reproduce.sh -- Generate paper tables and figures (Ba et al. ICCV 2015)
#
# Requires bash 4+ (associative arrays). On macOS, use: brew install bash
#
# Checkpoint mode (set USE_CONFIG below):
#   CV     -- train.sh default: 5-fold CV, checkpoints in checkpoints/fold{i}/
#   SINGLE -- single run (--n_folds 1), checkpoints in checkpoints/
# =============================================================================
set -euo pipefail

# -- Options ------------------------------------------------------------------
INSTALL_LATEX=0
for arg in "$@"; do
    case "$arg" in
        --install-latex)
            INSTALL_LATEX=1
            ;;
        --help|-h)
            echo "Usage: bash reproduce.sh [--install-latex]"
            echo "  --install-latex   Auto-install LaTeX (xelatex) if missing"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash reproduce.sh [--install-latex]"
            exit 1
            ;;
    esac
done

# -- LaTeX installation (optional, cross-platform) ----------------------------
echo "=== Checking LaTeX installation ==="
if command -v xelatex &> /dev/null; then
    echo "XeLaTeX found"
elif [ "$INSTALL_LATEX" -eq 1 ]; then
    if command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu system"
        echo "Installing texlive (this may take a while)..."
        sudo apt update
        sudo apt install -y texlive-full
    elif command -v brew &> /dev/null; then
        echo "Detected macOS system"
        echo "Installing MacTeX (this may take a while)..."
        brew install mactex
    else
        echo "WARNING: No package manager found. Please install LaTeX manually."
        echo "  - Linux: sudo apt install texlive-full"
        echo "  - macOS: brew install mactex"
    fi
else
    echo "XeLaTeX not found; skipping auto-install (pass --install-latex to enable)."
fi

# -- Download checkpoints from HuggingFace ------------------------------------
if [ -d "checkpoints" ] && [ -n "$(ls -A checkpoints 2>/dev/null)" ]; then
    echo "Checkpoints directory already exists and is non-empty, skipping download"
else
    printf "Do you want to download checkpoints from HuggingFace? (y/n): "
    read -r choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        git clone https://huggingface.co/LiXiuyin/zero-shot-cnn-comp7404-group17 checkpoints
        echo "Checkpoints downloaded successfully"
    else
        echo "Download skipped"
    fi
fi

# -- Environment --------------------------------------------------------------
echo ""
echo "=== Setting up environment ==="
uv sync && source .venv/bin/activate

echo ""
echo "=== Downloading datasets (if needed) ==="
cd data && python download_dataset.py && cd ..

# -- Checkpoint auto-detection -------------------------------------------------
# All checkpoints are auto-detected by pattern from checkpoints/ (root + fold*/).
# No manual path configuration needed — just ensure checkpoints/ has the files.
OUT_DIR="results"

# -- Table 1: Model type comparison (CUB + Flowers) ---------------------------
echo ""
echo "=== Generating Table 1: Model type comparison ==="

python scripts/reproduce/table1.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir "$OUT_DIR"

# -- Table 2: Loss function comparison (CUB) ----------------------------------
echo ""
echo "=== Generating Table 2: Loss function comparison ==="

python scripts/reproduce/table2.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

# -- Table 3: Conv feature layer ablation (CUB) -------------------------------
echo ""
echo "=== Generating Table 3: Conv feature layer ablation ==="

python scripts/reproduce/table3.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

# -- Table 4: Supervised baseline 50/50 split ---------------------------------
echo ""
echo "=== Generating Table 4: Supervised baseline (50/50 split) ==="

python scripts/reproduce/table4.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir "$OUT_DIR"

# -- Compile all tables to LaTeX ----------------------------------------------
echo ""
echo "=== Compiling all tables to LaTeX ==="

python scripts/reproduce/compile_all_tables.py
if [ -f "results/tex/AllTables.tex" ]; then
    xelatex -output-directory=results results/tex/AllTables.tex
    echo "LaTeX compilation complete"
else
    echo "WARNING: AllTables.tex not found, skipping LaTeX compilation"
fi

# -- Figure 2: Word sensitivity + Nearest neighbor retrieval ------------------
echo ""
echo "=== Generating Figure 2: Word sensitivity + Nearest neighbor ==="

python scripts/reproduce/figure2.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

# -- Figure 5: Conv filter visualization (Appendix) --------------------------
echo ""
echo "=== Generating Figure 5: Conv filter visualization (Appendix) ==="

python scripts/reproduce/figure5.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir "$OUT_DIR"

echo ""
echo "=== reproduce.sh complete ==="
echo ""
echo "Results saved to: $OUT_DIR/"
echo "  - tables/ : CSV data files"
echo "  - tex/    : LaTeX table files"
echo "  - figures/: Figure images"
