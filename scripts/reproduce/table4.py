"""
Table 4. Performance of our model trained on the full dataset, 50/50 split per class.
Results are computed by evaluating on the 50% test split (same classes as train).
Output: results/tables/Table4.csv, Table4.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import ImageClassDataset, prepare_birds_50_50, prepare_flowers_50_50
from scripts.reproduce.common import get_tables_dir, get_tex_dir, read_table_csv, resolve_checkpoint as _resolve_checkpoint, write_table_csv
from scripts.reproduce.eval_utils import compute_mean_metrics, load_model, run_inference
from utils.config import (
    CONV_CHANNELS,
    CONV_FEATURE_LAYER,
    FT_HIDDEN,
    GV_HIDDEN,
    K,
    TEXT_DIM,
)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 4 from model evaluation.")
    parser.add_argument("--cub_root", type=str, default="", help="Path to CUB_200_2011")
    parser.add_argument("--flowers_root", type=str, default="", help="Single Flowers root (102 class dirs) for 50/50 split (Table 4)")
    parser.add_argument("--wikipedia_birds", type=str, default="data/wikipedia/birds.jsonl", help="Path to birds Wikipedia text")
    parser.add_argument("--wikipedia_flowers", type=str, default="data/wikipedia/flowers.jsonl", help="Path to flowers Wikipedia text")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Default dir for checkpoints (used when --checkpoint_* not set)")
    parser.add_argument("--checkpoint_fc", type=str, default="")
    parser.add_argument("--checkpoint_fc_conv", type=str, default="")
    parser.add_argument("--conv_feature_layer", type=str, default=CONV_FEATURE_LAYER,
                        choices=("conv5_3", "conv4_3", "pool5"),
                        help="Conv feature layer used during training (must match checkpoint)")
    parser.add_argument("--image_backbone", type=str, default="vgg19",
                        choices=("vgg19", "densenet121", "resnet50"),
                        help="Image backbone used during training (default: vgg19)")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    tables_dir = get_tables_dir()
    tex_dir = get_tex_dir()
    if args.out_dir:
        tables_dir = Path(args.out_dir) / "tables"
        tex_dir = Path(args.out_dir) / "tex"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tex_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_kw = dict(
        text_dim=TEXT_DIM,
        k=K,
        ft_hidden=FT_HIDDEN,
        gv_hidden=GV_HIDDEN,
        conv_channels=CONV_CHANNELS,
        conv_feature_layer=args.conv_feature_layer,
        image_backbone=args.image_backbone,
    )

    headers = ["Model", "CUB-2011 (Paper)", "CUB-2011 (Ours)", "Oxford Flowers (Paper)", "Oxford Flowers (Ours)"]
    rows = [
        ["Ours (fc)", "0.64", "—", "0.73", "—"],
        ["Ours (fc+conv)", "0.66", "—", "0.77", "—"],
    ]
    # Table 4: Paper results from Ba et al. ICCV 2015; Ours results from 50/50 split evaluation
    cub_metric_fc = None
    cub_metric_fc_conv = None
    flower_metric_fc = None
    flower_metric_fc_conv = None

    # Track which datasets are being evaluated in this run
    datasets_to_update = set()

    # Load existing data if available (to preserve results from previous runs)
    existing = read_table_csv(tables_dir, 4)
    if existing:
        existing_headers, existing_rows = existing
        # Determine which datasets will be updated based on command line args
        if args.cub_root and Path(args.cub_root).exists():
            datasets_to_update.add("CUB")
        if args.flowers_root and Path(args.flowers_root).exists():
            datasets_to_update.add("Flowers")

        # Merge: keep Paper values, preserve "Ours" values ONLY for datasets NOT being updated
        for i, row in enumerate(existing_rows):
            if i < len(rows):
                # Row structure: Model, CUB (Paper), CUB (Ours), Flowers (Paper), Flowers (Ours)
                # Only preserve "Ours" values for datasets NOT being updated
                if "CUB" not in datasets_to_update and 2 < len(row) and row[2] != "—":
                    rows[i][2] = row[2]  # Preserve CUB (Ours) if not updating CUB
                if "Flowers" not in datasets_to_update and 4 < len(row) and row[4] != "—":
                    rows[i][4] = row[4]  # Preserve Flowers (Ours) if not updating Flowers

    if args.cub_root and Path(args.cub_root).exists():
        jsonl_birds = code_root / args.wikipedia_birds
        if jsonl_birds.exists():
            try:
                train_p, train_l, test_p, test_l, class_names, text_feat = prepare_birds_50_50(
                    args.cub_root, jsonl_birds
                )
                num_classes = len(class_names)
                text_t = torch.from_numpy(text_feat).float()
                loader = DataLoader(
                    ImageClassDataset(test_p, test_l),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                from tqdm import tqdm

                models_to_eval = [
                    (_resolve_checkpoint("fc_bce_cub_5050", args.checkpoint_dir, args.checkpoint_fc), "fc", 2),
                    (_resolve_checkpoint("fc_conv_bce_cub_5050", args.checkpoint_dir, args.checkpoint_fc_conv), "fc+conv", 2),
                ]
                for ckpt_arg, model_type, col_idx in tqdm(models_to_eval, desc="CUB 50/50 models", unit="model"):
                    if not ckpt_arg:
                        continue
                    model = load_model(model_type, ckpt_arg, device, **model_kw)
                    scores, labels = run_inference(model, loader, text_t, device, num_classes, desc=f"{model_type} inference")
                    m = compute_mean_metrics(scores, labels, num_classes)
                    row_idx = 0 if model_type == "fc" else 1
                    # For Table 4 we report mean Top-1 accuracy on the test split
                    rows[row_idx][col_idx] = _fmt(m["top1_mean"])  # Use col_idx directly
            except Exception as e:
                print(f"CUB 50/50 evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    # Flowers: 50/50 per class (paper Table 4)
    jsonl_flowers = code_root / args.wikipedia_flowers
    if jsonl_flowers.exists() and args.flowers_root and Path(args.flowers_root).exists():
        try:
            train_p, train_l, test_p, test_l, class_names, text_feat = prepare_flowers_50_50(
                args.flowers_root, jsonl_flowers
            )
            num_classes = len(class_names)
            text_t = torch.from_numpy(text_feat).float()
            loader = DataLoader(
                ImageClassDataset(test_p, test_l),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            from tqdm import tqdm

            models_to_eval = [
                (_resolve_checkpoint("fc_bce_flowers_5050", args.checkpoint_dir, args.checkpoint_fc), "fc", 4),
                (_resolve_checkpoint("fc_conv_bce_flowers_5050", args.checkpoint_dir, args.checkpoint_fc_conv), "fc+conv", 4),
            ]
            for ckpt_arg, model_type, col_idx in tqdm(models_to_eval, desc="Flowers 50/50 models", unit="model"):
                if not ckpt_arg:
                    continue
                model = load_model(model_type, ckpt_arg, device, **model_kw)
                scores, labels = run_inference(model, loader, text_t, device, num_classes, desc=f"{model_type} inference")
                m = compute_mean_metrics(scores, labels, num_classes)
                row_idx = 0 if model_type == "fc" else 1
                # For Table 4 we report mean Top-1 accuracy on the test split
                rows[row_idx][col_idx] = _fmt(m["top1_mean"])  # Use col_idx directly
        except Exception as e:
                print(f"Flowers evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    # Results are already filled in during evaluation above

    path = write_table_csv(tables_dir, 4, headers, rows)
    print(f"Saved {path}")

    # Generate Table 4 LaTeX with style matching compile_all_tables.py
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{Performance trained on full dataset with 50/50 train/test split per class (Top-1 accuracy).}
\label{tab:table4}
\begin{tabular}{lrrrr}
\toprule
"""
    content += " & ".join(str(c) for c in headers) + r" \\ \midrule" + "\n"

    for row in rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r""" \bottomrule
\end{tabular}
\end{table}

"""
    # Write to file
    output_path = tex_dir / "Table4.tex"
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
