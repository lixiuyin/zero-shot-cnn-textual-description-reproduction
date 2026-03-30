"""
Table 2. Model performance using various objective functions.
Note: Paper Table 2 reports results on CUB-200-2010. Our 'Ours' column uses CUB-200-2011
(the only publicly available version, ~2x images). The 'Paper' column values are copied
from the original paper (Ba et al. ICCV 2015 Table 2).
Output: results/tables/Table2.csv, Table2.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import ImageClassDataset, prepare_birds_zero_shot
from scripts.reproduce.common import get_tables_dir, get_tex_dir, read_table_csv, resolve_with_cv, write_table_csv
from scripts.reproduce.eval_utils import (
    compute_zero_shot_metrics,
    evaluate_cv_folds,
    load_model,
    run_inference,
)
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


def _fmt_metric(m: dict, key: str) -> str:
    val = m.get(key)
    if val is None:
        return "—"
    return _fmt(val)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 from model evaluation.")
    parser.add_argument("--cub_root", type=str, default="", help="Path to CUB_200_2011")
    parser.add_argument("--wikipedia_birds", type=str, default="data/wikipedia/birds.jsonl", help="Path to birds Wikipedia text")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Default dir for checkpoints (used when --checkpoint_* not set)")
    parser.add_argument("--checkpoint_bce", type=str, default="", help="Checkpoint trained with BCE")
    parser.add_argument("--checkpoint_hinge", type=str, default="", help="Checkpoint trained with Hinge")
    parser.add_argument("--checkpoint_euclidean", type=str, default="", help="Checkpoint trained with Euclidean")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_folds", type=int, default=0,
                        help="Number of CV folds to average (0 = auto-detect from fold* dirs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed used during training (fold_seed = seed + fold_idx)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/test split ratio used during training (default: 0.8, paper value)")
    parser.add_argument("--n_unseen", type=int, default=None,
                        help="Unseen classes used during training (default: 40 for CUB)")
    parser.add_argument("--conv_feature_layer", type=str, default=CONV_FEATURE_LAYER,
                        choices=("conv5_3", "conv4_3", "pool5"),
                        help="VGG conv feature layer used during training (default: conv5_3)")
    parser.add_argument("--image_backbone", type=str, default="vgg19",
                        choices=("vgg19", "densenet121", "resnet50"),
                        help="Image backbone used during training (default: vgg19)")
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

    headers = ["Metrics", "BCE (Paper)", "BCE (Ours)", "Hinge (Paper)", "Hinge (Ours)", "Euclidean (Paper)", "Euclidean (Ours)"]
    rows = [
        ["unseen ROC-AUC", "0.82", "—", "0.795", "—", "0.70", "—"],
        ["seen ROC-AUC", "0.973", "—", "0.97", "—", "0.95", "—"],
        ["mean ROC-AUC", "0.937", "—", "0.934", "—", "0.90", "—"],
        ["unseen PR-AUC", "0.103", "—", "0.10", "—", "0.076", "—"],
        ["seen PR-AUC", "0.33", "—", "0.41", "—", "0.37", "—"],
        ["mean PR-AUC", "0.287", "—", "0.35", "—", "0.31", "—"],
        ["unseen class acc.", "0.01", "—", "0.006", "—", "0.12", "—"],
        ["seen class acc.", "0.35", "—", "0.43", "—", "0.263", "—"],
        ["mean class acc.", "0.17", "—", "0.205", "—", "0.19", "—"],
        ["unseen top-5 acc.", "0.176", "—", "0.182", "—", "0.428", "—"],
        ["seen top-5 acc.", "0.58", "—", "0.668", "—", "0.45", "—"],
        ["mean top-5 acc.", "0.38", "—", "0.41", "—", "0.44", "—"],
    ]
    # Column indices: BCE (Ours)=2, Hinge (Ours)=4, Euclidean (Ours)=6
    col_idx = {
        "Ours (BCE)": 2,
        "Ours (Hinge)": 4,
        "Ours (Euclidean)": 6,
    }
    metric_keys = [
        "roc_auc_unseen", "roc_auc_seen", "roc_auc_mean",
        "pr_auc_unseen", "pr_auc_seen", "pr_auc_mean",
        "top1_unseen", "top1_seen", "top1_mean",
        "top5_unseen", "top5_seen", "top5_mean",
    ]

    # Track which loss functions will be updated
    loss_functions_to_update = set()

    # Load existing data if available (to preserve results from previous runs)
    existing = read_table_csv(tables_dir, 2)
    if existing:
        existing_headers, existing_rows = existing
        # Determine which loss functions will be updated based on command line args
        if args.checkpoint_bce and Path(args.checkpoint_bce).exists():
            loss_functions_to_update.add("BCE")
        if args.checkpoint_hinge and Path(args.checkpoint_hinge).exists():
            loss_functions_to_update.add("Hinge")
        if args.checkpoint_euclidean and Path(args.checkpoint_euclidean).exists():
            loss_functions_to_update.add("Euclidean")

        # Merge: keep Paper values, preserve "Ours" values ONLY for loss functions NOT being updated
        for i, row in enumerate(existing_rows):
            if i < len(rows):
                # Check which loss function this row corresponds to by checking column 2
                # We preserve "Ours" values if the loss function is NOT being updated
                if "BCE" not in loss_functions_to_update and 2 < len(row) and row[2] != "—":
                    rows[i][2] = row[2]  # Preserve BCE (Ours)
                if "Hinge" not in loss_functions_to_update and 4 < len(row) and row[4] != "—":
                    rows[i][4] = row[4]  # Preserve Hinge (Ours)
                if "Euclidean" not in loss_functions_to_update and 6 < len(row) and row[6] != "—":
                    rows[i][6] = row[6]  # Preserve Euclidean (Ours)

    jsonl_birds = code_root / args.wikipedia_birds
    if not args.cub_root or not Path(args.cub_root).exists() or not jsonl_birds.exists():
        print("CUB root or wikipedia jsonl missing; writing table with placeholders.")
    else:
        try:
            n_unseen_cub = args.n_unseen if args.n_unseen is not None else 40
            out = prepare_birds_zero_shot(
                args.cub_root, jsonl_birds,
                n_unseen=n_unseen_cub,
                unseen_seed=args.seed,
                split_seed=args.seed,
            )
            train_p, train_l, test_p, test_l, class_names, text_feat, seen_idx, unseen_idx = out
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
                # Dataset-specific keys prevent cross-dataset checkpoint contamination
                (args.checkpoint_bce, "fc_bce_cub", "Ours (BCE)"),
                (args.checkpoint_hinge, "fc_hinge_cub", "Ours (Hinge)"),
                (args.checkpoint_euclidean, "fc_euclidean_cub", "Ours (Euclidean)"),
            ]
            for explicit_ckpt, cv_key, col_name in tqdm(models_to_eval, desc="Loss functions", unit="model"):
                root, fold_ckpts = resolve_with_cv(cv_key, args.n_folds, args.checkpoint_dir, explicit_ckpt)
                if len(fold_ckpts) >= 2:
                    print(f"  {col_name}: averaging {len(fold_ckpts)} CV folds (per-fold split)")
                    m = evaluate_cv_folds(
                        fold_ckpts,
                        "fc",
                        dataset="cub",
                        images_root=args.cub_root,
                        wikipedia_jsonl=str(jsonl_birds),
                        device=device,
                        batch_size=args.batch_size,
                        base_seed=args.seed,
                        n_unseen=n_unseen_cub,
                        train_ratio=args.train_ratio,
                        **model_kw,
                    )
                else:
                    ckpt = root
                    if not ckpt:
                        continue
                    model = load_model("fc", ckpt, device, **model_kw)
                    scores, labels = run_inference(model, loader, text_t, device, num_classes, desc=f"{col_name} inference")
                    m = compute_zero_shot_metrics(scores, labels, seen_idx, unseen_idx)
                j = col_idx[col_name]
                for i, key in enumerate(metric_keys):
                    rows[i][j] = _fmt_metric(m, key)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    path = write_table_csv(tables_dir, 2, headers, rows)
    print(f"Saved {path}")

    # Generate Table 2 LaTeX with style matching compile_all_tables.py
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{Model performance using various objective functions. Paper values are from CUB-200-2010; Ours from CUB-200-2011.}
\label{tab:table2}
\begin{tabular}{lrrrrrr}
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
    output_path = tex_dir / "Table2.tex"
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
