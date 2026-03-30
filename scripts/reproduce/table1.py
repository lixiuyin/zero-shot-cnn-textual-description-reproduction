"""
Table 1. ROC-AUC and PR-AUC(AP) performance compared to other methods.
Results are computed by running the loaded model on CUB and Oxford Flowers.
Baseline rows (DAP, SSE, DA) are left as "—" (no baseline model in this repo).
Output: results/tables/Table1.csv, Table1.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import ImageClassDataset, prepare_birds_zero_shot, prepare_flowers_zero_shot
from scripts.reproduce.common import get_tables_dir, get_tex_dir, read_table_csv, resolve_checkpoint, resolve_cv_checkpoints, write_table_csv, resolve_with_cv
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


def _fmt(x) -> str:
    if x is None or (isinstance(x, str) and x.strip() == "—"):
        return "—"
    return f"{float(x):.3f}"


def _best_across_losses(
    loss_keys: list[str],
    model_type: str,
    loader,
    text_t,
    device,
    num_classes: int,
    seen_idx,
    unseen_idx,
    model_kw: dict,
    n_folds: int,
    checkpoint_dir: str,
    cv_kw: dict,
) -> dict | None:
    """Evaluate per-loss checkpoints; return metrics from the best (highest roc_auc_mean).

    For each key in loss_keys:
      - If CV fold dirs exist → evaluate_cv_folds (mean across folds)
      - Else if root checkpoint exists → single checkpoint evaluation
    No fallback — if a key has no matching file it is skipped.
    """
    best_m, best_score = None, -1.0

    for key in loss_keys:
        root, fold_ckpts = resolve_with_cv(key, n_folds, checkpoint_dir)
        if len(fold_ckpts) >= 2:
            m = evaluate_cv_folds(fold_ckpts, model_type, **cv_kw)
        elif root:
            mdl = load_model(model_type, root, device, **model_kw)
            scores, labels = run_inference(
                mdl, loader, text_t, device, num_classes,
                desc=f"{model_type}/{key}",
            )
            m = compute_zero_shot_metrics(scores, labels, seen_idx, unseen_idx)
        else:
            continue
        if m and m.get("roc_auc_mean", 0.0) > best_score:
            best_score, best_m = m["roc_auc_mean"], m

    return best_m


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 1 from model evaluation.")
    parser.add_argument("--cub_root", type=str, default="", help="Path to CUB_200_2011")
    parser.add_argument("--flowers_root", type=str, default="", help="Path to Oxford Flowers-102")
    parser.add_argument("--wikipedia_birds", type=str, default="data/wikipedia/birds.jsonl")
    parser.add_argument("--wikipedia_flowers", type=str, default="data/wikipedia/flowers.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Default dir for checkpoints (e.g. checkpoints/); used when --checkpoint_* not set")
    # Checkpoint directory — all checkpoints are auto-detected by key pattern
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
                        help="Unseen classes used during training (default: 40 CUB / 20 Flowers)")
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
    # Reorganized table structure: Paper and Ours side-by-side for easy comparison
    # Keeps all detailed metrics (unseen/seen/mean) from the paper
    headers = [
        "Dataset",
        "Model",
        "ROC-AUC unseen (Paper)",
        "ROC-AUC unseen (Ours)",
        "ROC-AUC seen (Paper)",
        "ROC-AUC seen (Ours)",
        "ROC-AUC mean (Paper)",
        "ROC-AUC mean (Ours)",
        "PR-AUC unseen (Paper)",
        "PR-AUC unseen (Ours)",
        "PR-AUC seen (Paper)",
        "PR-AUC seen (Ours)",
        "PR-AUC mean (Paper)",
        "PR-AUC mean (Ours)",
    ]
    # Note: Paper results from Ba et al., ICCV 2015
    # Ours results are automatically computed from evaluation
    rows = [
        ["CU-Bird200-2011", "fc", "0.82", "—", "0.974", "—", "0.943", "—", "0.11", "—", "0.33", "—", "0.286", "—"],
        ["CU-Bird200-2011", "conv", "0.80", "—", "0.96", "—", "0.925", "—", "0.085", "—", "0.15", "—", "0.14", "—"],
        ["CU-Bird200-2011", "fc+conv", "0.85", "—", "0.98", "—", "0.953", "—", "0.13", "—", "0.37", "—", "0.31", "—"],
        ["Oxford Flower", "fc", "0.70", "—", "0.987", "—", "0.90", "—", "0.07", "—", "0.65", "—", "0.52", "—"],
        ["Oxford Flower", "conv", "0.65", "—", "0.97", "—", "0.85", "—", "0.054", "—", "0.61", "—", "0.46", "—"],
        ["Oxford Flower", "fc+conv", "0.71", "—", "0.989", "—", "0.93", "—", "0.067", "—", "0.69", "—", "0.56", "—"],
    ]
    row_key = {
        ("CU-Bird200-2011", "fc"): 0,
        ("CU-Bird200-2011", "conv"): 1,
        ("CU-Bird200-2011", "fc+conv"): 2,
        ("Oxford Flower", "fc"): 3,
        ("Oxford Flower", "conv"): 4,
        ("Oxford Flower", "fc+conv"): 5,
    }

    # Track which datasets are being evaluated in this run
    datasets_to_update = set()

    # Load existing data if available (to preserve results from previous runs)
    existing = read_table_csv(tables_dir, 1)
    if existing:
        existing_headers, existing_rows = existing
        # Determine which datasets will be updated based on command line args
        if args.cub_root and Path(args.cub_root).exists():
            datasets_to_update.add("CU-Bird200-2011")
        if args.flowers_root and Path(args.flowers_root).exists():
            datasets_to_update.add("Oxford Flower")

        # Merge: keep Paper values, preserve "Ours" values ONLY for datasets NOT being updated
        for i, row in enumerate(existing_rows):
            if i < len(rows):
                row_dataset = row[0]  # Dataset is in column 0 (from existing_rows)
                # Only preserve existing "Ours" values if this dataset is NOT being updated
                if row_dataset not in datasets_to_update:
                    # Row structure: Dataset, Model, then alternating Paper/Ours columns
                    # Preserve existing "Ours" values (indices 3, 5, 7, 9, 11, 13)
                    for j in [3, 5, 7, 9, 11, 13]:
                        if j < len(row) and row[j] != "—":
                            rows[i][j] = row[j]

    def _fmt_metric(m: dict, key: str) -> str:
        """Format a metric value (mean only; std is ignored for table display)."""
        val = m.get(key)
        if val is None:
            return "—"
        return _fmt(val)

    def fill_row(dataset_name: str, model_name: str, m: dict):
        idx = row_key.get((dataset_name, model_name))
        if idx is None:
            return
        rows[idx] = [
            dataset_name,
            model_name,
            rows[idx][2],  # ROC-AUC unseen (Paper) - keep original
            _fmt_metric(m, "roc_auc_unseen"),
            rows[idx][4],  # ROC-AUC seen (Paper) - keep original
            _fmt_metric(m, "roc_auc_seen"),
            rows[idx][6],  # ROC-AUC mean (Paper) - keep original
            _fmt_metric(m, "roc_auc_mean"),
            rows[idx][8],  # PR-AUC unseen (Paper) - keep original
            _fmt_metric(m, "pr_auc_unseen"),
            rows[idx][10], # PR-AUC seen (Paper) - keep original
            _fmt_metric(m, "pr_auc_seen"),
            rows[idx][12], # PR-AUC mean (Paper) - keep original
            _fmt_metric(m, "pr_auc_mean"),
        ]

    # CUB (birds dataset): we report as CU-Bird200-2011
    if args.cub_root and Path(args.cub_root).exists():
        jsonl_birds = code_root / args.wikipedia_birds
        if jsonl_birds.exists():
            try:
                n_unseen_cub = args.n_unseen if args.n_unseen is not None else 40
                out = prepare_birds_zero_shot(
                    args.cub_root, jsonl_birds,
                    n_unseen=n_unseen_cub,
                    unseen_seed=args.seed,
                    split_seed=args.seed,
                )
                (train_p, train_l, test_p, test_l, class_names, text_feat, seen_idx, unseen_idx) = out
                num_classes = len(class_names)
                text_t = torch.from_numpy(text_feat).float()
                loader = DataLoader(
                    ImageClassDataset(test_p, test_l),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                from tqdm import tqdm

                _cv_kw_cub = dict(
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
                _loss_keys_cub = {
                    "fc":     ["fc_bce_cub", "fc_hinge_cub", "fc_euclidean_cub"],
                    "conv":   ["conv_bce_cub", "conv_hinge_cub"],
                    "fc+conv": ["fc_conv_bce_cub", "fc_conv_hinge_cub", "fc_conv_euclidean_cub"],
                }
                for model_type in tqdm(["fc", "conv", "fc+conv"], desc="CUB models", unit="model"):
                    m = _best_across_losses(
                        loss_keys=_loss_keys_cub[model_type],
                        model_type=model_type,
                        loader=loader,
                        text_t=text_t,
                        device=device,
                        num_classes=num_classes,
                        seen_idx=seen_idx,
                        unseen_idx=unseen_idx,
                        model_kw=model_kw,
                        n_folds=args.n_folds,
                        checkpoint_dir=args.checkpoint_dir,
                        cv_kw=_cv_kw_cub,
                    )
                    if m:
                        fill_row("CU-Bird200-2011", model_type, m)
            except Exception as e:
                print(f"CUB evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    # Oxford Flowers
    if args.flowers_root and Path(args.flowers_root).exists():
        jsonl_flowers = code_root / args.wikipedia_flowers
        if jsonl_flowers.exists():
            try:
                n_unseen_flowers = args.n_unseen if args.n_unseen is not None else 20
                out = prepare_flowers_zero_shot(
                    args.flowers_root,
                    jsonl_flowers,
                    n_unseen=n_unseen_flowers,
                    unseen_seed=args.seed,
                    split_seed=args.seed,
                )
                (train_p, train_l, test_p, test_l, class_names, text_feat, seen_idx, unseen_idx) = out
                num_classes = len(class_names)
                text_t = torch.from_numpy(text_feat).float()
                loader = DataLoader(
                    ImageClassDataset(test_p, test_l),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                from tqdm import tqdm

                _cv_kw_flowers = dict(
                    dataset="flowers",
                    images_root=args.flowers_root,
                    wikipedia_jsonl=str(jsonl_flowers),
                    device=device,
                    batch_size=args.batch_size,
                    base_seed=args.seed,
                    n_unseen=n_unseen_flowers,
                    train_ratio=args.train_ratio,
                    **model_kw,
                )
                _loss_keys_flowers = {
                    "fc":     ["fc_bce_flowers", "fc_hinge_flowers", "fc_euclidean_flowers"],
                    "conv":   ["conv_bce_flowers", "conv_hinge_flowers"],
                    "fc+conv": ["fc_conv_bce_flowers", "fc_conv_hinge_flowers", "fc_conv_euclidean_flowers"],
                }
                for model_type in tqdm(["fc", "conv", "fc+conv"], desc="Flowers models", unit="model"):
                    m = _best_across_losses(
                        loss_keys=_loss_keys_flowers[model_type],
                        model_type=model_type,
                        loader=loader,
                        text_t=text_t,
                        device=device,
                        num_classes=num_classes,
                        seen_idx=seen_idx,
                        unseen_idx=unseen_idx,
                        model_kw=model_kw,
                        n_folds=args.n_folds,
                        checkpoint_dir=args.checkpoint_dir,
                        cv_kw=_cv_kw_flowers,
                    )
                    if m:
                        fill_row("Oxford Flower", model_type, m)
            except Exception as e:
                print(f"Flowers evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    path = write_table_csv(tables_dir, 1, headers, rows)
    print(f"Saved {path}")

    # Generate Table 1 LaTeX with style matching compile_all_tables.py
    # Uses \scriptsize font and multi-level headers with \cmidrule
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{ROC-AUC and PR-AUC performance on CUB-200-2011 and Oxford Flowers-102.}
\label{tab:table1}
\begin{tabular}{llcccccccccccc}
\toprule
Dataset & Model & \multicolumn{6}{c}{ROC-AUC} & \multicolumn{6}{c}{PR-AUC} \\
\cmidrule(lr){3-8} \cmidrule(lr){9-14}
 & & \multicolumn{2}{c}{unseen} & \multicolumn{2}{c}{seen} & \multicolumn{2}{c}{mean} & \multicolumn{2}{c}{unseen} & \multicolumn{2}{c}{seen} & \multicolumn{2}{c}{mean} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14}
 & & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours \\
\midrule
"""
    for row in rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}

"""
    # Write to file
    output_path = tex_dir / "Table1.tex"
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
