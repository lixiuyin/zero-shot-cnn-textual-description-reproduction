"""
Table 3. Performance comparison using different intermediate ConvLayers from VGG (fc+conv models).
Note: Paper Table 3 reports results on CUB-200-2010. Our 'Ours' column uses CUB-200-2011
(the only publicly available version, ~2x images). The 'Paper' column values are copied
from the original paper (Ba et al. ICCV 2015 Table 3).
Output: results/tables/Table3.csv, Table3.tex
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
from utils.config import FT_HIDDEN, GV_HIDDEN, K, TEXT_DIM, CONV_CHANNELS


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _fmt_metric(m: dict, key: str) -> str:
    val = m.get(key)
    if val is None:
        return "—"
    return _fmt(val)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 3 from model evaluation.")
    parser.add_argument("--cub_root", type=str, default="", help="Path to CUB_200_2011")
    parser.add_argument("--wikipedia_birds", type=str, default="data/wikipedia/birds.jsonl", help="Path to birds Wikipedia text")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Default dir for checkpoints (used when --checkpoint_* not set)")
    parser.add_argument("--checkpoint_conv4", type=str, default="", help="fc+conv with conv4_3")
    parser.add_argument("--checkpoint_conv5", type=str, default="", help="fc+conv with conv5_3")
    parser.add_argument("--checkpoint_pool5", type=str, default="", help="fc+conv with pool5")
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
    base_kw = dict(
        text_dim=TEXT_DIM,
        k=K,
        ft_hidden=FT_HIDDEN,
        gv_hidden=GV_HIDDEN,
        conv_channels=CONV_CHANNELS,
        image_backbone=args.image_backbone,
    )

    # Use LaTeX-safe headers in CSV so compile_all_tables.py (which reads CSV) produces valid .tex
    headers_csv = ["Metrics", r"Conv5\_3 (Paper)", r"Conv5\_3 (Ours)", r"Conv4\_3 (Paper)", r"Conv4\_3 (Ours)", "Pool5 (Paper)", "Pool5 (Ours)"]
    header_tex = headers_csv
    rows = [
        ["mean ROC-AUC", "0.91", "—", "0.6", "—", "0.82", "—"],
        ["mean PR-AUC", "0.28", "—", "0.09", "—", "0.173", "—"],
        ["mean top-5 acc.", "0.25", "—", "0.153", "—", "0.02", "—"],
    ]
    # column index: Ours (Conv5_3)=2, Ours (Conv4_3)=4, Ours (Pool5)=6
    # Tuple: (key, conv_feature_layer, col_j, explicit_ckpt)
    configs = [
        ("fc_conv_bce_cub_conv5_3", "conv5_3", 2, args.checkpoint_conv5),
        ("fc_conv_bce_cub_conv4_3", "conv4_3", 4, args.checkpoint_conv4),
        ("fc_conv_bce_cub_pool5",   "pool5",   6, args.checkpoint_pool5),
    ]

    # Track which conv layers will be updated
    layers_to_update = set()

    # Load existing data if available (to preserve results from previous runs)
    existing = read_table_csv(tables_dir, 3)
    if existing:
        existing_headers, existing_rows = existing
        # Determine which layers will be updated based on resolved checkpoints
        for key, conv_layer, _, explicit in configs:
            root, folds = resolve_with_cv(key, args.n_folds, args.checkpoint_dir, explicit)
            if root or len(folds) >= 2:
                layers_to_update.add(conv_layer)

        # Merge: keep Paper values, preserve "Ours" values ONLY for layers NOT being updated
        for i, row in enumerate(existing_rows):
            if i < len(rows):
                # We preserve "Ours" values if the layer is NOT being updated
                if "conv5_3" not in layers_to_update and 2 < len(row) and row[2] != "—":
                    rows[i][2] = row[2]  # Preserve Conv5_3 (Ours)
                if "conv4_3" not in layers_to_update and 4 < len(row) and row[4] != "—":
                    rows[i][4] = row[4]  # Preserve Conv4_3 (Ours)
                if "pool5" not in layers_to_update and 6 < len(row) and row[6] != "—":
                    rows[i][6] = row[6]  # Preserve Pool5 (Ours)

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

            for key, conv_layer, col_j, explicit in tqdm(configs, desc="Conv layers", unit="model"):
                root, fold_ckpts = resolve_with_cv(key, args.n_folds, args.checkpoint_dir, explicit)
                if len(fold_ckpts) >= 2:
                    print(f"  {conv_layer}: averaging {len(fold_ckpts)} CV folds (per-fold split)")
                    m = evaluate_cv_folds(
                        fold_ckpts,
                        "fc+conv",
                        dataset="cub",
                        images_root=args.cub_root,
                        wikipedia_jsonl=str(jsonl_birds),
                        device=device,
                        batch_size=args.batch_size,
                        base_seed=args.seed,
                        n_unseen=n_unseen_cub,
                        train_ratio=args.train_ratio,
                        conv_feature_layer=conv_layer,
                        **base_kw,
                    )
                else:
                    ckpt = root
                    if not ckpt:
                        continue
                    model = load_model(
                        "fc+conv", ckpt, device,
                        conv_feature_layer=conv_layer, **base_kw,
                    )
                    scores, labels = run_inference(model, loader, text_t, device, num_classes, desc=f"{conv_layer} inference")
                    m = compute_zero_shot_metrics(scores, labels, seen_idx, unseen_idx)
                rows[0][col_j] = _fmt_metric(m, "roc_auc_mean")
                rows[1][col_j] = _fmt_metric(m, "pr_auc_mean")
                rows[2][col_j] = _fmt_metric(m, "top5_mean")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    path = write_table_csv(tables_dir, 3, headers_csv, rows)
    print(f"Saved {path}")

    # Generate Table 3 LaTeX with style matching compile_all_tables.py
    content = r"""
\begin{table}[t]
\centering
\scriptsize
\caption{Performance comparison using different intermediate ConvLayers (fc+conv models). Paper values from CUB-200-2010; Ours from CUB-200-2011.}
\label{tab:table3}
\begin{tabular}{lrrrrrr}
\toprule
"""
    content += " & ".join(str(c) for c in header_tex) + r" \\ \midrule" + "\n"

    for row in rows:
        content += " & ".join(str(c) for c in row) + r" \\" + "\n"

    content += r""" \bottomrule
\end{tabular}
\end{table}

"""
    # Write to file
    output_path = tex_dir / "Table3.tex"
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
