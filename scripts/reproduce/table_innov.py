"""
TableInnov: Innovation ablation comparison table (Ba et al. ICCV 2015 extensions).

Sections (matching innovate.sh):
  A  -- Loss function ablation     (fc+conv, VGG-19, TF-IDF)
  B  -- Text encoder ablation      (fc+conv, VGG-19)
  C  -- Image backbone ablation    (fc+conv & fc, TF-IDF)

Rows per section table: PR-AUC seen / ROC-AUC seen / PR-AUC unseen / ROC-AUC unseen.
A0 (BCE baseline) is looked up in BOTH checkpoints/ and checkpoints/innov/.

Handles both single checkpoints and CV fold checkpoints automatically.

Output: results/tables/TableInnov.csv
        results/tex/TableInnov.tex

Usage:
    python scripts/reproduce/table_innov.py \\
        --cub_root data/images/birds \\
        [--innov_dir checkpoints/innov] \\
        [--n_folds 0]   # 0 = auto-detect from fold* dirs
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import prepare_birds_zero_shot, ImageClassDataset
from scripts.reproduce.common import get_tables_dir, get_tex_dir, write_table_csv
from scripts.reproduce.eval_utils import (
    compute_zero_shot_metrics,
    evaluate_cv_folds,
    load_model,
    run_inference,
)
from utils.config import K, FT_HIDDEN, GV_HIDDEN, CONV_CHANNELS, CONV_FEATURE_LAYER, _TEXT_ENCODER_DIMS


# ---------------------------------------------------------------------------
# Innovation registry
# (id, col_name, model_type, ckpt_name, text_encoder, image_backbone, fc_mode)
#
# Matches innovate.sh sections A/B/C exactly.
# A0 is the BCE baseline from train.sh (checkpoint lives in checkpoints/).
# ---------------------------------------------------------------------------
_INNOVATIONS = [
    # Section A – loss function ablation (fc+conv, VGG-19, TF-IDF)
    ("A0", "BCE (baseline)", "fc+conv", "fc_conv_bce",            "tfidf",       "vgg19",      "default"),
    ("A1", "CLIP-loss",      "fc+conv", "fc_conv_clip",           "tfidf",       "vgg19",      "default"),
    ("A2", "CenterAlign",    "fc+conv", "fc_conv_center_align",   "tfidf",       "vgg19",      "default"),
    ("A3", "EmbMSE",         "fc+conv", "fc_conv_embedding_mse",  "tfidf",       "vgg19",      "default"),
    # Section B – text encoder ablation (fc+conv, VGG-19)
    ("B1", "SBERT",          "fc+conv", "fc_conv_sbert",          "sbert",       "vgg19",      "default"),
    ("B2", "SBERT-multi",    "fc+conv", "fc_conv_sbert_multi",    "sbert_multi", "vgg19",      "default"),
    ("B3", "CLIP-text",      "fc+conv", "fc_conv_clip_text",      "clip",        "vgg19",      "default"),
    ("B4", "CLIP-multi",     "fc+conv", "fc_conv_clip_multi",     "clip_multi",  "vgg19",      "default"),
    # Section C – image backbone ablation (fc+conv & fc, TF-IDF)
    ("C1", "DenseNet-121",          "fc+conv", "fc_conv_densenet121",        "tfidf", "densenet121", "default"),
    ("C2", "ResNet-50",             "fc+conv", "fc_conv_resnet50",           "tfidf", "resnet50",    "default"),
    ("C3", "DenseNet-121 (penult)", "fc+conv", "fc_conv_densenet121_penult", "tfidf", "densenet121", "penultimate"),
    ("C4", "ResNet-50 (penult)",    "fc+conv", "fc_conv_resnet50_penult",    "tfidf", "resnet50",    "penultimate"),
    ("C5", "DenseNet-121 fc-only",  "fc",      "fc_densenet121_penult",      "tfidf", "densenet121", "penultimate"),
    ("C6", "ResNet-50 fc-only",     "fc",      "fc_resnet50_penult",         "tfidf", "resnet50",    "penultimate"),
]

_METRIC_KEYS = [
    ("pr_auc_seen",    "PR-AUC seen"),
    ("roc_auc_seen",   "ROC-AUC seen"),
    ("pr_auc_unseen",  "PR-AUC unseen"),
    ("roc_auc_unseen", "ROC-AUC unseen"),
]

_TEX_METRICS_ORDER = [
    ("pr_auc_seen", "PR-AUC"),
    ("roc_auc_seen", "ROC-AUC"),
    ("pr_auc_unseen", "PR-AUC"),
    ("roc_auc_unseen", "ROC-AUC"),
]


def _find_checkpoints(
    innov_dir: Path,
    ckpt_name: str,
    extra_dirs: list[Path] | None = None,
) -> tuple[list[str], str]:
    """Return (paths, mode) where mode is 'single', 'cv', or 'missing'.

    Searches innov_dir first, then any extra_dirs (for baseline checkpoints).
    """
    for search_dir in [innov_dir] + (extra_dirs or []):
        if not search_dir.exists():
            continue

        # Single checkpoint
        single = search_dir / f"{ckpt_name}.pt"
        if single.exists():
            return [str(single)], "single"

        # CV fold checkpoints
        fold_ckpts = sorted(
            search_dir.glob(f"fold*/{ckpt_name}.pt"),
            key=lambda p: p.parent.name,
        )
        if fold_ckpts:
            return [str(p) for p in fold_ckpts], "cv"

    return [], "missing"


def _fmt(val, std=None) -> str:
    if val is None:
        return "—"
    return f"{val:.3f}"


def _parse_float(cell: str) -> float | None:
    """Parse numeric value from a table cell (supports '0.123' or '0.123±0.004')."""
    if not cell or cell == "—":
        return None
    head = cell.split("±", 1)[0].strip()
    try:
        return float(head)
    except ValueError:
        return None


def _bold_best_by_column(rows: list[list[str]], value_col_indices: list[int]) -> list[list[str]]:
    """Bold best (max) value per metric column within a section."""
    best: dict[int, float] = {}
    for j in value_col_indices:
        vals = [_parse_float(r[j]) for r in rows]
        vals2 = [v for v in vals if v is not None]
        if vals2:
            best[j] = max(vals2)

    out = [r[:] for r in rows]
    for i, r in enumerate(out):
        for j in value_col_indices:
            v = _parse_float(r[j])
            if v is None:
                continue
            if j in best and f"{v:.3f}" == f"{best[j]:.3f}":
                out[i][j] = r"\bfseries " + r[j]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate innovation ablation table (TableInnov).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cub_root", default="", help="CUB-200-2011 images root")
    parser.add_argument("--wikipedia_birds", default="data/wikipedia/birds.jsonl")
    parser.add_argument("--innov_dir", default="checkpoints/innov",
                        help="Directory containing innovation checkpoints")
    parser.add_argument("--n_folds", type=int, default=0,
                        help="Expected CV folds (0 = auto-detect from fold* dirs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_unseen", type=int, default=40)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--conv_feature_layer", type=str, default=CONV_FEATURE_LAYER,
                        choices=("conv5_3", "conv4_3", "pool5"),
                        help="VGG conv feature layer used during training (default: conv5_3)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=None,
                        help="Override output directory (default: results/)")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    innov_dir = Path(args.innov_dir)
    if not innov_dir.is_absolute():
        innov_dir = code_root / innov_dir

    # A0 baseline lives in the main checkpoints/ dir (from train.sh)
    main_ckpt_dir = code_root / "checkpoints"

    tables_dir = get_tables_dir()
    tex_dir = get_tex_dir()
    if args.out_dir:
        tables_dir = Path(args.out_dir) / "tables"
        tex_dir = Path(args.out_dir) / "tex"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tex_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Prepare CUB dataset (single split for single-checkpoint eval)
    jsonl_birds = code_root / args.wikipedia_birds
    cub_ok = args.cub_root and Path(args.cub_root).exists() and jsonl_birds.exists()
    if not cub_ok:
        print(f"[WARN] CUB data not found at '{args.cub_root}'. "
              "Pass --cub_root to enable evaluation.")

    # Pre-load dataset for tfidf single-checkpoint evaluation
    single_data = None
    text_feat_tfidf = None
    if cub_ok:
        out = prepare_birds_zero_shot(
            args.cub_root, str(jsonl_birds),
            n_unseen=args.n_unseen,
            unseen_seed=args.seed,
            split_seed=args.seed,
            train_ratio_seen=args.train_ratio,
        )
        _, _, test_p, test_l, _, text_feat_tfidf, seen_idx, unseen_idx = out
        single_data = dict(
            test_p=test_p, test_l=test_l,
            seen_idx=seen_idx, unseen_idx=unseen_idx,
            num_classes=len(seen_idx) + len(unseen_idx),
        )

    # -----------------------------------------------------------------
    # Evaluate each innovation and collect metrics
    # -----------------------------------------------------------------
    results: dict[str, dict] = {}

    for innov_id, col_name, model_type, ckpt_name, text_encoder, image_backbone, fc_mode in _INNOVATIONS:
        print(f"\n[{innov_id}] {col_name}  (ckpt={ckpt_name}, model={model_type}, "
              f"text={text_encoder}, backbone={image_backbone}, fc_mode={fc_mode})")

        # Search innov_dir, plus main checkpoints/ for baseline (A0)
        extra_dirs = [main_ckpt_dir] if innov_id == "A0" else []
        ckpt_paths, mode = _find_checkpoints(innov_dir, ckpt_name, extra_dirs)

        if mode == "missing":
            print(f"  → checkpoint not found, skipping")
            results[ckpt_name] = {}
            continue

        if not cub_ok:
            results[ckpt_name] = {}
            continue

        text_dim = _TEXT_ENCODER_DIMS[text_encoder]
        model_kw = dict(
            text_dim=text_dim,
            k=K,
            ft_hidden=FT_HIDDEN,
            gv_hidden=GV_HIDDEN,
            conv_channels=CONV_CHANNELS,
            conv_feature_layer=args.conv_feature_layer,
            image_backbone=image_backbone,
            fc_mode=fc_mode,
        )

        try:
            if mode == "cv" and len(ckpt_paths) >= 2:
                print(f"  → CV evaluation ({len(ckpt_paths)} folds)")
                m = evaluate_cv_folds(
                    ckpt_paths,
                    model_type=model_type,
                    dataset="cub",
                    images_root=args.cub_root,
                    wikipedia_jsonl=str(jsonl_birds),
                    device=device,
                    batch_size=args.batch_size,
                    base_seed=args.seed,
                    n_unseen=args.n_unseen,
                    train_ratio=args.train_ratio,
                    text_encoder=text_encoder,
                    **model_kw,
                )
            else:
                # Single checkpoint evaluation
                print(f"  → single checkpoint: {ckpt_paths[0]}")
                # For non-tfidf encoders, text features need to be re-prepared
                if text_encoder != "tfidf":
                    out = prepare_birds_zero_shot(
                        args.cub_root, str(jsonl_birds),
                        n_unseen=args.n_unseen,
                        unseen_seed=args.seed,
                        split_seed=args.seed,
                        train_ratio_seen=args.train_ratio,
                        text_encoder=text_encoder,
                    )
                    _, _, tp, tl, _, tf, si, ui = out
                    text_t = torch.from_numpy(tf).float()
                    nc = len(si) + len(ui)
                    loader = DataLoader(
                        ImageClassDataset(tp, tl),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                    )
                    seen_idx_eval, unseen_idx_eval = si, ui
                else:
                    text_t = torch.from_numpy(text_feat_tfidf).float()
                    nc = single_data["num_classes"]
                    loader = DataLoader(
                        ImageClassDataset(single_data["test_p"], single_data["test_l"]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                    )
                    seen_idx_eval = single_data["seen_idx"]
                    unseen_idx_eval = single_data["unseen_idx"]

                model = load_model(model_type, ckpt_paths[0], device, **model_kw)
                scores, labels = run_inference(
                    model, loader, text_t, device, nc, desc=f"{innov_id} inference"
                )
                m = compute_zero_shot_metrics(scores, labels, seen_idx_eval, unseen_idx_eval)

            results[ckpt_name] = m
            pr_u = _fmt(m.get("pr_auc_unseen"))
            roc_u = _fmt(m.get("roc_auc_unseen"))
            print(f"  PR-AUC unseen={pr_u}  ROC-AUC unseen={roc_u}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            results[ckpt_name] = {}

    # -----------------------------------------------------------------
    # Build CSV table: rows = metrics, columns = innovations
    # -----------------------------------------------------------------
    col_headers = [f"{iid} {cname}" for iid, cname, *_ in _INNOVATIONS]
    headers = ["Metric"] + col_headers

    rows = []
    for metric_key, metric_label in _METRIC_KEYS:
        row = [metric_label]
        for _, _, _, ckpt_name, *_ in _INNOVATIONS:
            m = results.get(ckpt_name, {})
            val = m.get(metric_key)
            row.append(_fmt(val))
        rows.append(row)

    csv_path = tables_dir / "TableInnov.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    # -----------------------------------------------------------------
    # Build LaTeX: three separate tables (Loss / Text / Backbone)
    # -----------------------------------------------------------------
    # Helper: get A0 baseline row for comparison
    a0_m = results.get("fc_conv_bce", {})
    def _baseline_row(label: str) -> str:
        vals = [_fmt(a0_m.get(k)) for k, _ in _TEX_METRICS_ORDER]
        return f"{label} & " + " & ".join(vals) + r" \\"

    def _section_rows(prefix: str) -> list[list[str]]:
        section = [
            (iid, cname, ckpt)
            for (iid, cname, _, ckpt, *_) in _INNOVATIONS
            if iid.startswith(prefix)
        ]
        out_rows: list[list[str]] = []
        for iid, cname, ckpt_name in section:
            m = results.get(ckpt_name, {})
            vals = [_fmt(m.get(k)) for k, _ in _TEX_METRICS_ORDER]
            out_rows.append([cname] + vals)
        return _bold_best_by_column(out_rows, [1, 2, 3, 4])

    def _make_table(
        section_prefix: str,
        caption: str,
        label: str,
        first_col_header: str,
        baseline_label: str | None = None,
        skip_ids: set[str] | None = None,
    ) -> list[str]:
        """Generate LaTeX lines for one section table."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{threeparttable}",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{label}}}",
            r"\begin{tabular}{l S S S S}",
            r"\toprule",
            r" & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
            rf"{first_col_header} & {{PR-AUC}} & {{ROC-AUC}} & {{PR-AUC}} & {{ROC-AUC}} \\",
            r"\midrule",
        ]
        section = _section_rows(section_prefix)
        for i, (iid, *_) in enumerate(
            (iid, cname) for iid, cname, *_ in _INNOVATIONS if iid.startswith(section_prefix)
        ):
            if skip_ids and iid in skip_ids:
                continue
            if i < len(section):
                lines.append(" & ".join(section[i]) + r" \\")
        if baseline_label:
            lines.append(r"\midrule")
            lines.append(_baseline_row(baseline_label))
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{threeparttable}",
            r"\end{table}",
            r"",
        ]
        return lines

    latex_lines = [
        r"\documentclass{article}",
        r"",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{siunitx}",
        r"\usepackage{threeparttable}",
        r"",
        r"\sisetup{",
        r"  detect-weight=true,",
        r"  detect-inline-weight=math,",
        r"  group-digits=false,",
        r"  input-symbols = {—},",
        r"  table-format=1.3,",
        r"}",
        r"",
        r"\begin{document}",
        r"",
    ]

    # Section A: Loss ablation (skip A0 from data rows, show as baseline)
    latex_lines += [r"% ===================== Table: Loss ====================="]
    latex_lines += _make_table(
        "A",
        caption="Ablation of loss functions on CUB-200-2011 (fc+conv, VGG-19, TF-IDF)",
        label="innov_loss",
        first_col_header="Loss",
        baseline_label="BCE (baseline)",
        skip_ids={"A0"},
    )

    # Section B: Text encoder ablation
    latex_lines += [r"% ===================== Table: Text Encoder ====================="]
    latex_lines += _make_table(
        "B",
        caption="Ablation of text encoders on CUB-200-2011 (fc+conv, VGG-19)",
        label="innov_text",
        first_col_header="Text encoder",
        baseline_label="TF-IDF (baseline)",
    )

    # Section C: Backbone ablation
    latex_lines += [r"% ===================== Table: Backbone ====================="]
    latex_lines += _make_table(
        "C",
        caption="Ablation of image backbones on CUB-200-2011 (TF-IDF)",
        label="innov_backbone",
        first_col_header="Backbone",
        baseline_label="VGG-19 fc+conv (baseline)",
    )

    latex_lines += [
        r"\end{document}",
        r"",
    ]

    tex_path = tex_dir / "TableInnov.tex"
    tex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
