"""
Evaluation: ROC-AUC, PR-AUC (AP), Top-1 / Top-5 accuracy (Ba et al. ICCV 2015 Sec 5).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ZeroShotDataset
from models.zero_shot_model import ZeroShotModel
from utils.config import (
    MODEL_TYPE,
    K,
    FT_HIDDEN,
    CONV_CHANNELS,
    CONV_FEATURE_LAYER,
    GV_HIDDEN,
    NUM_WORKERS,
    CUB_UNSEEN,
    FLOWER_UNSEEN,
    TEXT_ENCODER,
    IMAGE_BACKBONE,
    _TEXT_ENCODER_DIMS,
)
from utils.seed_utils import set_seed


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """
    scores: [N, C] (logits), labels: [N] in [0, C-1].
    For each class c, binary labels 1 if true class else 0; then ROC-AUC and PR-AUC per class, then mean.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=int)
    n = scores.shape[0]
    roc_aucs, pr_aucs = [], []
    for c in range(num_classes):
        y_true = (labels == c).astype(np.float64)
        if y_true.sum() == 0 or y_true.sum() == n:
            continue
        y_score = scores[:, c]
        try:
            roc_aucs.append(roc_auc_score(y_true, y_score))
        except ValueError as e:
            # Single class or all same labels - skip this class
            logger.debug(f"Skipping ROC-AUC for class {c}: {e}")
        try:
            pr_aucs.append(average_precision_score(y_true, y_score))
        except ValueError as e:
            # Single class or all same labels - skip this class
            logger.debug(f"Skipping PR-AUC for class {c}: {e}")
    return {
        "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else 0.0,
        "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else 0.0,
    }


def topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """scores [N, C], labels [N]. Top-k correct if label in top-k predicted."""
    pred = np.argsort(-scores, axis=1)[:, :k]  # [N, k]
    return (pred == labels.reshape(-1, 1)).any(axis=1).mean()


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot CNN on test splits.")
    parser.add_argument("--checkpoint", type=str, default="", help="Model checkpoint path")
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE, choices=("fc", "conv", "fc+conv"))
    parser.add_argument("--k", type=int, default=K,
                        help="Joint embedding dimension (must match training --k)")
    parser.add_argument("--ft_hidden", type=int, default=FT_HIDDEN,
                        help="Text encoder hidden dim (must match training --ft_hidden)")
    parser.add_argument("--conv_feature_layer", type=str, default=CONV_FEATURE_LAYER, choices=("conv5_3", "conv4_3", "pool5"))
    parser.add_argument("--gv_hidden", type=int, default=GV_HIDDEN)
    parser.add_argument("--dataset", type=str, default="cub", choices=("cub", "flowers"),
                        help="Dataset to use (CUB-200-2011 or Oxford Flowers-102)")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory containing images/ and wikipedia/ subfolders")
    parser.add_argument("--n_unseen", type=int, default=None,
                        help="Number of unseen classes (default: 40 for CUB, 20 for Flowers)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train ratio for seen classes (must match training)")
    parser.add_argument("--wikipedia_jsonl", type=str, default="",
                        help="Optional explicit path to Wikipedia JSONL (overrides default)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text_encoder", default=TEXT_ENCODER,
                        choices=("tfidf", "sbert", "sbert_multi", "clip", "clip_multi"),
                        help="Text feature extractor — must match training setting")
    parser.add_argument("--image_backbone", default=IMAGE_BACKBONE,
                        choices=("vgg19", "densenet121", "resnet50"),
                        help="Image backbone — must match training setting")
    parser.add_argument("--fc_mode", default="default",
                        choices=("default", "penultimate"),
                        help="FC branch mode — must match training setting")
    args = parser.parse_args()
    set_seed(42)  # Reproducible evaluation (same data split and any RNG in model)
    device = torch.device(args.device)

    code_root = Path(__file__).resolve().parents[1]
    data_root = (code_root / args.data_root).resolve()

    if args.n_unseen is None:
        if args.dataset == "cub":
            n_unseen = CUB_UNSEEN
        else:
            n_unseen = FLOWER_UNSEEN
    else:
        n_unseen = args.n_unseen

    if args.wikipedia_jsonl:
        wikipedia_jsonl = Path(args.wikipedia_jsonl)
    else:
        if args.dataset == "cub":
            wikipedia_jsonl = data_root / "wikipedia" / "birds.jsonl"
        else:
            wikipedia_jsonl = data_root / "wikipedia" / "flowers.jsonl"

    # Datasets and loaders for seen/unseen test splits
    test_seen_dataset = ZeroShotDataset(
        jsonl_path=wikipedia_jsonl,
        images_base=data_root / "images",
        mode="test_seen",
        n_unseen=n_unseen,
        train_ratio=args.train_ratio,
        seed=42,
        text_encoder=args.text_encoder,
    )
    test_unseen_dataset = ZeroShotDataset(
        jsonl_path=wikipedia_jsonl,
        images_base=data_root / "images",
        mode="test_unseen",
        n_unseen=n_unseen,
        train_ratio=args.train_ratio,
        seed=42,
        text_encoder=args.text_encoder,
    )

    test_seen_loader = DataLoader(
        test_seen_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_unseen_loader = DataLoader(
        test_unseen_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Prepare subset text features for seen/unseen evaluation (CRITICAL for zero-shot)
    seen_classes = test_seen_dataset.seen_classes
    unseen_classes = test_seen_dataset.unseen_classes
    seen_indices = [test_seen_dataset.label_to_idx[c] for c in seen_classes]
    unseen_indices = [test_seen_dataset.label_to_idx[c] for c in unseen_classes]

    # Create label mapping tensors (global label -> subset label)
    num_classes = len(test_seen_dataset.label_to_idx)
    seen_label_map = {test_seen_dataset.label_to_idx[c]: i for i, c in enumerate(seen_classes)}
    unseen_label_map = {test_seen_dataset.label_to_idx[c]: i for i, c in enumerate(unseen_classes)}

    seen_label_map_tensor = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for global_idx, subset_idx in seen_label_map.items():
        seen_label_map_tensor[global_idx] = subset_idx

    unseen_label_map_tensor = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for global_idx, subset_idx in unseen_label_map.items():
        unseen_label_map_tensor[global_idx] = subset_idx

    # Subset text features for each split (CRITICAL: seen uses [160], unseen uses [40])
    text_features_all = torch.from_numpy(test_seen_dataset.text_features).float().to(device)
    actual_text_dim = text_features_all.shape[1]

    # Get expected text dimension from config based on text_encoder choice
    expected_text_dim = _TEXT_ENCODER_DIMS[args.text_encoder]

    # Validate: actual dimension should match expected dimension
    if actual_text_dim != expected_text_dim:
        raise ValueError(
            f"[TEXT DIM MISMATCH] Text encoder '{args.text_encoder}' should produce "
            f"{expected_text_dim}-d features, but dataset has {actual_text_dim}-d. "
            f"Ensure evaluation uses the same text_encoder as training."
        )

    text_features_seen = text_features_all[seen_indices]    # [C_seen, text_dim]
    text_features_unseen = text_features_all[unseen_indices]  # [C_unseen, text_dim]

    model = ZeroShotModel(
        text_input_dim=expected_text_dim,
        k=args.k,
        ft_hidden=args.ft_hidden,
        conv_channels=CONV_CHANNELS,
        gv_hidden=args.gv_hidden,
        conv_feature_layer=args.conv_feature_layer,
        image_backbone=args.image_backbone,
        model_type=args.model_type,
        fc_mode=args.fc_mode,
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    model.eval()

    def _run_split(loader, split_name: str, eval_text_features: torch.Tensor, label_map_tensor: torch.Tensor):
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                scores = model(images, eval_text_features)  # [B, C_subset]
                # Remap global labels to subset indices for correct accuracy
                remapped_labels = label_map_tensor[labels].cpu().numpy()
                all_scores.append(scores.cpu().numpy())
                all_labels.append(remapped_labels)
        if not all_scores:
            print(f"No samples for split {split_name}.")
            return
        scores_np = np.concatenate(all_scores, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        num_subset_classes = eval_text_features.shape[0]
        metrics = compute_metrics(scores_np, labels_np, num_subset_classes)
        top1 = topk_accuracy(scores_np, labels_np, 1)
        top5 = topk_accuracy(scores_np, labels_np, 5)
        print(f"[{split_name}] ROC-AUC (mean): {metrics['roc_auc_mean']:.4f}")
        print(f"[{split_name}] PR-AUC (mean): {metrics['pr_auc_mean']:.4f}")
        print(f"[{split_name}] Top-1 acc: {top1:.4f}")
        print(f"[{split_name}] Top-5 acc: {top5:.4f}")

    _run_split(test_seen_loader, "test_seen", text_features_seen, seen_label_map_tensor)
    _run_split(test_unseen_loader, "test_unseen", text_features_unseen, unseen_label_map_tensor)


if __name__ == "__main__":
    main()
