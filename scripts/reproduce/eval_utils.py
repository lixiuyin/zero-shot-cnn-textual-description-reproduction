"""
Evaluation helpers for reproduction scripts: run model on data and compute paper metrics.
All results are computed from the model, not copied from the paper.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

# Configure logger for this module
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_model(
    model_type: str,
    checkpoint_path: str | Path | None,
    device: torch.device,
    text_dim: int = 9763,
    k: int = 50,
    ft_hidden: int = 300,
    gv_hidden: int = 300,
    conv_channels: int = 5,
    conv_feature_layer: str = "conv5_3",
    image_backbone: str = "vgg19",
    fc_mode: str = "default",
):
    from models.zero_shot_model import ZeroShotModel

    # Load checkpoint and extract config metadata if available
    state_dict = None
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            # Use saved config, warn if caller-provided values differ
            cfg = ckpt.get("config", {})
            _cfg_map = {
                "text_dim": text_dim, "k": k, "ft_hidden": ft_hidden,
                "gv_hidden": gv_hidden, "conv_channels": conv_channels,
                "conv_feature_layer": conv_feature_layer,
                "image_backbone": image_backbone,
                "fc_mode": fc_mode,
            }
            for key, caller_val in _cfg_map.items():
                saved_val = cfg.get(key)
                if saved_val is not None and saved_val != caller_val:
                    logger.warning(
                        f"[CONFIG OVERRIDE] checkpoint has {key}={saved_val!r}, "
                        f"caller passed {caller_val!r}; using checkpoint value"
                    )
            text_dim = cfg.get("text_dim", text_dim)
            k = cfg.get("k", k)
            ft_hidden = cfg.get("ft_hidden", ft_hidden)
            gv_hidden = cfg.get("gv_hidden", gv_hidden)
            conv_channels = cfg.get("conv_channels", conv_channels)
            conv_feature_layer = cfg.get("conv_feature_layer", conv_feature_layer)
            image_backbone = cfg.get("image_backbone", image_backbone)
            fc_mode = cfg.get("fc_mode", fc_mode)
        else:
            # Legacy checkpoint: bare state_dict (no metadata)
            state_dict = ckpt

    model = ZeroShotModel(
        text_input_dim=text_dim,
        k=k,
        ft_hidden=ft_hidden,
        gv_hidden=gv_hidden,
        conv_channels=conv_channels,
        conv_feature_layer=conv_feature_layer,
        image_backbone=image_backbone,
        model_type=model_type,
        fc_mode=fc_mode,
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(
    model,
    loader: DataLoader,
    text_features: torch.Tensor,
    device: torch.device,
    num_classes: int,
    desc: str = "Evaluating",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores [N, C], labels [N])."""
    from tqdm import tqdm

    model.eval()
    text_features = text_features.to(device)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            scores = model(images, text_features)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return scores, labels


def _metrics_over_classes(
    scores: np.ndarray,
    labels: np.ndarray,
    class_indices: list[int],
) -> tuple[float, float, float, float]:
    """ROC-AUC mean, PR-AUC mean, top-1 acc, top-5 acc over the given classes only."""
    if not class_indices:
        return 0.0, 0.0, 0.0, 0.0
    n = scores.shape[0]
    subset = np.isin(labels, class_indices)
    if subset.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    lab_sub = labels[subset]
    sco_sub = scores[subset][:, class_indices]
    # Optimized label remapping: O(N) instead of O(N²)
    # Create mapping dict and use vectorized lookup
    label_to_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
    lab_local = np.array([label_to_idx[lbl] for lbl in lab_sub])
    roc_aucs, pr_aucs = [], []
    for i, c in enumerate(class_indices):
        y_true = (lab_sub == c).astype(np.float64)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        y_score = sco_sub[:, i]
        try:
            roc_aucs.append(roc_auc_score(y_true, y_score))
            pr_aucs.append(average_precision_score(y_true, y_score))
        except ValueError as e:
            # Single class or all same labels - skip this class
            logger.debug(f"Skipping metrics for class {c}: {e}")
    roc_mean = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    pr_mean = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    pred_top1 = np.argmax(sco_sub, axis=1)
    top5_k = min(5, sco_sub.shape[1])
    pred_top5 = np.argsort(-sco_sub, axis=1)[:, :top5_k]
    top1 = (pred_top1 == lab_local).mean()
    top5 = (pred_top5 == lab_local.reshape(-1, 1)).any(axis=1).mean()
    return roc_mean, pr_mean, float(top1), float(top5)


def compute_zero_shot_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    seen_class_idx: list[int],
    unseen_class_idx: list[int],
) -> dict[str, float]:
    """
    Returns dict with roc_auc_unseen, roc_auc_seen, roc_auc_mean, pr_auc_unseen, pr_auc_seen, pr_auc_mean,
    top1_unseen, top1_seen, top5_unseen, top5_seen, etc.
    """
    roc_u, pr_u, t1_u, t5_u = _metrics_over_classes(scores, labels, unseen_class_idx)
    roc_s, pr_s, t1_s, t5_s = _metrics_over_classes(scores, labels, seen_class_idx)
    # Paper Table 1: "mean" is class-count-weighted, not sample-count-weighted.
    # Verified: CUB fc → (0.82×40 + 0.974×160)/200 = 0.943; (0.11×40 + 0.33×160)/200 = 0.286
    n_uc = len(unseen_class_idx)
    n_sc = len(seen_class_idx)
    n_total = n_uc + n_sc
    roc_mean = (roc_u * n_uc + roc_s * n_sc) / n_total if n_total > 0 else 0.0
    pr_mean = (pr_u * n_uc + pr_s * n_sc) / n_total if n_total > 0 else 0.0
    t1_mean = (t1_u * n_uc + t1_s * n_sc) / n_total if n_total > 0 else 0.0
    t5_mean = (t5_u * n_uc + t5_s * n_sc) / n_total if n_total > 0 else 0.0
    return {
        "roc_auc_unseen": roc_u,
        "roc_auc_seen": roc_s,
        "roc_auc_mean": roc_mean,
        "pr_auc_unseen": pr_u,
        "pr_auc_seen": pr_s,
        "pr_auc_mean": pr_mean,
        "top1_unseen": t1_u,
        "top1_seen": t1_s,
        "top1_mean": t1_mean,
        "top5_unseen": t5_u,
        "top5_seen": t5_s,
        "top5_mean": t5_mean,
    }


def evaluate_cv_folds(
    fold_checkpoints: list[str],
    model_type: str,
    dataset: str,
    images_root: str,
    wikipedia_jsonl: str,
    device: torch.device,
    batch_size: int = 64,
    base_seed: int = 42,
    n_unseen: int | None = None,
    train_ratio: float = 0.8,
    text_encoder: str = "tfidf",
    **model_kw,
) -> dict[str, float]:
    """Evaluate each fold checkpoint on its own data split and return mean ± std.

    fold_seed is derived from the directory name: ``fold0`` → ``base_seed + 0``,
    ``fold1`` → ``base_seed + 1``, etc.  No metadata files are read or written.

    Note: Oxford Flowers has no seed parameter — all folds share the same class
    split, so CV only captures model variance across restarts.

    Returns a dict where every metric key ``k`` from ``compute_zero_shot_metrics``
    appears twice: ``k`` (mean across folds) and ``k + "_std"`` (std).
    Checkpoints with empty paths are silently skipped.
    """
    import re
    from pathlib import Path as _Path
    from torch.utils.data import DataLoader as _DataLoader

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from data import ImageClassDataset, prepare_birds_zero_shot, prepare_flowers_zero_shot
    from utils.config import _TEXT_ENCODER_DIMS

    # Validate text_dim matches text_encoder
    expected_text_dim = _TEXT_ENCODER_DIMS[text_encoder]
    provided_text_dim = model_kw.get("text_dim", 9763)
    if provided_text_dim != expected_text_dim:
        logger.warning(
            f"[TEXT DIM MISMATCH] text_encoder='{text_encoder}' expects {expected_text_dim}-d, "
            f"but model_kw has text_dim={provided_text_dim}. Overriding to {expected_text_dim}-d."
        )
        model_kw = dict(model_kw, text_dim=expected_text_dim)

    valid_ckpts = [p for p in fold_checkpoints if p]
    all_metrics: list[dict[str, float]] = []

    for ckpt in valid_ckpts:
        # ── 1. Derive fold_seed from directory name (fold0 → base_seed+0) ─
        dir_name = _Path(ckpt).parent.name
        m_idx = re.search(r"fold(\d+)$", dir_name)
        if m_idx is None:
            raise ValueError(
                f"Checkpoint directory '{dir_name}' does not match expected pattern 'fold<N>'. "
                f"Ensure checkpoints were saved with n_folds > 1 (creates fold0/, fold1/, ...)."
            )
        fold_idx = int(m_idx.group(1))
        fold_seed = base_seed + fold_idx

        # ── 2. Reconstruct the exact split used during training ───────────
        # Both CUB and Flowers use seed-based random splits: fold_seed controls
        # the unseen/seen class split and the train/test image split within seen classes.
        # Each fold gets a different fold_seed, so CV averages over different class splits.
        if dataset == "cub":
            out = prepare_birds_zero_shot(
                images_root, wikipedia_jsonl,
                n_unseen=n_unseen if n_unseen is not None else 40,
                train_ratio_seen=train_ratio,
                unseen_seed=fold_seed,
                split_seed=fold_seed,
                text_encoder=text_encoder,
            )
        else:
            out = prepare_flowers_zero_shot(
                images_root, wikipedia_jsonl,
                n_unseen=n_unseen if n_unseen is not None else 20,
                train_ratio_seen=train_ratio,
                unseen_seed=fold_seed,
                split_seed=fold_seed,
                text_encoder=text_encoder,
            )

        _, _, test_p, test_l, _, text_feat, seen_idx, unseen_idx = out
        text_t = torch.from_numpy(text_feat).float()
        num_classes = len(seen_idx) + len(unseen_idx)

        loader = _DataLoader(
            ImageClassDataset(test_p, test_l),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        # ── 3. Evaluate ───────────────────────────────────────────────────
        model = load_model(model_type, ckpt, device, **model_kw)
        scores, labels = run_inference(
            model, loader, text_t, device, num_classes,
            desc=f"fold {fold_idx} ({len(all_metrics)+1}/{len(valid_ckpts)})",
        )
        fold_m = compute_zero_shot_metrics(scores, labels, seen_idx, unseen_idx)
        all_metrics.append(fold_m)
        logger.info(
            f"  fold {fold_idx} (seed={fold_seed}): "
            f"unseen_roc={fold_m['roc_auc_unseen']:.3f}  seen_roc={fold_m['roc_auc_seen']:.3f}"
        )

    if not all_metrics:
        return {}

    result: dict[str, float] = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        result[k] = float(np.mean(vals))
        result[k + "_std"] = float(np.std(vals))
    return result


def fmt_cv(mean: float, std: float) -> str:
    """Format a CV metric as ``mean±std`` (3 decimal places)."""
    return f"{mean:.3f}±{std:.3f}"


def compute_mean_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """When there is no seen/unseen split (e.g. 50/50), just mean ROC-AUC, PR-AUC, top-1, top-5."""
    roc_aucs, pr_aucs = [], []
    for c in range(num_classes):
        y_true = (labels == c).astype(np.float64)
        if y_true.sum() == 0 or y_true.sum() == len(labels):
            continue
        try:
            roc_aucs.append(roc_auc_score(y_true, scores[:, c]))
            pr_aucs.append(average_precision_score(y_true, scores[:, c]))
        except ValueError as e:
            # Single class or all same labels - skip this class
            logger.debug(f"Skipping metrics for class {c}: {e}")
    roc_mean = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    pr_mean = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    pred = np.argmax(scores, axis=1)
    top1 = (pred == labels).mean()
    top5_k = min(5, scores.shape[1])
    top5 = (np.argsort(-scores, axis=1)[:, :top5_k] == labels.reshape(-1, 1)).any(axis=1).mean()
    return {
        "roc_auc_mean": roc_mean,
        "pr_auc_mean": pr_mean,
        "top1_mean": float(top1),
        "top5_mean": float(top5),
    }
