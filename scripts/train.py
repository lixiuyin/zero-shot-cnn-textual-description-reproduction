"""
Training loop — Ba et al. ICCV 2015 zero-shot CNN.

Paper Sec 4 minibatch loss: sum only over classes present in the batch (U ≤ B),
cost O(B×U). Supports BCE (Eq. 6), Hinge (Eq. 7), Euclidean (Sec 4.2.1).
Euclidean = Hinge + L2 on embeddings (Sec 4.2.1); L2 penalty added for fc/fc+conv models.

Extensions (non-paper, via CLI flags):
  --text_encoder    sbert | sbert_multi | clip   (default: tfidf)
  --image_backbone  densenet121 | resnet50        (default: vgg19)
  --use_clip_loss                                  (auxiliary contrastive loss)
  --n_folds         N>1 for N-fold cross-validation (default: 1 = single run)
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cuDNN benchmark for speed; use --deterministic for exact GPU reproducibility
torch.backends.cudnn.benchmark = True

from data import ZeroShotDataset, ClassAwareSampler
from models.zero_shot_model import ZeroShotModel
from utils.config import (
    BATCH_SIZE,
    LR,
    LR_CONV,
    LOSS,
    HINGE_MARGIN,
    K,
    MODEL_TYPE,
    CONV_CHANNELS,
    CONV_FEATURE_LAYER,
    FT_HIDDEN,
    GV_HIDDEN,
    NUM_WORKERS,
    CUB_UNSEEN,
    FLOWER_UNSEEN,
    TEXT_ENCODER,
    IMAGE_BACKBONE,
    CLIP_WEIGHT,
    CLIP_TEMPERATURE,
    _TEXT_ENCODER_DIMS,
)
from utils.losses import (
    get_criterion,
    euclidean_loss,
    clip_contrastive_loss,
    center_alignment_loss,
    embedding_mse_loss,
)
from utils.filename_utils import generate_filename_components
from utils.seed_utils import set_seed, worker_init_fn


def train_step(
    model: ZeroShotModel,
    optimizer: torch.optim.Optimizer,
    criterion,
    images: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    use_hinge: bool = False,
    use_euclidean: bool = False,
    non_blocking: bool = False,
    use_clip_loss: bool = False,
    clip_weight: float = 0.1,
    clip_temperature: float = 0.07,
    # Alignment loss arguments
    use_alignment: bool = False,
    align_weight: float = 0.1,
    # Direct embedding loss (Variant 2)
    use_embedding_loss: bool = False,
    embedding_weight: float = 1.0,
) -> float:
    """
    Single training step with minibatch loss (Paper Sec 4).

    Paper: Loss computed only over classes present in batch (O(B*U), U≤B).
    Quote: "In practice, we do not sum over all the classes in the dataset.
    Instead, we sum over all the image labels from the minibatch only."

    CRITICAL: The model outputs scores for ALL classes, but loss is only computed
    for classes present in the batch. This matches the paper's approach and ensures
    training/evaluation use the same score computation.

    Args:
        model: ZeroShotModel instance
        optimizer: Adam optimizer
        criterion: Loss function (BCE, Hinge, or Euclidean with SUM reduction)
        images: [B,3,224,224] tensor
        text_features: [C_seen, text_dim] tensor (seen classes only, on device)
        labels: [B] tensor in [0, C_seen-1] (seen-subset indices, pre-remapped by caller)
        device: torch device
        use_hinge: Whether using hinge loss
        use_euclidean: Whether using Euclidean embedding loss (Paper Sec 4.2.1)
        non_blocking: Whether to use async GPU transfer
        use_clip_loss: Whether to add CLIP-style contrastive loss (fc/fc+conv only)
        clip_weight: Weight for the CLIP contrastive loss term
        clip_temperature: Temperature for the CLIP softmax (default 0.07)
        use_alignment: Whether to add center alignment loss
        align_weight: Weight for the center alignment term
        use_embedding_loss: Whether to use direct embedding MSE loss (Variant 2)
        embedding_weight: Weight for the embedding MSE term

    Returns:
        Loss value (scalar, summed over batch*classes_in_batch + alignment terms)
    """
    model.train()
    images = images.to(device, non_blocking=non_blocking)
    labels = labels.to(device, non_blocking=non_blocking)

    # Euclidean (Sec 4.2.1) needs embeddings for distance computation.
    # Also needed for auxiliary alignment losses.
    return_embeddings = use_euclidean or use_clip_loss or use_alignment or use_embedding_loss

    if return_embeddings:
        all_scores, image_emb, text_emb = model(images, text_features, return_embeddings=True)
    else:
        all_scores = model(images, text_features)  # [B, C_all]
        image_emb, text_emb = None, None

    # All losses operate on scores/embeddings over batch classes
    unique_classes, inverse = torch.unique(labels, return_inverse=True)
    batch_scores = all_scores[:, unique_classes]  # [B, U]

    num_classes = unique_classes.numel()

    if use_euclidean and image_emb is not None and text_emb is not None:
        # Direct Euclidean distance loss (Paper Sec 4.2.1):
        # Positive pairs: minimize ||g_i - f_j||²
        # Negative pairs: max(0, margin - ||g_i - f_j||)²
        batch_text_emb = text_emb[unique_classes]  # [U, k]
        targets = torch.zeros(batch_scores.size(0), num_classes, device=batch_scores.device, dtype=batch_scores.dtype)
        targets[torch.arange(batch_scores.size(0), device=batch_scores.device), inverse] = 1.0
        targets = 2 * targets - 1  # {+1, -1}
        loss = euclidean_loss(image_emb, batch_text_emb, targets, margin=HINGE_MARGIN)
    else:
        targets = torch.zeros(batch_scores.size(0), num_classes, device=batch_scores.device, dtype=batch_scores.dtype)
        targets[torch.arange(batch_scores.size(0), device=batch_scores.device), inverse] = 1.0
        if use_hinge:
            targets = 2 * targets - 1  # Convert to {+1,-1}
        loss = criterion(batch_scores, targets)

    # Add CLIP contrastive loss
    if use_clip_loss and image_emb is not None and text_emb is not None:
        pos_text_emb = text_emb[labels]  # [B, k] — ground-truth class embedding per image
        loss = loss + clip_weight * clip_contrastive_loss(image_emb, pos_text_emb, temperature=clip_temperature)

    # Add center alignment loss
    if use_alignment and image_emb is not None and text_emb is not None:
        pos_text_emb = text_emb[labels]  # [B, k]
        loss_align = center_alignment_loss(image_emb, pos_text_emb)
        loss = loss + align_weight * loss_align

    # Add direct embedding MSE loss (Variant 2)
    if use_embedding_loss and image_emb is not None and text_emb is not None:
        pos_text_emb = text_emb[labels]  # [B, k]
        loss_emb = embedding_mse_loss(image_emb, pos_text_emb, reduction="sum")
        loss = loss + embedding_weight * loss_emb

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def _run_one_fold(args, fold_seed: int, fold_idx: int, n_folds: int) -> dict:
    """
    Run one training fold with the given seed.

    In zero-shot CV, each fold uses a different seed so that a different random
    subset of classes is held out as unseen. fold_seed = args.seed + fold_idx.

    Returns a dict of best-epoch metrics:
        seen_top1, seen_top5, unseen_top1, unseen_top5, best_epoch, fold_idx, fold_seed
    """
    # Reproducibility: same fold_seed => same data splits, model init, and training order
    set_seed(fold_seed, deterministic=args.deterministic)

    if args.use_clip_loss and args.model_type == "conv":
        logger.warning(
            "CLIP loss is not supported for model_type='conv' (no FC text embeddings). "
            "The CLIP term will be silently skipped every batch. "
            "Use model_type='fc' or 'fc+conv' to benefit from CLIP loss."
        )

    # Alignment loss compatibility warnings
    if args.model_type == "conv":
        if args.use_center_align:
            logger.warning(
                "Center alignment loss is not supported for model_type='conv' (no FC text embeddings). "
                "The alignment term will be silently skipped. "
                "Use model_type='fc' or 'fc+conv' to benefit from alignment loss."
            )
        if args.use_embedding_loss:
            logger.warning(
                "Embedding MSE loss is not supported for model_type='conv' (no FC text embeddings). "
                "The embedding term will be silently skipped. "
                "Use model_type='fc' or 'fc+conv' to benefit from embedding loss."
            )


    # Early stopping is enabled by default
    early_stopping_enabled = not args.no_early_stopping
    device = torch.device(args.device)

    # IMPORTANT: Determine n_unseen BEFORE generating checkpoint/log filenames
    if args.n_unseen is None:
        if args.dataset == "cub":
            n_unseen = CUB_UNSEEN
        else:
            n_unseen = FLOWER_UNSEEN
    else:
        n_unseen = args.n_unseen

    # Initialize CSV logging
    csv_file = None
    csv_writer = None
    log_path = None

    # Determine log file path
    log_input = args.log_file.strip() if args.log_file else ""

    if log_input:
        if "/" in log_input or "\\" in log_input:
            log_path = Path(f"{log_input}.csv")
        else:
            log_path = Path("logs") / f"{log_input}.csv"
    else:
        components = generate_filename_components(
            args.model_type, args.loss, args.dataset, args.conv_feature_layer, n_unseen,
            args.train_ratio, args.text_encoder, args.image_backbone,
            args.use_clip_loss, args.clip_weight, args.fc_mode,
        )
        log_path = Path("logs") / ("_".join(components) + ".csv")

    # Save each fold to its own subdirectory: logs/fold{i}/base_name.csv
    if n_folds > 1:
        log_path = log_path.parent / f"fold{fold_idx}" / log_path.name

    # Create parent directory and open file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'loss', 'seen_top1', 'seen_top5', 'unseen_top1', 'unseen_top5',
        'learning_rate', 'model_type', 'dataset'
    ])

    code_root = Path(__file__).resolve().parents[1]
    data_root = (code_root / args.data_root).resolve()

    # Determine wikipedia jsonl path
    if args.wikipedia_jsonl:
        wikipedia_jsonl = Path(args.wikipedia_jsonl)
    else:
        if args.dataset == "cub":
            wikipedia_jsonl = data_root / "wikipedia" / "birds.jsonl"
        else:
            wikipedia_jsonl = data_root / "wikipedia" / "flowers.jsonl"

    # Create train dataset first to get shared text_features (fold_seed for reproducible splits)
    train_dataset = ZeroShotDataset(
        jsonl_path=wikipedia_jsonl,
        images_base=data_root / "images",
        mode="train",
        n_unseen=n_unseen,
        train_ratio=args.train_ratio,
        seed=fold_seed,
        text_encoder=args.text_encoder,
    )

    # Share text_features across all splits to ensure consistency
    test_seen_dataset = ZeroShotDataset(
        jsonl_path=wikipedia_jsonl,
        images_base=data_root / "images",
        mode="test_seen",
        n_unseen=n_unseen,
        train_ratio=args.train_ratio,
        seed=fold_seed,
        text_encoder=args.text_encoder,
    )
    test_seen_dataset.text_features = train_dataset.text_features
    test_seen_dataset.tfidf_matrix = train_dataset.text_features  # backward-compat alias
    test_seen_dataset.label_to_idx = train_dataset.label_to_idx

    test_unseen_dataset = ZeroShotDataset(
        jsonl_path=wikipedia_jsonl,
        images_base=data_root / "images",
        mode="test_unseen",
        n_unseen=n_unseen,
        train_ratio=args.train_ratio,
        seed=fold_seed,
        text_encoder=args.text_encoder,
    )
    test_unseen_dataset.text_features = train_dataset.text_features
    test_unseen_dataset.tfidf_matrix = train_dataset.text_features  # backward-compat alias
    test_unseen_dataset.label_to_idx = train_dataset.label_to_idx

    pin_memory = device.type == "cuda"
    prefetch_factor = 4 if NUM_WORKERS > 0 else None

    if args.standard_sampler:
        g = torch.Generator()
        g.manual_seed(fold_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=prefetch_factor,
            worker_init_fn=partial(worker_init_fn, base_seed=fold_seed),
        )
        sampler_type = "RandomSampler (paper method)"
    else:
        train_sampler = ClassAwareSampler(
            train_dataset,
            batch_size=args.batch_size,
            classes_per_batch=min(50, len(train_dataset.seen_classes)),
            seed=fold_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=prefetch_factor,
            worker_init_fn=partial(worker_init_fn, base_seed=fold_seed),
        )
        sampler_type = "ClassAwareSampler (better diversity)"

    test_seen_loader = DataLoader(
        test_seen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=prefetch_factor,
    )
    test_unseen_loader = DataLoader(
        test_unseen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=prefetch_factor,
    )

    # Global text features matrix [C, text_dim] shared across splits
    text_features = torch.from_numpy(train_dataset.text_features).float().to(device)
    num_classes = text_features.shape[0]
    actual_text_dim = text_features.shape[1]

    # Get expected text dimension from config based on text_encoder choice
    expected_text_dim = _TEXT_ENCODER_DIMS[args.text_encoder]

    # Validate: actual dimension should match expected dimension
    if actual_text_dim != expected_text_dim:
        raise ValueError(
            f"[TEXT DIM MISMATCH] Text encoder '{args.text_encoder}' should produce "
            f"{expected_text_dim}-d features, but dataset has {actual_text_dim}-d. "
            f"Check text_encoder configuration or TF-IDF max_features setting."
        )

    # Optional: --text_dim override (advanced use only, for custom experiments)
    text_input_dim = expected_text_dim
    if args.text_dim != -1:
        if args.text_dim != expected_text_dim:
            logger.warning(
                f"[TEXT DIM OVERRIDE] Using --text_dim={args.text_dim} instead of "
                f"expected {expected_text_dim} for {args.text_encoder}. This may cause "
                f"model incompatibility with checkpoints."
            )
        text_input_dim = args.text_dim

    model = ZeroShotModel(
        text_input_dim=text_input_dim,
        k=args.k,
        ft_hidden=args.ft_hidden,
        gv_hidden=args.gv_hidden,
        conv_channels=args.conv_channels,
        conv_feature_layer=args.conv_feature_layer,
        image_backbone=args.image_backbone,
        model_type=args.model_type,
        fc_mode=args.fc_mode,
    ).to(device)

    # Learning rate selection (note: differs from paper for conv modes)
    effective_lr = args.lr
    if args.model_type in ("conv", "fc+conv"):
        effective_lr = LR_CONV
        logger.info(
            f"[CONFIG NOTE] Using lr={effective_lr} for {args.model_type} mode "
            f"(Paper uses lr={LR} for all modes. "
            f"This empirical adjustment helps conv convergence.)"
        )

    if args.loss.lower() == "euclidean" and args.model_type == "conv":
        raise ValueError(
            "Euclidean loss requires FC embeddings but model_type='conv' has none. "
            "Use model_type='fc' or 'fc+conv', or switch to loss='bce'/'hinge'."
        )

    criterion = get_criterion(args.loss, HINGE_MARGIN)
    use_hinge = args.loss.lower() == "hinge"
    use_euclidean = args.loss.lower() == "euclidean"

    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)

    # Pre-compute evaluation data (constant across epochs)
    seen_classes = train_dataset.seen_classes
    unseen_classes = train_dataset.unseen_classes

    seen_indices = [train_dataset.label_to_idx[c] for c in seen_classes]
    unseen_indices = [train_dataset.label_to_idx[c] for c in unseen_classes]

    seen_label_map = {train_dataset.label_to_idx[c]: i for i, c in enumerate(seen_classes)}
    unseen_label_map = {train_dataset.label_to_idx[c]: i for i, c in enumerate(unseen_classes)}

    text_features_seen = text_features[seen_indices].to(device)
    text_features_unseen = text_features[unseen_indices].to(device)

    is_full_dataset = (n_unseen == 0)

    num_classes = len(train_dataset.label_to_idx)
    seen_label_map_tensor = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for global_idx, subset_idx in seen_label_map.items():
        seen_label_map_tensor[global_idx] = subset_idx

    unseen_label_map_tensor = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for global_idx, subset_idx in unseen_label_map.items():
        unseen_label_map_tensor[global_idx] = subset_idx

    def _evaluate(loader, split_name: str, eval_text_features: torch.Tensor, label_map_tensor: torch.Tensor, position: int = 0) -> tuple[float, float]:
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        n_eval_classes = eval_text_features.size(0)
        top5_k = min(5, n_eval_classes)
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"  Evaluating {split_name}", position=position, leave=False, ncols=70, colour="blue", disable=True):
                images = batch["image"].to(device, non_blocking=pin_memory)
                labels = batch["label"].to(device, non_blocking=pin_memory)
                remapped_labels = label_map_tensor[labels]
                scores = model(images, eval_text_features)
                _, pred_top1 = scores.max(1)
                _, pred_top5 = scores.topk(top5_k, dim=1)
                correct_top1 += (pred_top1 == remapped_labels).sum().item()
                correct_top5 += (pred_top5 == remapped_labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)
        if total > 0:
            top1 = correct_top1 / total
            top5 = correct_top5 / total
            return top1, top5
        return 0.0, 0.0

    # Early stopping variables
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # Best-epoch metrics tracked unconditionally (for CV summary reporting)
    best_seen_top1 = 0.0
    best_seen_top5 = 0.0
    best_unseen_top1 = 0.0
    best_unseen_top5 = 0.0

    # Determine checkpoint path
    save_input = args.save.strip() if args.save else ""

    if save_input:
        if "/" in save_input or "\\" in save_input:
            checkpoint_path = Path(f"{save_input}.pt")
        else:
            checkpoint_path = Path("checkpoints") / f"{save_input}.pt"
    else:
        components = generate_filename_components(
            args.model_type, args.loss, args.dataset, args.conv_feature_layer, n_unseen,
            args.train_ratio, args.text_encoder, args.image_backbone,
            args.use_clip_loss, args.clip_weight, args.fc_mode,
        )
        checkpoint_path = Path("checkpoints") / ("_".join(components) + ".pt")

    # Save each fold to its own subdirectory: checkpoints/fold{i}/base_name.pt
    if n_folds > 1:
        checkpoint_path = checkpoint_path.parent / f"fold{fold_idx}" / checkpoint_path.name

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training configuration summary ───────────────────────────────────
    fold_header = f" [Fold {fold_idx + 1}/{n_folds}, seed={fold_seed}]" if n_folds > 1 else ""
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        f"Training: {args.model_type} | {args.dataset} | {args.epochs} epochs | "
        f"lr={effective_lr:.6f} | {args.loss} | {device}{fold_header}"
    )
    logger.info(
        f"Data: {len(train_dataset.seen_classes)} seen / {len(train_dataset.unseen_classes)} unseen | "
        f"train={len(train_dataset)} | test_seen={len(test_seen_dataset)} | test_unseen={len(test_unseen_dataset)}"
    )
    encoder_parts = [f"text={args.text_encoder} ({text_input_dim}-d)", f"image={args.image_backbone}"]
    logger.info(f"Encoders: {' | '.join(encoder_parts)}")
    if args.use_clip_loss:
        logger.info(f"CLIP loss: weight={args.clip_weight}, temperature={args.clip_temperature}")

    # Alignment loss logging
    align_parts = []
    if args.use_center_align:
        align_parts.append(f"center_align(w={args.center_align_weight})")
    if args.use_embedding_loss:
        align_parts.append(f"embedding_mse(w={args.embedding_weight})")
    if align_parts:
        logger.info(f"Alignment losses: {' | '.join(align_parts)}")

    if is_full_dataset:
        es_monitor = "test accuracy"
    else:
        es_monitor = "unseen accuracy"
    early_stop_info = (
        f"Early stopping: {es_monitor} (patience={args.patience}, min_epochs={args.min_epochs})"
        if early_stopping_enabled else "Early stopping: disabled"
    )
    conv_info = f"conv_layer={args.conv_feature_layer}" if args.model_type in ("conv", "fc+conv") else ""
    config_parts = [f"batch={args.batch_size}", f"n_unseen={n_unseen}", conv_info, early_stop_info]
    logger.info(f"Config: {' | '.join(p for p in config_parts if p)}")
    logger.info(
        f"Repro: seed={fold_seed} | deterministic={args.deterministic} | "
        f"workers={NUM_WORKERS} | pin_memory={pin_memory} | sampler={sampler_type}"
    )
    logger.info(f"Output: ckpt={checkpoint_path} | log={log_path or 'disabled'}")
    logger.info("=" * 80)
    logger.info("")

    # Initialize epoch metric variables (guards against --epochs 0 edge case)
    seen_top1 = seen_top5 = unseen_top1 = unseen_top5 = 0.0
    avg_loss = 0.0

    for ep in range(args.epochs):
        epoch_id = ep + 1

        # Recompute unseen text embeddings with current text_encoder weights each epoch
        model.train()
        running_loss = 0.0
        num_samples = 0

        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch_id}/{args.epochs}", ncols=100, unit="batch", leave=False)
        for batch in pbar:
            images = batch["image"]
            labels = batch["label"]
            batch_size = images.size(0)
            # Remap global row indices → seen-subset indices, and use only seen
            # class text features to avoid computing scores for unseen classes
            # (unseen scores are always discarded in loss; this saves ~20% compute).
            labels_seen = seen_label_map_tensor[labels.to(device)]
            loss = train_step(
                model, optimizer, criterion, images, text_features_seen, labels_seen, device,
                use_hinge=use_hinge,
                use_euclidean=use_euclidean,
                non_blocking=pin_memory,
                use_clip_loss=args.use_clip_loss,
                clip_weight=args.clip_weight,
                clip_temperature=args.clip_temperature,
                # Alignment loss arguments
                use_alignment=args.use_center_align,
                align_weight=args.center_align_weight,
                use_embedding_loss=args.use_embedding_loss,
                embedding_weight=args.embedding_weight,
            )
            running_loss += loss
            num_samples += batch_size
            pbar.set_postfix({"loss": f"{running_loss/num_samples:.4f}"})
        pbar.close()

        avg_loss = running_loss / num_samples if num_samples > 0 else running_loss

        seen_top1, seen_top5 = _evaluate(test_seen_loader, "test_seen", text_features_seen, seen_label_map_tensor, position=1)
        if is_full_dataset:
            unseen_top1, unseen_top5 = 0.0, 0.0
        else:
            unseen_top1, unseen_top5 = _evaluate(test_unseen_loader, "test_unseen", text_features_unseen, unseen_label_map_tensor, position=2)

        elapsed = time.time() - epoch_start
        eta_seconds = elapsed * (args.epochs - epoch_id)
        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)

        if is_full_dataset:
            logger.info(
                f"Epoch {epoch_id:3d}/{args.epochs} | "
                f"Loss: {avg_loss:.3f} | "
                f"Test: {seen_top1*100:5.1f}% (Top-5: {seen_top5*100:5.1f}%) | "
                f"ETA: {eta_min:02d}:{eta_sec:02d}"
            )
        else:
            logger.info(
                f"Epoch {epoch_id:3d}/{args.epochs} | "
                f"Loss: {avg_loss:.3f} | "
                f"Seen: {seen_top1*100:5.1f}% (Top-5: {seen_top5*100:5.1f}%) | "
                f"Unseen: {unseen_top1*100:5.1f}% (Top-5: {unseen_top5*100:5.1f}%) | "
                f"ETA: {eta_min:02d}:{eta_sec:02d}"
            )

        if csv_writer:
            csv_writer.writerow([
                epoch_id,
                round(avg_loss, 6),
                round(seen_top1, 6),
                round(seen_top5, 6),
                round(unseen_top1, 6) if not is_full_dataset else None,
                round(unseen_top5, 6) if not is_full_dataset else None,
                round(effective_lr, 8),
                args.model_type,
                args.dataset
            ])
            csv_file.flush()

        # Early stopping + best-metrics tracking
        current_metric = seen_top1 if is_full_dataset else unseen_top1
        metric_name = "test top-1" if is_full_dataset else "unseen top-1"

        # Early stopping uses three phases:
        #   1. Warmup (epoch < min_save_epoch): track best_metric for logging
        #      only — don't save model state (random-init weights can produce
        #      misleadingly high unseen accuracy that never gets beaten).
        #   2. Observation (min_save_epoch <= epoch < min_epochs): save
        #      best_model_state on improvement, but don't count patience.
        #   3. Patience (epoch >= min_epochs): save best_model_state on
        #      improvement AND count patience toward early stopping.
        min_save_epoch = max(5, args.min_epochs // 10)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch_id
            best_seen_top1 = seen_top1
            best_seen_top5 = seen_top5
            best_unseen_top1 = unseen_top1
            best_unseen_top5 = unseen_top5
            if epoch_id >= min_save_epoch:
                best_model_state = copy.deepcopy(model.state_dict())
            if epoch_id >= args.min_epochs:
                patience_counter = 0
        elif epoch_id >= args.min_epochs:
            patience_counter += 1

        if early_stopping_enabled and epoch_id >= args.min_epochs and patience_counter >= args.patience:
            logger.info(f"\n🛑 Early stopping triggered at epoch {epoch_id}")
            logger.info(f"   Best {metric_name}: {best_metric*100:5.1f}% at epoch {best_epoch}")
            logger.info(f"   No improvement for {patience_counter} epochs (patience={args.patience})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(f"   Restored best model from epoch {best_epoch}")
                break
            else:
                logger.warning("No best model state found, skipping restoration")
                break

    # Save checkpoint with training config metadata
    config_meta = {
        "text_dim": text_input_dim,
        "k": args.k,
        "ft_hidden": args.ft_hidden,
        "gv_hidden": args.gv_hidden,
        "conv_channels": args.conv_channels,
        "conv_feature_layer": args.conv_feature_layer,
        "image_backbone": args.image_backbone,
        "model_type": args.model_type,
        "text_encoder": args.text_encoder,
        "fc_mode": args.fc_mode,
    }
    if early_stopping_enabled and best_model_state is not None:
        torch.save({"state_dict": best_model_state, "config": config_meta}, checkpoint_path)
        metric_name = "test_top1" if is_full_dataset else "unseen_top1"
        logger.info(f"Best checkpoint saved to {checkpoint_path} (epoch {best_epoch}, {metric_name}={best_metric*100:5.1f}%)")
    else:
        torch.save({"state_dict": model.state_dict(), "config": config_meta}, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        # When early stopping disabled, record final-epoch metrics as "best"
        if best_epoch == 0:
            best_seen_top1 = seen_top1
            best_seen_top5 = seen_top5
            best_unseen_top1 = unseen_top1
            best_unseen_top5 = unseen_top5
            best_epoch = args.epochs


    if csv_file:
        csv_file.close()
        logger.info(f"Training history saved to {log_path}")

    return {
        "seen_top1": best_seen_top1,
        "seen_top5": best_seen_top5,
        "unseen_top1": best_unseen_top1,
        "unseen_top5": best_unseen_top5,
        "best_epoch": best_epoch,
        "fold_idx": fold_idx,
        "fold_seed": fold_seed,
    }


def _print_cv_summary(fold_results: list, args) -> None:
    """Print mean ± std across all CV folds."""
    n = len(fold_results)
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Cross-Validation Summary ({n} folds, base seed={args.seed})")
    logger.info("=" * 80)
    for key in ["seen_top1", "seen_top5", "unseen_top1", "unseen_top5"]:
        vals = [r[key] for r in fold_results]
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        std = variance ** 0.5
        per_fold = "  ".join(f"fold{r['fold_idx']}={r[key]*100:.1f}%" for r in fold_results)
        logger.info(f"  {key:15s}: {mean*100:.2f}% ± {std*100:.2f}%   [{per_fold}]")
    best_epochs = [r["best_epoch"] for r in fold_results]
    mean_ep = sum(best_epochs) / n
    logger.info(f"  {'best_epoch':15s}: {mean_ep:.1f} avg   {best_epochs}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train zero-shot CNN (Ba et al. ICCV 2015).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model architecture ────────────────────────────────────────────────
    parser.add_argument("--model_type", default=MODEL_TYPE,
                        choices=("fc", "conv", "fc+conv"),
                        help="[paper] fc (Sec 3.2) | conv (Sec 3.3) | fc+conv (Sec 3.4)")
    parser.add_argument("--k", type=int, default=K,
                        help="[paper] joint embedding dimension")
    parser.add_argument("--ft_hidden", type=int, default=FT_HIDDEN,
                        help="[paper] text encoder hidden dim")
    parser.add_argument("--gv_hidden", type=int, default=GV_HIDDEN,
                        help="[paper] image fc-branch hidden dim")
    parser.add_argument("--conv_channels", type=int, default=CONV_CHANNELS,
                        help="[paper] K' predicted conv filters")
    parser.add_argument("--conv_feature_layer", default=CONV_FEATURE_LAYER,
                        choices=("conv5_3", "conv4_3", "pool5"),
                        help="[paper] VGG feature layer for conv branch")

    # ── Text encoding ─────────────────────────────────────────────────────
    parser.add_argument("--text_encoder", default=TEXT_ENCODER,
                        choices=("tfidf", "sbert", "sbert_multi", "clip", "clip_multi"),
                        help="[paper: tfidf] text feature extractor")
    parser.add_argument("--text_dim", type=int, default=-1,
                        help="Text feature dim (auto-detected from dataset when -1)")

    # ── Image backbone ────────────────────────────────────────────────────
    parser.add_argument("--image_backbone", default=IMAGE_BACKBONE,
                        choices=("vgg19", "densenet121", "resnet50"),
                        help="[paper: vgg19] image backbone; conv/fc+conv require vgg19")
    parser.add_argument("--fc_mode", default="default",
                        choices=("default", "penultimate"),
                        help="[extension] FC branch mode for DenseNet/ResNet: "
                             "'default' uses classifier head (1000-d), "
                             "'penultimate' skips it (1024/2048-d). Ignored for VGG-19.")

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="[paper] minibatch size")
    parser.add_argument("--lr", type=float, default=LR,
                        help="[paper] Adam learning rate (conv modes use LR_CONV by default)")
    parser.add_argument("--loss", default=LOSS,
                        choices=("bce", "hinge", "euclidean"),
                        help="[paper default: bce] loss function")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum training epochs")

    # ── CLIP contrastive loss (extension) ────────────────────────────────
    parser.add_argument("--use_clip_loss", action="store_true",
                        help="[extension] auxiliary CLIP-style contrastive loss (fc/fc+conv only)")
    parser.add_argument("--clip_weight", type=float, default=CLIP_WEIGHT,
                        help="[extension] CLIP loss weight λ (total = base + λ·clip)")
    parser.add_argument("--clip_temperature", type=float, default=CLIP_TEMPERATURE,
                        help="[extension] CLIP softmax temperature")

    # ── Alignment losses (GAN-augmented variants without GAN) ────────────
    parser.add_argument("--use_center_align", action="store_true",
                        help="[alignment] center alignment loss L_align = MSE(μ_g, μ_f) (fc/fc+conv only)")
    parser.add_argument("--center_align_weight", type=float, default=0.1,
                        help="[alignment] weight λ for center alignment (total = base + λ·L_align)")
    parser.add_argument("--use_embedding_loss", action="store_true",
                        help="[alignment] direct embedding MSE L_emb = MSE(g, f) (Variant 2, fc/fc+conv only)")
    parser.add_argument("--embedding_weight", type=float, default=1.0,
                        help="[alignment] weight for embedding MSE (total = base + λ·L_emb)")

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument("--dataset", default="cub", choices=("cub", "flowers"),
                        help="[paper] CUB-200-2011 or Oxford Flowers-102")
    parser.add_argument("--n_unseen", type=int, default=None,
                        help="Unseen classes (default: 40 CUB / 20 Flowers; 0 = supervised)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="[paper] seen-class train split ratio")
    parser.add_argument("--data_root", default="data",
                        help="Root dir with images/ and wikipedia/ subfolders")
    parser.add_argument("--wikipedia_jsonl", default="",
                        help="Explicit Wikipedia JSONL path (overrides auto-detect)")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--save", default="",
                        help="Checkpoint path without .pt (auto-generated if empty)")
    parser.add_argument("--log_file", default="",
                        help="CSV log path without .csv (auto-generated if empty)")

    # ── Reproducibility & hardware ────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds (default: 5). "
                             "Each fold uses seed+fold_idx to draw a different unseen-class split. "
                             "Checkpoints/logs get a _fold{i} suffix when n_folds > 1.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Deterministic cuDNN (exact reproducibility, slower)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Early stopping ────────────────────────────────────────────────────
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Disable early stopping (enabled by default)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience in epochs")
    parser.add_argument("--min_epochs", type=int, default=50,
                        help="Minimum epochs before early stopping can trigger")

    # ── Sampler ───────────────────────────────────────────────────────────
    parser.add_argument("--standard_sampler", action="store_true",
                        help="Use RandomSampler (paper); default: ClassAwareSampler")

    args = parser.parse_args()

    fold_results = []
    for fold_idx in range(args.n_folds):
        fold_seed = args.seed + fold_idx
        result = _run_one_fold(args, fold_seed, fold_idx, args.n_folds)
        fold_results.append(result)

    if args.n_folds > 1:
        _print_cv_summary(fold_results, args)


if __name__ == "__main__":
    main()
