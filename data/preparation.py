"""
Data preparation functions for zero-shot learning (Ba et al. ICCV 2015).
Paper-aligned data splits:
- CUB-200-2010: 40 unseen / 160 seen (zero-shot); seen classes 80% train / 20% test
- CUB-200-2011: All 200 classes, 50/50 split per class
- Oxford Flowers: 20 unseen / 82 seen (zero-shot); seen classes 80% train / 20% test
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np

from .text_processor import texts_to_tfidf
from .text_sbert import texts_to_sbert
from .text_sbert_multi import texts_to_sbert_multi
from .text_clip import texts_to_clip
from .text_clip_multi import texts_to_clip_multi


def _get_text_features(ordered_texts: list[str], text_encoder: str) -> np.ndarray:
    """Dispatch text encoding based on encoder name."""
    _choices = ("tfidf", "sbert", "sbert_multi", "clip", "clip_multi")
    if text_encoder not in _choices:
        raise ValueError(f"text_encoder must be one of {_choices}; got {text_encoder!r}.")
    if text_encoder == "tfidf":
        features, _ = texts_to_tfidf(ordered_texts, max_features=9763)
    elif text_encoder == "sbert":
        features = texts_to_sbert(ordered_texts)
    elif text_encoder == "sbert_multi":
        features = texts_to_sbert_multi(ordered_texts)
    elif text_encoder == "clip":
        features = texts_to_clip(ordered_texts)
    elif text_encoder == "clip_multi":
        features = texts_to_clip_multi(ordered_texts)
    return features


def _load_cub_images_and_classes(cub_root: str | Path) -> tuple[list[str], list[int], list[str]]:
    """
    Load CUB-200-2011 images following paper structure.

    Args:
        cub_root: Path to CUB data root. Can be either:
            - data/images/birds (direct class dirs)
            - CUB_200_2011 (with images/ subdirectory)

    Returns:
        paths: list of image paths
        labels: list of class indices (0-199)
        class_names: list of class names aligned with indices
    """
    cub_root = Path(cub_root)

    # Check if images are directly under cub_root or under cub_root/images
    if (cub_root / "images").exists():
        images_dir = cub_root / "images"
    else:
        images_dir = cub_root

    if not images_dir.exists():
        raise FileNotFoundError(f"CUB images directory not found: {cub_root}")

    paths, labels = [], []
    class_names = []

    # CUB structure: images_dir/001.Black_footed_Albatross/*.jpg
    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])

    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name  # e.g., "001.Black_footed_Albatross"
        class_names.append(class_name)

        # Load all images for this class
        for img_path in sorted(class_dir.glob("*.jpg")):
            paths.append(str(img_path))
            labels.append(class_id)

    return paths, labels, class_names


def _load_flowers_images_and_classes(
    flowers_root: str | Path,
) -> tuple[list[str], list[int], list[str]]:
    """
    Load Oxford Flowers-102 images.
    Structure: flowers_root/class_name/*.jpg or flowers_root/label/*.jpg

    Returns:
        paths: list of image paths
        labels: list of class indices (0-101)
        class_names: list of class names aligned with indices
    """
    flowers_root = Path(flowers_root)

    if not flowers_root.exists():
        raise FileNotFoundError(f"Flowers directory not found: {flowers_root}")

    paths, labels = [], []
    class_names = []
    class_id_map = {}

    class_dirs = sorted([d for d in flowers_root.iterdir() if d.is_dir()])

    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        class_id_map[class_name] = class_id

        for img_path in sorted(class_dir.glob("*.jpg")):
            paths.append(str(img_path))
            labels.append(class_id)

    return paths, labels, class_names


def prepare_birds_zero_shot(
    cub_root: str | Path,
    wikipedia_jsonl: str | Path,
    n_unseen: int = 40,
    train_ratio_seen: float = 0.8,
    unseen_seed: int = 42,
    split_seed: int = 42,
    text_encoder: str = "tfidf",
) -> tuple[list[str], list[int], list[str], list[int], list[str], np.ndarray, list[int], list[int]]:
    """
    Prepare CUB-200-2010 zero-shot split (paper Table 1-3, Figure 2).

    Paper Section 5: "We use 40 classes as unseen and the remaining 160 classes as seen.
    Within the 160 seen classes, we use 80% of the images for training and the rest for testing."

    Args:
        cub_root: Path to CUB_200_2011 directory (contains images/)
        wikipedia_jsonl: Path to Wikipedia text JSONL for birds
        n_unseen: Number of unseen classes (paper: 40)
        train_ratio_seen: Train ratio for seen classes (paper: 0.8)
        unseen_seed: Random seed for unseen/seen class split
        split_seed: Random seed for train/test split within seen classes

    Returns:
        train_paths: Training image paths
        train_labels: Training labels (0-199)
        test_paths: Test image paths (both seen and unseen)
        test_labels: Test labels (0-199)
        class_names: All class names (200)
        text_features: TF-IDF features [200, 9763]
        seen_class_idx: List of seen class indices
        unseen_class_idx: List of unseen class indices
    """
    # Load all images and classes
    all_paths, all_labels, class_names = _load_cub_images_and_classes(cub_root)
    num_classes = len(class_names)

    # Split classes into seen/unseen (paper: 40 unseen, 160 seen)
    random.seed(unseen_seed)
    all_class_ids = list(range(num_classes))
    random.shuffle(all_class_ids)

    unseen_class_idx = sorted(all_class_ids[:n_unseen])
    seen_class_idx = sorted(all_class_ids[n_unseen:])

    # Group images by class
    class_to_images = {i: [] for i in range(num_classes)}
    for path, label in zip(all_paths, all_labels):
        class_to_images[label].append(path)

    # Split seen classes into train/test (paper: 80/20)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    random.seed(split_seed)
    for cls in seen_class_idx:
        imgs = class_to_images[cls]
        random.shuffle(imgs)
        split_idx = int(len(imgs) * train_ratio_seen)

        train_paths.extend(imgs[:split_idx])
        train_labels.extend([cls] * split_idx)
        test_paths.extend(imgs[split_idx:])
        test_labels.extend([cls] * len(imgs[split_idx:]))

    # Add all unseen class images to test set
    for cls in unseen_class_idx:
        imgs = class_to_images[cls]
        test_paths.extend(imgs)
        test_labels.extend([cls] * len(imgs))

    # Load Wikipedia texts and compute TF-IDF
    from .dataset import load_from_json
    _, _, class_texts, _ = load_from_json(wikipedia_jsonl, Path(cub_root) / "images", verbose=False)

    # Build ordered text list aligned with class_names
    ordered_texts = []
    for i, cname in enumerate(class_names):
        # Extract class index from name (e.g., "001.Black_footed_Albatross" -> 1)
        parts = cname.split(".", 1)
        class_id = int(parts[0])
        text = class_texts.get(class_id, "")
        ordered_texts.append(text)

    # Compute text features
    text_features = _get_text_features(ordered_texts, text_encoder)

    print(f"CUB-200 zero-shot: {len(seen_class_idx)} seen classes, {len(unseen_class_idx)} unseen classes")
    print(f"Train: {len(train_paths)} images, Test: {len(test_paths)} images")

    return (
        train_paths,
        train_labels,
        test_paths,
        test_labels,
        class_names,
        text_features,
        seen_class_idx,
        unseen_class_idx,
    )


def prepare_birds_50_50(
    cub_root: str | Path,
    wikipedia_jsonl: str | Path,
    seed: int = 42,
    text_encoder: str = "tfidf",
) -> tuple[list[str], list[int], list[str], list[int], list[str], np.ndarray]:
    """
    Prepare CUB-200-2011 full dataset 50/50 split (paper Table 4).

    Paper Section 5: "For CUB-200-2011 dataset... a 50/50 split is used for each class."

    Returns:
        train_paths, train_labels, test_paths, test_labels, class_names, text_features
    """
    all_paths, all_labels, class_names = _load_cub_images_and_classes(cub_root)
    num_classes = len(class_names)

    # Group images by class
    class_to_images = {i: [] for i in range(num_classes)}
    for path, label in zip(all_paths, all_labels):
        class_to_images[label].append(path)

    # 50/50 split per class
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    random.seed(seed)
    for cls in range(num_classes):
        imgs = class_to_images[cls]
        random.shuffle(imgs)
        split_idx = len(imgs) // 2

        train_paths.extend(imgs[:split_idx])
        train_labels.extend([cls] * split_idx)
        test_paths.extend(imgs[split_idx:])
        test_labels.extend([cls] * len(imgs[split_idx:]))

    # Load Wikipedia texts
    from .dataset import load_from_json
    _, _, class_texts, _ = load_from_json(wikipedia_jsonl, Path(cub_root) / "images", verbose=False)

    ordered_texts = []
    for i, cname in enumerate(class_names):
        parts = cname.split(".", 1)
        class_id = int(parts[0])
        text = class_texts.get(class_id, "")
        ordered_texts.append(text)

    text_features = _get_text_features(ordered_texts, text_encoder)

    print(f"CUB-200 50/50: {num_classes} classes, Train: {len(train_paths)}, Test: {len(test_paths)}")

    return train_paths, train_labels, test_paths, test_labels, class_names, text_features


def prepare_flowers_zero_shot(
    flowers_root: str | Path,
    wikipedia_jsonl: str | Path,
    n_unseen: int = 20,
    train_ratio_seen: float = 0.8,
    unseen_seed: int = 42,
    split_seed: int = 42,
    text_encoder: str = "tfidf",
) -> tuple[list[str], list[int], list[str], list[int], list[str], np.ndarray, list[int], list[int]]:
    """
    Prepare Oxford Flowers-102 zero-shot split (paper Table 1).

    Paper Section 5: "Oxford Flower dataset has 102 flower categories...
    We use 20 classes as unseen and the remaining 82 classes as seen."

    Mirrors prepare_birds_zero_shot: random seen/unseen class split and random
    train/test image split within seen classes, both controlled by seeds.
    Expects a single root directory containing one subdirectory per class.

    Args:
        flowers_root: Path to directory containing per-class image subdirectories
        wikipedia_jsonl: Path to Wikipedia text JSONL for flowers
        n_unseen: Number of unseen classes (paper: 20)
        train_ratio_seen: Train ratio for seen classes (paper: 0.8)
        unseen_seed: Random seed for unseen/seen class split
        split_seed: Random seed for train/test split within seen classes

    Returns:
        train_paths, train_labels, test_paths, test_labels, class_names,
        text_features, seen_class_idx, unseen_class_idx
    """
    all_paths, all_labels, class_names = _load_flowers_images_and_classes(flowers_root)
    num_classes = len(class_names)

    # Split classes into seen/unseen (same approach as CUB)
    random.seed(unseen_seed)
    all_class_ids = list(range(num_classes))
    random.shuffle(all_class_ids)

    unseen_class_idx = sorted(all_class_ids[:n_unseen])
    seen_class_idx = sorted(all_class_ids[n_unseen:])

    # Group images by class
    class_to_images = {i: [] for i in range(num_classes)}
    for path, label in zip(all_paths, all_labels):
        class_to_images[label].append(path)

    # Split seen classes into train/test (paper: 80/20)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    random.seed(split_seed)
    for cls in seen_class_idx:
        imgs = class_to_images[cls]
        random.shuffle(imgs)
        split_idx = int(len(imgs) * train_ratio_seen)

        train_paths.extend(imgs[:split_idx])
        train_labels.extend([cls] * split_idx)
        test_paths.extend(imgs[split_idx:])
        test_labels.extend([cls] * len(imgs[split_idx:]))

    # Add all unseen class images to test set
    for cls in unseen_class_idx:
        imgs = class_to_images[cls]
        test_paths.extend(imgs)
        test_labels.extend([cls] * len(imgs))

    # Load Wikipedia texts
    from .dataset import load_from_json
    _, _, class_texts, jsonl_names = load_from_json(wikipedia_jsonl, Path(flowers_root).parent, verbose=False)

    # Build name→text lookup from JSONL for robust matching (name may differ from dir name)
    _sorted_ids = sorted(class_texts.keys())
    _name_to_text = {name: class_texts[idx] for idx, name in zip(_sorted_ids, jsonl_names)}

    ordered_texts = []
    for i, cname in enumerate(class_names):
        # Priority 1: direct name match against JSONL class_name
        text = _name_to_text.get(cname, "")
        # Priority 2: numeric ID extracted from directory name (e.g. "042" or "042.tulip" → 42)
        if not text:
            try:
                numeric_id = int(cname.split(".")[0])
                text = class_texts.get(numeric_id, "")
            except (ValueError, IndexError):
                pass
        # Priority 3: positional 1-indexed fallback
        if not text:
            text = class_texts.get(i + 1, "")
        ordered_texts.append(text)

    text_features = _get_text_features(ordered_texts, text_encoder)

    print(f"Oxford Flowers zero-shot: {len(seen_class_idx)} seen, {len(unseen_class_idx)} unseen")
    print(f"Train: {len(train_paths)} images, Test: {len(test_paths)} images")

    return (
        train_paths,
        train_labels,
        test_paths,
        test_labels,
        class_names,
        text_features,
        seen_class_idx,
        unseen_class_idx,
    )


def prepare_flowers_50_50(
    flowers_root: str | Path,
    wikipedia_jsonl: str | Path,
    seed: int = 42,
    text_encoder: str = "tfidf",
) -> tuple[list[str], list[int], list[str], list[int], list[str], np.ndarray]:
    """
    Prepare Oxford Flowers-102 full dataset 50/50 split (paper Table 4).

    Returns:
        train_paths, train_labels, test_paths, test_labels, class_names, text_features
    """
    all_paths, all_labels, class_names = _load_flowers_images_and_classes(flowers_root)
    num_classes = len(class_names)

    # Group images by class
    class_to_images = {i: [] for i in range(num_classes)}
    for path, label in zip(all_paths, all_labels):
        class_to_images[label].append(path)

    # 50/50 split per class
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    random.seed(seed)
    for cls in range(num_classes):
        imgs = class_to_images[cls]
        random.shuffle(imgs)
        split_idx = len(imgs) // 2

        train_paths.extend(imgs[:split_idx])
        train_labels.extend([cls] * split_idx)
        test_paths.extend(imgs[split_idx:])
        test_labels.extend([cls] * len(imgs[split_idx:]))

    # Load Wikipedia texts
    from .dataset import load_from_json
    _, _, class_texts, jsonl_names = load_from_json(wikipedia_jsonl, Path(flowers_root).parent, verbose=False)

    _sorted_ids = sorted(class_texts.keys())
    _name_to_text = {name: class_texts[idx] for idx, name in zip(_sorted_ids, jsonl_names)}

    ordered_texts = []
    for i, cname in enumerate(class_names):
        text = _name_to_text.get(cname, "")
        if not text:
            try:
                numeric_id = int(cname.split(".")[0])
                text = class_texts.get(numeric_id, "")
            except (ValueError, IndexError):
                pass
        if not text:
            text = class_texts.get(i + 1, "")
        ordered_texts.append(text)

    text_features = _get_text_features(ordered_texts, text_encoder)

    print(f"Oxford Flowers 50/50: {num_classes} classes, Train: {len(train_paths)}, Test: {len(test_paths)}")

    return train_paths, train_labels, test_paths, test_labels, class_names, text_features


def prepare_flowers(
    flowers_train_root: str | Path,
    flowers_test_root: str | Path,
    wikipedia_jsonl: str | Path,
    text_encoder: str = "tfidf",
) -> tuple[list[str], list[int], list[str], list[int], list[str], np.ndarray]:
    """
    Prepare Oxford Flowers using predefined train/test splits.
    Used as alternative to 50/50 when train/test roots are provided separately.

    Returns:
        train_paths, train_labels, test_paths, test_labels, class_names, text_features
    """
    train_paths, train_labels, train_class_names = _load_flowers_images_and_classes(flowers_train_root)
    test_paths, test_labels, test_class_names = _load_flowers_images_and_classes(flowers_test_root)

    class_names = train_class_names if len(train_class_names) >= len(test_class_names) else test_class_names

    # Load Wikipedia texts
    from .dataset import load_from_json
    _, _, class_texts, jsonl_names = load_from_json(wikipedia_jsonl, Path(flowers_train_root).parent, verbose=False)

    _sorted_ids = sorted(class_texts.keys())
    _name_to_text = {name: class_texts[idx] for idx, name in zip(_sorted_ids, jsonl_names)}

    ordered_texts = []
    for i, cname in enumerate(class_names):
        text = _name_to_text.get(cname, "")
        if not text:
            try:
                numeric_id = int(cname.split(".")[0])
                text = class_texts.get(numeric_id, "")
            except (ValueError, IndexError):
                pass
        if not text:
            text = class_texts.get(i + 1, "")
        ordered_texts.append(text)

    text_features = _get_text_features(ordered_texts, text_encoder)

    print(f"Oxford Flowers: Train {len(train_paths)}, Test {len(test_paths)}")

    return train_paths, train_labels, test_paths, test_labels, class_names, text_features
