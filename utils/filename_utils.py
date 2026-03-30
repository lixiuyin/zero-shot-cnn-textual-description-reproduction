"""
Filename generation utilities for checkpoint and log files.

Naming convention (only non-default options are appended):
    {model_type}_{loss}_{dataset}_{layer}_{n_unseen}
        [_te{encoder}]     when text_encoder  != "tfidf"
        [_bb{backbone}]    when image_backbone != "vgg19"
        [_clip{weight}]    when use_clip_loss  == True
        [_tr{ratio}]       when train_ratio    != 0.8

Examples
--------
Paper default (fc, bce, cub, 40 unseen):
    fc_bce_cub_fc_40

With SBERT + ResNet50:
    fc_bce_cub_fc_40_tesbert_bbresnet50

With CLIP contrastive loss:
    fc_bce_cub_fc_40_clip0.1

Full-dataset supervised (train_ratio=0.5):
    fc_bce_cub_fc_0_tr0.5

All extensions combined:
    fc_bce_cub_fc_40_tesbert_bbresnet50_clip0.1
"""
from __future__ import annotations


def generate_filename_components(
    model_type: str,
    loss: str,
    dataset: str,
    conv_feature_layer: str,
    n_unseen: int,
    train_ratio: float,
    text_encoder: str = "tfidf",
    image_backbone: str = "vgg19",
    use_clip_loss: bool = False,
    clip_weight: float = 0.1,
    fc_mode: str = "default",
) -> list[str]:
    """Generate filename components encoding the full training configuration.

    Only non-default options are appended so that paper-reproduction runs
    produce clean, short filenames while innovation runs are self-describing.

    Args:
        model_type:         "fc" | "conv" | "fc+conv"
        loss:               "bce" | "hinge" | "euclidean"
        dataset:            "cub" | "flowers"
        conv_feature_layer: "conv5_3" | "conv4_3" | "pool5"
        n_unseen:           Number of unseen classes (0 = supervised baseline)
        train_ratio:        Train split for seen classes (0.8 = paper default)
        text_encoder:       "tfidf" | "sbert" | "sbert_multi" | "clip"
        image_backbone:     "vgg19" | "densenet121" | "resnet50"
        use_clip_loss:      Whether auxiliary CLIP contrastive loss is active
        clip_weight:        CLIP loss weight λ (only encoded when use_clip_loss=True)
        fc_mode:            "default" | "penultimate" (DenseNet/ResNet fc branch mode)

    Returns:
        List of string components to join with "_".
    """
    # "fc+conv" → "fc_conv" to avoid "+" in file paths
    model_name = "fc_conv" if model_type == "fc+conv" else model_type

    # Base components always present
    components = [model_name, loss, dataset]

    # Layer: fc models use "fc" as a sentinel; conv models use the actual layer
    if model_type in ("conv", "fc+conv"):
        components.append(conv_feature_layer or "conv5_3")
    else:
        components.append("fc")

    # Number of unseen classes
    components.append(str(n_unseen))

    # ── Extension suffixes (non-default only) ──────────────────────────────

    # Text encoder (paper default: tfidf)
    if text_encoder and text_encoder != "tfidf":
        # "sbert_multi" → "tesmbert" would be confusing; use "tesbert_multi"
        components.append(f"te{text_encoder}")

    # Image backbone (paper default: vgg19)
    if image_backbone and image_backbone != "vgg19":
        components.append(f"bb{image_backbone}")

    # FC mode (default: "default" = use classifier head)
    if fc_mode and fc_mode != "default":
        components.append(f"fc{fc_mode}")

    # CLIP contrastive loss
    if use_clip_loss:
        # Use %g format: removes trailing zeros without corrupting integer-like values
        # e.g. 0.1 → "0.1", 1.0 → "1", 10 → "10" (not "1")
        weight_str = f"{clip_weight:g}"
        components.append(f"clip{weight_str}")

    # Train ratio (paper default: 0.8)
    if train_ratio != 0.8:
        ratio_str = f"{train_ratio:g}"
        components.append(f"tr{ratio_str}")

    return components
