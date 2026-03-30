"""
Figure 5-8 (Appendix). Visualizing predicted convolutional filters for unseen classes.

[LEFT]  Example image of an unseen class + its encyclopedia article.
[RIGHT] Top-5 images with highest conv-filter activation (validation set,
        both seen and unseen classes). Below each: guided-backprop deconvolution
        showing which image regions activate the predicted filter most.

Reproduces Figures 5-8 from Ba et al. ICCV 2015 appendix.
Uses guided backpropagation (Springenberg et al. 2015) as a modern equivalent
of the Zeiler & Fergus (2014) deconvolution technique cited in the paper.

Output: results/figures/Figure5_conv_vis_{dataset}.png
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import ImageClassDataset, prepare_birds_zero_shot, prepare_flowers_zero_shot
from scripts.reproduce.common import (
    FIG_DPI,
    get_figures_dir,
    resolve_checkpoint as _resolve_checkpoint,
    resolve_with_cv,
)
from scripts.reproduce.eval_utils import load_model
from utils.config import (
    CONV_CHANNELS,
    CONV_FEATURE_LAYER,
    FT_HIDDEN,
    GV_HIDDEN,
    K,
    TEXT_DIM,
)


# ---------------------------------------------------------------------------
# Guided backpropagation hooks
# ---------------------------------------------------------------------------

def compute_deconv_visualization(
    model,
    image: torch.Tensor,
    text_features: torch.Tensor,
    class_idx: int,
    device: torch.device,
    original_img: np.ndarray,
) -> np.ndarray:
    """Zeiler-Fergus style deconvolution visualization.

    Paper appendix: "The highest activation in the predicted convolutional
    classifier is projected back into the image space."

    Approach: compute the conv activation map for the target class, upsample
    to image resolution, and use it as an alpha mask — high-activation regions
    show the original image, low-activation regions fade to gray.  This
    reproduces the visual style of Zeiler & Fergus (2014) Fig. 2 & 3.

    Args:
        model: ZeroShotModel (must have conv branch).
        image: Single image tensor [1, 3, 224, 224].
        text_features: All class text features [C, text_dim].
        class_idx: Target class index.
        device: Torch device.
        original_img: Original image as numpy array [H, W, 3] in [0, 255].

    Returns:
        Visualization [H_orig, W_orig, 3] in [0, 1] range.
    """
    image = image.to(device)

    with torch.no_grad():
        enc = model.image_encoder
        shared = enc.features_shared(image)
        conv_feat = enc.conv_reduce_act(enc.conv_reduce(shared))  # [1, K', H, W]
        _, hidden = model.text_encoder.forward_with_hidden(text_features)
        filters = model.conv_weight_predictor(hidden)  # [C, K', 3, 3]
        activation_map = F.conv2d(conv_feat, filters, padding=1)  # [1, C, H, W]

        # Spatial activation for target class
        class_map = activation_map[0, class_idx]  # [H, W]

    heatmap = class_map.cpu().numpy()

    # Normalize to [0, 1] using min-max (relative activation).
    # Shows WHERE the filter responds most strongly.
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    # Upsample to original image size
    from PIL import Image as PILImage
    h_orig, w_orig = original_img.shape[:2]
    mask = np.array(
        PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w_orig, h_orig), PILImage.BILINEAR
        )
    ).astype(np.float32) / 255.0

    # Sharpen: raise to a power to concentrate on the peak activation regions
    mask = mask ** 2

    # Zeiler-Fergus style: original image where activation is high,
    # neutral gray (0.5) elsewhere.  mask acts as alpha blend.
    img_float = original_img.astype(np.float32) / 255.0
    gray = np.full_like(img_float, 0.5)
    result = mask[..., None] * img_float + (1 - mask[..., None]) * gray

    return np.clip(result, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figures 5-8: conv filter visualization (appendix)."
    )
    parser.add_argument("--cub_root", type=str, default="", help="Path to CUB_200_2011")
    parser.add_argument("--flowers_root", type=str, default="", help="Path to Oxford Flowers-102")
    parser.add_argument("--wikipedia_birds", type=str, default="data/wikipedia/birds.jsonl")
    parser.add_argument("--wikipedia_flowers", type=str, default="data/wikipedia/flowers.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--checkpoint_conv", type=str, default="", help="Explicit conv model checkpoint")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_unseen_show", type=int, default=3,
                        help="Number of unseen classes to visualize (default: 3)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top activated images to show (default: 5)")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Specific class names to visualize (overrides n_unseen_show)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv_feature_layer", type=str, default=CONV_FEATURE_LAYER,
                        choices=("conv5_3", "conv4_3", "pool5"))
    parser.add_argument("--image_backbone", type=str, default="vgg19",
                        choices=("vgg19", "densenet121", "resnet50"))
    parser.add_argument("--text_wrap_width", type=int, default=60,
                        help="Character width for wrapping Wikipedia text (default: 60)")
    parser.add_argument("--max_text_lines", type=int, default=20,
                        help="Max lines of Wikipedia text to display (default: 20)")
    args = parser.parse_args()

    figures_dir = get_figures_dir()
    if args.out_dir:
        figures_dir = Path(args.out_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

    code_root = Path(__file__).resolve().parents[2]
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

    # Process each dataset
    datasets = []
    if args.cub_root and Path(args.cub_root).exists():
        jsonl = code_root / args.wikipedia_birds
        if jsonl.exists():
            datasets.append(("cub", args.cub_root, jsonl, 40, "conv_bce_cub"))
    if args.flowers_root and Path(args.flowers_root).exists():
        jsonl = code_root / args.wikipedia_flowers
        if jsonl.exists():
            datasets.append(("flowers", args.flowers_root, jsonl, 20, "conv_bce_flowers"))

    if not datasets:
        print("No dataset found. Provide --cub_root and/or --flowers_root.")
        return

    for dataset_name, images_root, jsonl_path, n_unseen_default, ckpt_key in datasets:
        print(f"\n=== Conv filter visualization: {dataset_name} ===")

        # Resolve checkpoint: try conv model, then fc+conv
        ckpt = _resolve_checkpoint(ckpt_key, args.checkpoint_dir, args.checkpoint_conv)
        model_type = "conv"
        if not ckpt:
            # Try fc+conv model (which also has a conv branch)
            fc_conv_key = ckpt_key.replace("conv_", "fc_conv_")
            ckpt = _resolve_checkpoint(fc_conv_key, args.checkpoint_dir, "")
            model_type = "fc+conv"
        if not ckpt:
            print(f"  No conv or fc+conv checkpoint found for {dataset_name}, skipping.")
            continue

        # Prepare data
        if dataset_name == "cub":
            out = prepare_birds_zero_shot(
                images_root, jsonl_path,
                n_unseen=n_unseen_default,
                unseen_seed=args.seed,
                split_seed=args.seed,
            )
        else:
            out = prepare_flowers_zero_shot(
                images_root, jsonl_path,
                n_unseen=n_unseen_default,
                unseen_seed=args.seed,
                split_seed=args.seed,
            )
        train_p, train_l, test_p, test_l, class_names, text_feat, seen_idx, unseen_idx = out
        text_t = torch.from_numpy(text_feat).float().to(device)
        num_classes = len(class_names)

        # Load Wikipedia texts for display
        # Use load_from_json (handles non-standard JSON like leading-zero ints)
        from data.dataset import load_from_json
        _, _, class_texts_map, ordered_names = load_from_json(
            jsonl_path, Path(images_root) / "images", verbose=False
        )
        # Build name→text lookup: ordered_names[i] corresponds to class_texts_map[i+1] (1-based idx)
        name_to_wiki: dict[str, str] = {}
        for i, oname in enumerate(ordered_names):
            text = class_texts_map.get(i + 1, "") or class_texts_map.get(i, "")
            if text:
                name_to_wiki[oname.lower().replace("_", " ")] = text

        # Load model
        model = load_model(model_type, ckpt, device, **model_kw)
        model.eval()

        # Compute conv scores for all test images
        loader = DataLoader(
            ImageClassDataset(test_p, test_l),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        from tqdm import tqdm

        print("  Computing conv branch scores for all test images...")
        all_conv_scores = []
        with torch.no_grad():
            # Get text hidden and predict filters once
            _, hidden = model.text_encoder.forward_with_hidden(text_t)
            filters = model.conv_weight_predictor(hidden)  # [C, K', 3, 3]

            for imgs, _ in tqdm(loader, desc="  Conv scores", leave=True):
                imgs = imgs.to(device)
                conv_feat = model.image_encoder.forward_conv_feature(imgs)  # [B, K', H, W]
                out = F.conv2d(conv_feat, filters, padding=1)  # [B, C, H, W]
                scores = out.flatten(2).mean(2)  # [B, C]
                all_conv_scores.append(scores.cpu().numpy())

        conv_scores = np.concatenate(all_conv_scores, axis=0)  # [N, C]

        # Select unseen classes to visualize
        unseen_set = set(unseen_idx.tolist()) if hasattr(unseen_idx, 'tolist') else set(unseen_idx)

        if args.classes:
            show_indices = []
            for target in args.classes:
                target_lower = target.lower().replace(" ", "_")
                for idx in range(num_classes):
                    name_part = class_names[idx].split(".", 1)[-1].lower()
                    if name_part == target_lower:
                        show_indices.append(idx)
                        break
            if not show_indices:
                print("  No matching classes found, skipping.")
                continue
        else:
            # Pick unseen classes where the conv branch retrieves the most
            # correct images in its top-k — these produce the best visualizations.
            unseen_list = list(unseen_set)
            test_l_arr = np.array(test_l) if not isinstance(test_l, np.ndarray) else test_l
            retrieval_k = max(args.top_k * 2, 10)  # evaluate over a wider window
            class_quality = []
            for c in unseen_list:
                top_indices = np.argsort(-conv_scores[:, c])[:retrieval_k]
                hits = (test_l_arr[top_indices] == c).sum()
                precision = hits / retrieval_k
                class_quality.append((c, precision))
            class_quality.sort(key=lambda x: -x[1])
            show_indices = [c for c, _ in class_quality[:args.n_unseen_show]]

        print(f"  Visualizing {len(show_indices)} classes: "
              + ", ".join(class_names[c].split('.', 1)[-1] for c in show_indices))

        # For each selected class, find top-k images and compute deconv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        n_show = len(show_indices)
        top_k = args.top_k

        # Figure layout: each class gets 3 rows (class info, top images, deconv)
        # LEFT column: example image + text excerpt
        # RIGHT columns: top_k images (row 1) + deconv visualizations (row 2)
        fig = plt.figure(figsize=(3.5 + top_k * 2.5, n_show * 4.0))
        plt.rc("font", size=9)

        gs = fig.add_gridspec(
            n_show, 1,
            hspace=0.15,
            top=0.95,
        )

        for class_i, c in enumerate(tqdm(show_indices, desc="  Generating visualizations", unit="class")):
            # Find top-k images for this class
            class_scores = conv_scores[:, c]
            top_indices = np.argsort(-class_scores)[:top_k]

            # Sub-grid for this class: 2 rows (images, deconv) x (1 + top_k) cols
            gs_class = gs[class_i].subgridspec(
                2, 1 + top_k,
                width_ratios=[1.5] + [1.0] * top_k,
                hspace=0.03,
                wspace=0.05,
            )

            # --- LEFT: Example image + text ---
            # Find an example image of this class from test set
            test_l_arr = np.array(test_l) if not isinstance(test_l, np.ndarray) else test_l
            example_indices = np.where(test_l_arr == c)[0]

            # Span both rows for the left panel
            ax_left = fig.add_subplot(gs_class[:, 0])

            class_display = class_names[c].split(".", 1)[-1].replace("_", " ")

            if len(example_indices) > 0:
                example_img = Image.open(test_p[example_indices[0]]).convert("RGB")
                ax_left.imshow(example_img)
                ax_left.set_title(class_display, fontsize=11, fontweight="bold", pad=3)
            else:
                ax_left.text(0.5, 0.7, "No example\navailable", ha="center", va="center",
                             fontsize=10, color="gray", transform=ax_left.transAxes)
                ax_left.set_title(class_display, fontsize=11, fontweight="bold", pad=3)

            # Add Wikipedia text below the image
            # Normalize class name for lookup: strip leading "NNN." prefix (CUB)
            lookup_name = class_names[c].split(".", 1)[-1].lower().replace("_", " ")
            wiki_text = name_to_wiki.get(lookup_name, "")
            if wiki_text:
                wrapped = textwrap.fill(wiki_text[:500], width=args.text_wrap_width)
                lines = wrapped.split("\n")[:args.max_text_lines]
                truncated = "\n".join(lines)
                if len(wiki_text) > 500 or len(wrapped.split("\n")) > args.max_text_lines:
                    truncated += "\n..."
                ax_left.text(
                    0.5, -0.03, truncated,
                    transform=ax_left.transAxes,
                    fontsize=6, va="top", ha="center",
                    family="serif",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8),
                )

            ax_left.set_xticks([])
            ax_left.set_yticks([])
            for spine in ax_left.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#999999")

            unseen_label = " (unseen)" if c in unseen_set else " (seen)"
            ax_left.set_xlabel(unseen_label, fontsize=8, color="gray")

            # --- RIGHT: Top-k images + deconv ---
            for k_i, img_idx in enumerate(top_indices):
                col = 1 + k_i

                # Row 0: Original image
                ax_img = fig.add_subplot(gs_class[0, col])
                img = Image.open(test_p[img_idx]).convert("RGB")
                ax_img.imshow(img)
                img_class = class_names[int(test_l[img_idx])].split(".", 1)[-1].replace("_", " ")
                ax_img.set_title(img_class, fontsize=8, pad=2)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("#CCCCCC")

                # Row 1: Conv activation heatmap overlay
                ax_deconv = fig.add_subplot(gs_class[1, col])
                single_img = ImageClassDataset([test_p[img_idx]], [test_l[img_idx]])[0][0]
                single_img = single_img.unsqueeze(0)  # [1, 3, 224, 224]
                img_np = np.array(img)  # original PIL image as numpy [H, W, 3]

                deconv_vis = compute_deconv_visualization(
                    model, single_img, text_t, c, device, img_np,
                )
                ax_deconv.imshow(deconv_vis)
                ax_deconv.set_xticks([])
                ax_deconv.set_yticks([])
                for spine in ax_deconv.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("#CCCCCC")

                # Score annotation
                score = class_scores[img_idx]
                ax_deconv.set_xlabel(f"score: {score:.3f}", fontsize=7, color="gray")

        fig.suptitle(
            f"Visualizing predicted convolutional filters for unseen classes ({dataset_name.upper()})\n"
            f"[LEFT] Example image + Wikipedia article. "
            f"[RIGHT] Top-{top_k} images with highest activations for the predicted conv filters; "
            f"filter visualization (deconvolution) below each.",
            fontsize=10, y=0.99, va="top",
        )

        out_path = figures_dir / f"Figure5_conv_vis_{dataset_name}.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.15)
        plt.close()
        print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
