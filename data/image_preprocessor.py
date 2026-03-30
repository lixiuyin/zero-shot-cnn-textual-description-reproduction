"""
Image preprocessing for VGG (Ba et al., ICCV 2015).
Resize shortest side to 224, then center crop 224x224. Optional ImageNet normalization.
"""
from __future__ import annotations

import torch
from torchvision import transforms

# Paper: "each image is resized so that the shortest dimension stays at 224 pixels.
# A center patch of 224x224 is then cropped from the resized image."
# VGG typically uses ImageNet mean/std for normalization when using pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Cached transform objects to avoid repeated creation
_TRAIN_TRANSFORM = None
_EVAL_TRANSFORM = None


def get_train_transform():
    global _TRAIN_TRANSFORM
    if _TRAIN_TRANSFORM is None:
        _TRAIN_TRANSFORM = transforms.Compose([
            transforms.Resize(224),  # resize shortest side to 224
            transforms.CenterCrop(224),  # center crop 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return _TRAIN_TRANSFORM


def get_eval_transform():
    global _EVAL_TRANSFORM
    if _EVAL_TRANSFORM is None:
        _EVAL_TRANSFORM = get_train_transform()
    return _EVAL_TRANSFORM


def preprocess_for_vgg(pil_image) -> torch.Tensor:
    """Single image: PIL -> [1, 3, 224, 224] tensor, normalized."""
    t = get_eval_transform()
    return t(pil_image).unsqueeze(0)
