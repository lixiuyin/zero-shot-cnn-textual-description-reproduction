"""
Default configuration for Ba et al. ICCV 2015 reproduction.

All constants marked [paper] are taken directly from the paper.
Constants marked [empirical] deviate slightly from the paper and are
noted with a reason. Extension constants (text encoders, image backbones,
CLIP loss) are innovations beyond the paper — their defaults keep the
original paper behaviour.
"""
from __future__ import annotations

# Text encoding  [paper: TF-IDF, 9763-d]
TEXT_ENCODER = "tfidf"  # default text encoder; choices: "tfidf" | "sbert" | "sbert_multi" | "clip"
TEXT_DIM = 9763            # [paper] TF-IDF dimension for CUB/Flowers
TEXT_DIM_SBERT = 384      # [extension] all-MiniLM-L6-v2 dimension
TEXT_DIM_SBERT_MULTI = 384  # [extension] SBERT multi-granularity dimension
TEXT_DIM_CLIP = 512       # [extension] CLIP ViT-B/32 text tower dimension
TEXT_DIM_CLIP_MULTI = 512 # [extension] CLIP multi-granularity (per-sentence pooling, same dim)

# Text encoder to dimension mapping
_TEXT_ENCODER_DIMS = {
    "tfidf": TEXT_DIM,
    "sbert": TEXT_DIM_SBERT,
    "sbert_multi": TEXT_DIM_SBERT_MULTI,
    "clip": TEXT_DIM_CLIP,
    "clip_multi": TEXT_DIM_CLIP_MULTI,
}

# Joint embedding space
K = 50       # [paper] joint embedding size (classifier weight dim)
FT_HIDDEN = 300  # [paper] text encoder hidden: p → 300 → k
GV_HIDDEN = 300  # [paper] image fc branch hidden: 4096 → 300 → k

# Image backbone  [paper: VGG-19, frozen]
IMAGE_FC_DIM = 4096       # [paper] VGG-19 fc1 output dimension
IMAGE_BACKBONE = "vgg19"  # default backbone; "densenet121" | "resnet50" for extensions

# Training  [paper Sec 5.1]
BATCH_SIZE = 200   # [paper] minibatch of 200 images
LR = 1e-4          # [paper] Adam learning rate
LR_CONV = 5e-4     # [empirical] higher LR for conv/fc+conv helps convergence
OPTIMIZER = "adam" # [paper] Adam optimizer
LOSS = "bce"       # [paper default] "bce" | "hinge" | "euclidean"
HINGE_MARGIN = 1.0 # [paper] hinge loss margin

# Model type  [paper Sec 3]
MODEL_TYPE = "fc"  # [paper] "fc" (Sec 3.2) | "conv" (Sec 3.3) | "fc+conv" (Sec 3.4)

# Conv branch  [paper Sec 3.3]
CONV_CHANNELS = 5            # [paper] K' predicted filters (3×3)
CONV_FEATURE_LAYER = "conv5_3"  # [paper] VGG feature layer; "conv4_3" | "pool5" also valid

# CLIP contrastive loss  [extension, not in paper]
CLIP_WEIGHT = 0.1       # auxiliary CLIP loss weight λ (total loss = base + λ·clip)
CLIP_TEMPERATURE = 0.07 # CLIP softmax temperature (matching original CLIP paper)

# Data / hardware
IMAGE_SIZE = 224
NUM_WORKERS = 8  # parallel DataLoader workers

# Dataset splits  [paper Sec 5]
CUB_UNSEEN = 40  # CUB-200-2011: 200 classes → 40 unseen / 160 seen; 80/20 within seen
FLOWER_UNSEEN = 20  # Oxford Flowers-102: 102 classes → 20 unseen / 82 seen
