"""
Image encoder (Ba et al. ICCV 2015 Sec 3.2 & 3.3), extended with multiple backbones.

fc branch gv(·):
  - vgg19      : VGG-19 fc2 (4096-d, the last 4096-d hidden layer) → 4096-gv_hidden-k
  - densenet121: full network → avg pool (1024-d) → classifier (1000-d) → 1000-gv_hidden-k
  - resnet50   : full network → avg pool (2048-d) → fc (1000-d) → 1000-gv_hidden-k

conv branch g'_v(·): supported for all three backbones.
  feature map → conv_reduce (d→K') → [B, K', H, W]; predicted filters [C, K', 3, 3] applied externally.
  - vgg19:       conv_feature_layer ("conv5_3" 512×14×14, "conv4_3" 512×28×28, "pool5" 512×7×7)
  - densenet121: after denseblock3 (1024×14×14), conv_feature_layer ignored
  - resnet50:    after layer3 (1024×14×14), conv_feature_layer ignored

When both branches are used (fc+conv), the shared convolutional prefix is computed
only once via forward_both().

All pretrained weights are frozen (no fine-tuning), matching the paper spirit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    VGG19_Weights,
    DenseNet121_Weights,
    ResNet50_Weights,
)

# VGG-19 conv feature layer → slice index into vgg.features.children()
CONV_FEATURE_SLICE = {
    "pool5": None,   # full feature extractor → 512×7×7
    "conv5_3": -3,   # after conv5_3+ReLU, before conv5_4 → 512×14×14
    "conv4_3": -12,  # after conv4_3+ReLU, before conv4_4 → 512×28×28
}

_BACKBONE_CHOICES = ("vgg19", "densenet121", "resnet50")


class ImageEncoder(nn.Module):
    """Image encoder supporting VGG19 / DenseNet121 / ResNet50 backbones.

    The fc branch and conv branch are both available for all three backbones.

    Architecture (shared prefix avoids redundant computation):
      features_shared  →  conv branch: conv_reduce (d→K') → [B, K', H, W]
                       →  fc branch:   features_fc_suffix → flatten → fc_branch → projection → [B, k]
    """

    def __init__(
        self,
        output_dim: int = 50,
        gv_hidden: int = 300,
        conv_channels: int = 5,
        conv_feature_layer: str = "conv5_3",
        backbone: str = "vgg19",
        fc_mode: str = "default",
    ):
        """
        Args:
            output_dim: Output dimension k for joint embedding.
            gv_hidden: Hidden dimension for the fc projection branch.
            conv_channels: Number of predicted conv filters K'.
            conv_feature_layer: VGG-19 layer for conv branch ('conv5_3', 'conv4_3', 'pool5').
            backbone: One of 'vgg19', 'densenet121', 'resnet50'.
            fc_mode: FC branch mode for DenseNet/ResNet (ignored for VGG-19):
                - "default": pass through pretrained classifier head (1000-d), then project.
                - "penultimate": skip classifier, use avgpool features directly
                  (1024-d for DenseNet, 2048-d for ResNet), then project.
        """
        super().__init__()
        self.conv_channels = conv_channels
        self.conv_feature_layer = conv_feature_layer.lower()
        self.backbone = backbone.lower()
        self.fc_mode = fc_mode.lower()

        if self.backbone not in _BACKBONE_CHOICES:
            raise ValueError(
                f"backbone must be one of {_BACKBONE_CHOICES}; got {backbone!r}."
            )

        # ----------------------------
        # VGG-19
        # ----------------------------
        if self.backbone == "vgg19":
            if self.conv_feature_layer not in CONV_FEATURE_SLICE:
                raise ValueError(
                    f"conv_feature_layer must be one of {list(CONV_FEATURE_SLICE)}; "
                    f"got {conv_feature_layer!r}."
                )
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

            # Split features into shared prefix + fc-only suffix
            # VGG classifier: [0]Linear(25088,4096) [1]ReLU [2]Dropout
            #                 [3]Linear(4096,4096)  [4]ReLU [5]Dropout
            #                 [6]Linear(4096,1000)
            children = list(vgg.features.children())
            slice_idx = CONV_FEATURE_SLICE[self.conv_feature_layer]

            if slice_idx is None:
                # pool5: shared = all features, no suffix
                self.features_shared = nn.Sequential(*children)
                self.features_fc_suffix = nn.Identity()
            else:
                # conv5_3 / conv4_3: shared = prefix, suffix = remaining layers
                self.features_shared = nn.Sequential(*children[:slice_idx])
                self.features_fc_suffix = nn.Sequential(*children[slice_idx:])

            # fc branch: fc2 activation (4096-d)
            # Paper Sec 5.1: "image features are extracted by running VGG
            # pre-trained on ImageNet without fine-tuning" → deterministic
            # feature extraction.  Dropout is replaced with Identity so that
            # frozen features stay deterministic regardless of train/eval mode,
            # while preserving state_dict key indices for checkpoint compat.
            self.fc_branch = nn.Sequential(
                vgg.classifier[0],  # [0] Linear(25088 → 4096) fc1
                vgg.classifier[1],  # [1] ReLU
                nn.Identity(),      # [2] replaces Dropout — deterministic
                vgg.classifier[3],  # [3] Linear(4096 → 4096) fc2
                vgg.classifier[4],  # [4] ReLU
            )
            self.projection = nn.Sequential(
                nn.Linear(4096, gv_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gv_hidden, output_dim),
            )

            # conv branch: reduce d→K' channels (paper Sec 3.3)
            # Paper: "non-linear dimensionality reduction to reduce the number
            # of feature maps as in Sec. (3.2)" → Conv2d + ReLU.
            # ReLU kept separate to preserve conv_reduce.weight/bias key names
            # for checkpoint backward compatibility.
            self.conv_reduce = nn.Conv2d(512, conv_channels, kernel_size=3, padding=1)
            self.conv_reduce_act = nn.ReLU(inplace=True)

            # freeze pretrained weights
            for p in self.features_shared.parameters():
                p.requires_grad = False
            for p in self.features_fc_suffix.parameters():
                p.requires_grad = False
            for p in self.fc_branch.parameters():
                p.requires_grad = False

        # ----------------------------
        # DenseNet-121
        # ----------------------------
        elif self.backbone == "densenet121":
            dense = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # shared = through denseblock3 → 1024×14×14 (conv branch forks here)
            # suffix = transition3 + denseblock4 + norm5 (fc branch continues)
            dense_children = list(dense.features.children())
            self.features_shared = nn.Sequential(*dense_children[:9])   # through denseblock3
            self.features_fc_suffix = nn.Sequential(*dense_children[9:])  # transition3, denseblock4, norm5

            if self.fc_mode == "penultimate":
                # Skip classifier: avgpool → 1024-d → projection
                self.fc_branch = nn.Identity()
                fc_dim = 1024
            else:
                # Default: avgpool → classifier(1024→1000) → projection
                self.fc_branch = dense.classifier  # Linear(1024 → 1000)
                fc_dim = 1000

            self.projection = nn.Sequential(
                nn.Linear(fc_dim, gv_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gv_hidden, output_dim),
            )

            # conv branch (non-linear reduction, consistent with paper Sec 3.3)
            self.conv_reduce = nn.Conv2d(1024, conv_channels, kernel_size=3, padding=1)
            self.conv_reduce_act = nn.ReLU(inplace=True)

            # freeze pretrained weights
            for p in self.features_shared.parameters():
                p.requires_grad = False
            for p in self.features_fc_suffix.parameters():
                p.requires_grad = False
            for p in self.fc_branch.parameters():
                p.requires_grad = False

        # ----------------------------
        # ResNet-50
        # ----------------------------
        elif self.backbone == "resnet50":
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # shared = through layer3 → 1024×14×14 (conv branch forks here)
            # suffix = layer4 + avgpool (fc branch continues)
            resnet_children = list(resnet.children())
            self.features_shared = nn.Sequential(*resnet_children[:7])   # through layer3
            self.features_fc_suffix = nn.Sequential(*resnet_children[7:-1])  # layer4, avgpool

            if self.fc_mode == "penultimate":
                # Skip fc: avgpool → 2048-d → projection
                self.fc_branch = nn.Identity()
                fc_dim = 2048
            else:
                # Default: avgpool → fc(2048→1000) → projection
                self.fc_branch = resnet.fc  # Linear(2048 → 1000)
                fc_dim = 1000

            self.projection = nn.Sequential(
                nn.Linear(fc_dim, gv_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gv_hidden, output_dim),
            )

            # conv branch (non-linear reduction, consistent with paper Sec 3.3)
            self.conv_reduce = nn.Conv2d(1024, conv_channels, kernel_size=3, padding=1)
            self.conv_reduce_act = nn.ReLU(inplace=True)

            # freeze pretrained weights
            for p in self.features_shared.parameters():
                p.requires_grad = False
            for p in self.features_fc_suffix.parameters():
                p.requires_grad = False
            for p in self.fc_branch.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """Override train() to keep frozen modules in eval mode.

        DenseNet-121 and ResNet-50 contain BatchNorm layers inside the frozen
        shared prefix and fc suffix.  If left in train mode, BatchNorm uses
        batch statistics (noisy, batch-dependent) instead of pretrained
        running statistics, silently corrupting frozen feature extraction.
        VGG-19 has no BatchNorm so this is a no-op for the paper backbone.
        """
        super().train(mode)
        if mode:
            # Force frozen submodules back to eval so BatchNorm uses
            # pretrained running_mean / running_var (deterministic features).
            self.features_shared.eval()
            self.features_fc_suffix.eval()
            self.fc_branch.eval()
        return self

    def _frozen_features(self, x: torch.Tensor):
        """Run all frozen layers once, return inputs for trainable heads.

        Returns:
            (fc_in [B, D], shared [B, C, H, W]) for projection and conv_reduce.
        """
        with torch.no_grad():
            shared = self.features_shared(x)
            fc_x = self.features_fc_suffix(shared)
            if self.backbone == "densenet121":
                fc_x = F.relu(fc_x, inplace=False)
                fc_x = F.adaptive_avg_pool2d(fc_x, (1, 1))
            fc_x = torch.flatten(fc_x, 1)
            fc_x = self.fc_branch(fc_x)
        return fc_x, shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fc branch embeddings. [B,3,224,224] → [B,k]."""
        fc_x, _ = self._frozen_features(x)
        return self.projection(fc_x)

    def _apply_conv_reduce(self, shared: torch.Tensor) -> torch.Tensor:
        """Apply conv_reduce + ReLU (paper Sec 3.3 non-linear reduction)."""
        return self.conv_reduce_act(self.conv_reduce(shared))

    def forward_conv_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute conv branch features. [B,3,224,224] → [B,K',H,W]."""
        with torch.no_grad():
            shared = self.features_shared(x)
        return self._apply_conv_reduce(shared)

    def forward_both(self, x: torch.Tensor):
        """Compute both branches, sharing the frozen prefix.

        Returns:
            (fc_embeddings [B,k], conv_features [B,K',H,W]).
        """
        fc_x, shared = self._frozen_features(x)
        return self.projection(fc_x), self._apply_conv_reduce(shared)
