"""
Zero-shot models from Ba et al. ICCV 2015 Sec 3.2, 3.3, 3.4.
- fc: Sec 3.2, text predicts FC output weights; score y_c = w_c^T gv(x), w_c = f_t(t_c).
- conv: Sec 3.3, text predicts conv filters; score from global pool of conv(feature_map, predicted filters).
- fc+conv: Sec 3.4, joint; score = fc_score + conv_score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .weight_predictor import ConvWeightPredictor


class ZeroShotModel(nn.Module):
    """Zero-shot model supporting fc, conv, or fc+conv architectures.

    Implements three models from Ba et al. ICCV 2015:
    - Sec 3.2: fc model - text predicts FC output weights
    - Sec 3.3: conv model - text predicts conv filters
    - Sec 3.4: fc+conv model - joint combination

    Attributes:
        model_type: One of 'fc', 'conv', or 'fc+conv'.
        text_encoder: Encodes text TF-IDF features to embedding space.
        image_encoder: Encodes images using frozen pretrained features.
        conv_weight_predictor: Predicts conv filters from text (conv/fc+conv only).
        conv_channels: Number of conv channels (K' in paper).
    """

    def __init__(
        self,
        text_input_dim: int = 9763,
        k: int = 50,
        ft_hidden: int = 300,
        gv_hidden: int = 300,
        conv_channels: int = 5,
        conv_feature_layer: str = "conv5_3",
        image_backbone: str = "vgg19",
        model_type: str = "fc",
        fc_mode: str = "default",
    ):
        """Initialize the ZeroShotModel.

        Args:
            text_input_dim: Dimension of text features (9763 for TF-IDF, 384 for SBERT, 512 for CLIP).
            k: Size of joint embedding space (default 50).
            ft_hidden: Hidden dimension for text encoder (default 300).
            gv_hidden: Hidden dimension for image fc branch (default 300).
            conv_channels: Number of conv filters to predict K' (default 5).
            conv_feature_layer: VGG layer for conv branch ('conv5_3', 'conv4_3', 'pool5').
            image_backbone: Image backbone — 'vgg19', 'densenet121', or 'resnet50'.
            model_type: Model architecture — 'fc', 'conv', or 'fc+conv'.
            fc_mode: FC branch mode for DenseNet/ResNet — 'default' or 'penultimate'.
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.image_backbone = image_backbone.lower()

        if self.model_type not in ("fc", "conv", "fc+conv"):
            raise ValueError("model_type must be 'fc', 'conv', or 'fc+conv'")
        if self.image_backbone not in ("vgg19", "densenet121", "resnet50"):
            raise ValueError(
                "image_backbone must be one of 'vgg19', 'densenet121', 'resnet50'; "
                f"got {image_backbone!r}."
            )

        self.text_encoder = TextEncoder(
            input_dim=text_input_dim,
            hidden_dim=ft_hidden,
            output_dim=k,
        )
        self.image_encoder = ImageEncoder(
            output_dim=k,
            gv_hidden=gv_hidden,
            conv_channels=conv_channels,
            conv_feature_layer=conv_feature_layer,
            backbone=self.image_backbone,
            fc_mode=fc_mode,
        )
        self.conv_channels = conv_channels

        if self.model_type in ("conv", "fc+conv"):
            self.conv_weight_predictor = ConvWeightPredictor(
                hidden_dim=ft_hidden,
                k_prime=conv_channels,
                filter_size=3,
            )
        else:
            self.conv_weight_predictor = None

    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Compute classification scores for images vs text classes.

        Args:
            images: Batch of images of shape [B, 3, 224, 224].
            text_features: TF-IDF features for all classes [C, text_dim].
            return_embeddings: If True, also return the raw image and text
                embeddings ``(scores, g, f)`` for use with auxiliary losses
                such as CLIP contrastive loss.  For the ``conv`` model ``f``
                is ``None``.  Ignored (always False) during evaluation.

        Returns:
            When ``return_embeddings=False`` (default): scores ``[B, C]``.
            When ``return_embeddings=True``: tuple
            ``(scores [B, C], g [B, k], f [C, k] or None)``.

        Raises:
            ValueError: If input tensors have incorrect shapes.
        """
        # Input validation
        if images.dim() != 4:
            raise ValueError(f"Expected images to be 4D [B, 3, 224, 224], got shape {images.shape}")
        if images.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected images shape [B, 3, 224, 224], got {images.shape}")
        if text_features.dim() != 2:
            raise ValueError(f"Expected text_features to be 2D [C, D], got shape {text_features.shape}")
        if images.size(0) == 0:
            raise ValueError("Batch size cannot be zero")
        if text_features.size(0) == 0:
            raise ValueError("Number of classes cannot be zero")
        if self.model_type == "fc":
            return self._forward_fc(images, text_features, return_embeddings)
        if self.model_type == "conv":
            scores = self._forward_conv(images, text_features)
            if return_embeddings:
                return scores, None, None
            return scores
        # fc+conv: compute shared prefixes once for both encoders
        g_emb, conv_feat = self.image_encoder.forward_both(images)
        f, hidden = self.text_encoder.forward_with_hidden(text_features)
        fc_scores = g_emb @ f.T

        filters = self.conv_weight_predictor(hidden)
        conv_scores = F.conv2d(conv_feat, filters, padding=1).flatten(2).mean(2)

        if return_embeddings:
            return fc_scores + conv_scores, g_emb, f
        return fc_scores + conv_scores

    def _forward_fc(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Compute FC branch scores (Sec 3.2).

        Paper Eq: y_c = w_c^T gv(x), where w_c = f_t(t_c)
        Score: g(images) @ f(texts).T

        Args:
            images: Batch of images [B, 3, 224, 224].
            text_features: TF-IDF features [C, text_dim].
            return_embeddings: If True, return ``(scores, g, f)`` instead of
                just scores.

        Returns:
            FC scores [B, C], or tuple ``(scores, g, f)`` if
            ``return_embeddings=True``.
        """
        # gv(x): image encoder → g ∈ R^{B×k}
        # f_t(t_c): text encoder → f ∈ R^{C×k}
        # Score: g @ f.T ∈ R^{B×C}
        g = self.image_encoder(images)
        f = self.text_encoder(text_features)
        scores = g @ f.T
        if return_embeddings:
            return scores, g, f
        return scores

    def _forward_conv(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute conv branch scores (Sec 3.3).

        Paper: Predict K' filters (3x3) from text, convolve with g'_v(images),
        then global average pool.

        Args:
            images: Batch of images [B, 3, 224, 224].
            text_features: TF-IDF features [C, text_dim].

        Returns:
            Conv scores [B, C] from conv(predicted_filters, g'_v(images))
            with global average pooling.
        """
        # g'_v(x): conv_reduce(shared features) → [B, K', H, W]
        # f'_t(t_h): predict K' filters of size 3x3 → [C, K', 3, 3]
        # Conv: conv2d(g'_v, predicted_filters) → [B, C, H, W]
        # Global avg pool: flatten + mean → [B, C]
        conv_feat = self.image_encoder.forward_conv_feature(images)  # [B, K', H, W]
        _, hidden = self.text_encoder.forward_with_hidden(text_features)  # [C, 300]
        filters = self.conv_weight_predictor(hidden)               # [C, K', 3, 3]
        out = F.conv2d(conv_feat, filters, padding=1)               # [B, C, H, W]
        return out.flatten(2).mean(2)                               # [B, C]
