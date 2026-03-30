"""Model definitions: text/image encoders, conv weight predictor, zero-shot model (fc / conv / fc+conv).
"""

from .zero_shot_model import ZeroShotModel
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .weight_predictor import ConvWeightPredictor


__all__ = [
    "ZeroShotModel",
    "ImageEncoder",
    "TextEncoder",
    "ConvWeightPredictor"
]
