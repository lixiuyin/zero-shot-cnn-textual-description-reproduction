"""CLIP text encoder: encodes class descriptions using CLIP ViT-B/32 via HuggingFace."""
import numpy as np
import torch

_device = None
_model = None
_tokenizer = None

_MODEL_NAME = "openai/clip-vit-base-patch32"


def texts_to_clip(texts: list[str]) -> np.ndarray:
    """Encode texts with CLIP ViT-B/32 text encoder.

    Args:
        texts: List of class description strings.

    Returns:
        L2-normalized embeddings of shape [N, 512].
    """
    global _model, _device, _tokenizer

    if _model is None:
        from transformers import CLIPModel, CLIPTokenizer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = CLIPTokenizer.from_pretrained(_MODEL_NAME)
        _model = CLIPModel.from_pretrained(_MODEL_NAME).to(_device).eval()

    with torch.no_grad():
        inputs = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        # Bypass get_text_features() which returns BaseModelOutputWithPooling
        # in some transformers versions instead of a plain tensor.
        # Replicate what get_text_features does internally: text_model → project → normalize.
        text_out = _model.text_model(**inputs)
        features = _model.text_projection(text_out.pooler_output)  # [N, 512]
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype(np.float32)
