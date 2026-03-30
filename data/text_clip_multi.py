"""Multi-granularity CLIP text encoder: per-sentence encoding with mean-pooling.

Standard CLIP truncates at 77 tokens (~50-60 words), discarding ~85% of a
typical Wikipedia article (~400 words).  This module splits each article into
sentences, encodes each sentence independently with CLIP ViT-B/32, and
mean-pools the sentence embeddings per class.  The result captures far more
of the article's content while preserving CLIP's vision-aligned text space.
"""
import re

import numpy as np
import torch

_device = None
_model = None
_tokenizer = None

_MODEL_NAME = "openai/clip-vit-base-patch32"


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _split_into_sentences(text: str) -> list[str]:
    """Coarse sentence splitting on . ! ? ; and newlines."""
    text = _normalize_whitespace(text)
    if not text:
        return [""]
    parts = re.split(r'(?<=[\.\!\?\;])\s+|\n+', text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [text]


def _encode_sentences(sentences: list[str]) -> np.ndarray:
    """Encode a list of sentences with CLIP, returning L2-normalized [S, 512]."""
    global _model, _device, _tokenizer

    if _model is None:
        from transformers import CLIPModel, CLIPTokenizer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = CLIPTokenizer.from_pretrained(_MODEL_NAME)
        _model = CLIPModel.from_pretrained(_MODEL_NAME).to(_device).eval()

    with torch.no_grad():
        inputs = _tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        text_out = _model.text_model(**inputs)
        features = _model.text_projection(text_out.pooler_output)  # [S, 512]
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype(np.float32)


def texts_to_clip_multi(
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Multi-granularity CLIP encoding: split → encode per-sentence → mean-pool.

    Args:
        texts: List of class description strings (one per class).
        batch_size: Max sentences per forward pass to limit GPU memory.

    Returns:
        L2-normalized embeddings of shape [N, 512].
    """
    class_embeddings = []
    for text in texts:
        sentences = _split_into_sentences(text)

        # Encode in batches to avoid OOM on long articles
        all_embeds = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            all_embeds.append(_encode_sentences(batch))
        sent_embeds = np.concatenate(all_embeds, axis=0)  # [num_sent, 512]

        # Mean-pool + L2-normalize
        class_embed = sent_embeds.mean(axis=0)
        norm = np.linalg.norm(class_embed)
        if norm > 0:
            class_embed = class_embed / norm
        class_embeddings.append(class_embed.astype(np.float32))

    return np.stack(class_embeddings, axis=0).astype(np.float32)
