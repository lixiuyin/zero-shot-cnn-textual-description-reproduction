"""Multi-granularity SBERT: per-sentence pooling for class descriptions."""
import re
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


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


def _truncate_sentence(text: str, max_chars: int = 300) -> str:
    text = _normalize_whitespace(text)
    return text if len(text) <= max_chars else text[:max_chars]


def texts_to_sbert_multi(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
    max_chars_per_sentence: int = 300,
) -> np.ndarray:
    """Multi-granularity SBERT encoding: split text into sentences, embed each,
    then mean-pool and L2-normalize per class.

    Args:
        texts: List of class description strings.
        model_name: sentence-transformers model name.
        normalize_embeddings: Whether to L2-normalize sentence embeddings.
        max_chars_per_sentence: Character cap per sentence before encoding.

    Returns:
        L2-normalized embeddings of shape [N, 384].
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)

    class_embeddings = []
    for text in texts:
        sentences = [_truncate_sentence(s, max_chars_per_sentence) for s in _split_into_sentences(text)]
        sent_embeds = _model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )  # [num_sent, 384]
        class_embed = sent_embeds.mean(axis=0)
        norm = np.linalg.norm(class_embed)
        if norm > 0:
            class_embed = class_embed / norm
        class_embeddings.append(class_embed.astype(np.float32))

    return np.stack(class_embeddings, axis=0).astype(np.float32)
