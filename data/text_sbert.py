"""SBERT text encoder: encodes class descriptions with sentence-transformers."""
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def texts_to_sbert(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode texts with a Sentence-BERT model.

    Args:
        texts: List of class description strings.
        model_name: sentence-transformers model name (default: all-MiniLM-L6-v2).

    Returns:
        L2-normalized embeddings of shape [N, 384].
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)

    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)
