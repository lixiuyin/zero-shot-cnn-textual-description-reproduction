"""
Text processing for zero-shot CNN (Ba et al., ICCV 2015).
Paper: one Wikipedia article per class -> TF-IDF (e.g. 9763-d for CUB),
log normalization (sublinear_tf).
Supports results.jsonl: one JSON per line with class_name, wikipedia_text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize_class_name(name: str) -> str:
    """Lowercase, spaces→underscore, strip leading digits/dot."""
    s = name.strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"^[0-9._]*", "", s).strip("_")
    return s


def load_class_texts_from_jsonl(
    jsonl_path: str | Path,
    class_names: list[str] | None = None,
):
    """
    Load Wikipedia text per class from a JSONL file (one JSON per line).
    
    Each line typically contains keys like "class_name" / "class" and
    "wikipedia_text" / "text".
    
    Two usage modes (for backward compatibility):
    
    1) load_class_texts_from_jsonl(path, class_names):
       - Returns a list[str] aligned with the given class_names.
       - Class names are matched via a normalized key (lowercase, spaces/-
         to underscore, strip leading digits and dots).
       - Missing classes get "".
    
    2) load_class_texts_from_jsonl(path):
       - Returns a dict[str, str] mapping normalized class-name keys to
         raw text. This matches older call sites that expect a mapping
         rather than an ordered list.
    """
    jsonl_path = Path(jsonl_path)
    key_to_text: dict[str, str] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cname = obj.get("class_name") or obj.get("class")
            text = obj.get("wikipedia_text") or obj.get("text") or ""
            if cname is not None:
                key_to_text[_normalize_class_name(str(cname))] = str(text)
    if class_names is None:
        return key_to_text
    return [key_to_text.get(_normalize_class_name(c), "") for c in class_names]


def load_class_texts_from_dir(
    text_dir: str | Path,
    class_names: list[str],
    extension: str = ".txt",
    encoding: str = "utf-8",
) -> list[str]:
    """
    Load one text file per class, in class order (CUB: class_names from
    load_cub_200_2011). File names: text_dir / (class_name + extension),
    e.g. 001.Black_footed_Albatross.txt. Fallback: text_dir / (index+1).txt
    if class-named file missing. Returns list[str].
    """
    text_dir = Path(text_dir)
    texts = []
    for i, name in enumerate(class_names):
        path = text_dir / (name.strip() + extension)
        if not path.exists():
            path = text_dir / f"{i + 1:03d}{extension}"
        if not path.exists():
            path = text_dir / f"{i + 1}{extension}"
        if not path.exists():
            raise FileNotFoundError(
                f"No text file for class {i + 1} ({name!r}), tried {path}"
            )
        texts.append(path.read_text(encoding=encoding))
    return texts


def build_tfidf(
    texts: list[str],
    max_features: int = 9763,
    sublinear_tf: bool = True,
    **kwargs,
) -> TfidfVectorizer:
    """Build TF-IDF vectorizer.

    Paper specifies log normalization (sublinear_tf=True).

    Args:
        texts: List of text documents (one per class).
        max_features: Maximum number of features (default 9763 for CUB).
        sublinear_tf: Whether to use sublinear TF scaling (default True).
        **kwargs: Additional arguments passed to TfidfVectorizer.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        **kwargs,
    )


def texts_to_tfidf(
    texts: list[str],
    vectorizer: TfidfVectorizer | None = None,
    max_features: int = 9763,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Convert texts to TF-IDF matrix with exact dimension.

    Ensures output is exactly max_features wide by padding with zeros if necessary,
    as required by Ba et al. ICCV 2015.

    Args:
        texts: List of text documents (one per class).
        vectorizer: Pre-fitted TfidfVectorizer. If None, creates and fits new one.
        max_features: Exact output dimension (default 9763 for CUB).

    Returns:
        A tuple containing:
            - TF-IDF matrix [n_texts, max_features]
            - Fitted TfidfVectorizer
    """
    if vectorizer is None:
        vectorizer = build_tfidf(texts, max_features=max_features)
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)

    arr = X.toarray().astype(np.float32)
    # If the vocabulary is smaller than max_features, pad with zeros
    if arr.shape[1] < max_features:
        padding = np.zeros((arr.shape[0], max_features - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, padding])
    elif arr.shape[1] > max_features:
        arr = arr[:, :max_features]

    return arr, vectorizer
