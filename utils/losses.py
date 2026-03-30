"""
Losses from Ba et al. ICCV 2015 Sec 4: BCE (Eq. 6), Hinge (Eq. 7), Euclidean (Sec 4.2.1).
- Eq. 6: L = sum_ij [ I_ij log sigma(yhat) + (1-I_ij) log(1-sigma(yhat)) ], I in {0,1}, sigma sigmoid.
- Eq. 7: L = sum_ij max(0, margin - I_ij * yhat), I in {+1,-1}, margin = 1.
- Euclidean (Sec 4.2.1): hinge loss + L2 on embeddings.
  Paper: -1/2||a-b||^2 = a'b - 1/2||a||^2 - 1/2||b||^2; hinge + L2(w_c, g_v)
  is equivalent to minimizing Euclidean distance ||w_c - g_v(x)||^2.
Minibatch per paper: sum only over (i,j) for images and classes in the batch; train_step passes [B,U].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy loss (Paper Eq. 6).

    Paper Eq. 6:
        L = Σ_ij [ I_ij log σ(yhat_ij) + (1-I_ij) log(1-σ(yhat_ij)) ]

    where σ is the sigmoid function and I_ij ∈ {0,1}.

    Note: Paper uses SUM (not mean) over all i,j in batch.

    Args:
        scores: Predicted logits [N, C].
        targets: Binary targets [N, C] with values in {0, 1}.

    Returns:
        Scalar loss (sum over all elements).
    """
    return F.binary_cross_entropy_with_logits(scores, targets.float(), reduction="sum")


def hinge_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge loss (Paper Eq. 7).

    Paper Eq. 7:
        L = Σ_ij max(0, margin - I_ij * y_ij)

    where I_ij ∈ {+1,-1} and margin = 1.

    Note: Paper uses SUM (not mean) over all i,j in batch.

    Args:
        scores: Predicted logits [N, C].
        targets: Binary targets [N, C] with values in {+1, -1}.
        margin: Margin for hinge loss (default 1.0).

    Returns:
        Scalar loss (sum over all elements).
    """
    return F.relu(margin - targets * scores).sum()


def euclidean_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Direct Euclidean distance loss (Paper Sec 4.2.1).

    Paper motivation: minimize ||w_c - g_v(x)||² for correct (image, class)
    pairs, maximize distance for incorrect pairs.

    Direct implementation using pairwise Euclidean distances, avoiding the
    hinge+L2 decomposition which suffers from parasitic L2 penalty on
    satisfied pairs (pushing embeddings to zero).

    Contrastive formulation:
        Positive pairs (I=+1): L = ||g_i - f_j||²  (pull together)
        Negative pairs (I=-1): L = max(0, margin - ||g_i - f_j||)²  (push apart)

    Args:
        image_emb: Image embeddings [B, k] from g_v(·).
        text_emb: Batch-class text embeddings [U, k] from f_t(·).
        targets: Binary targets [B, U] with values in {+1, -1}.
        margin: Distance margin for negative pairs (default 1.0).

    Returns:
        Scalar loss (sum over all pairs).
    """
    # Pairwise squared Euclidean distances [B, U]
    # ||g_i - f_j||² = ||g_i||² - 2*g_i^T*f_j + ||f_j||²
    dist_sq = (image_emb.pow(2).sum(dim=1, keepdim=True)   # [B, 1]
               - 2 * (image_emb @ text_emb.T)               # [B, U]
               + text_emb.pow(2).sum(dim=1).unsqueeze(0))    # [1, U]
    dist_sq = dist_sq.clamp(min=0.0)  # numerical safety

    pos_mask = (targets > 0).float()   # [B, U]
    neg_mask = (targets < 0).float()   # [B, U]

    # Positive: minimize squared distance (pull correct class closer)
    loss_pos = (dist_sq * pos_mask).sum()

    # Negative: push apart if closer than margin
    dist = dist_sq.sqrt()
    loss_neg = (F.relu(margin - dist).pow(2) * neg_mask).sum()

    return loss_pos + loss_neg


def get_criterion(loss_name: str = "bce", hinge_margin: float = 1.0):
    """Get loss function by name.

    Args:
        loss_name: Type of loss - 'bce', 'hinge', or 'euclidean'.
        hinge_margin: Margin for hinge loss (default 1.0).

    Returns:
        Loss function that takes (scores, targets) and returns scalar loss.
        For 'euclidean', returns hinge criterion (L2 added separately in train_step).

    Raises:
        ValueError: If loss_name is not recognized.
    """
    if loss_name.lower() == "bce":
        return lambda s, t: bce_loss(s, t)
    if loss_name.lower() == "hinge":
        return lambda s, t: hinge_loss(s, t, margin=hinge_margin)
    if loss_name.lower() == "euclidean":
        # Direct Euclidean distance loss; handled entirely in train_step.
        return None
    raise ValueError(f"Unknown loss: {loss_name}")


def clip_contrastive_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """CLIP-style symmetric contrastive loss.

    Pulls together matched (image, text) pairs and pushes apart mismatched
    pairs within the mini-batch.  Uses L2-normalised embeddings and a
    learnable-equivalent temperature parameter.

    Note: Returns a sum (not mean) over the batch to be consistent with the
    other loss functions in this module.

    Args:
        image_emb: Image embeddings of shape ``[B, k]``.
        text_emb: Text embeddings of shape ``[B, k]`` (one per image,
            corresponding to the ground-truth class).
        temperature: Softmax temperature (default 0.07, as in CLIP).

    Returns:
        Scalar contrastive loss (sum over the batch).
    """
    image_emb = F.normalize(image_emb, dim=1)
    text_emb = F.normalize(text_emb, dim=1)

    logits = image_emb @ text_emb.T / temperature  # [B, B]
    labels = torch.arange(image_emb.size(0), device=image_emb.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    # Average direction losses, then scale to sum for batch-size consistency
    return (loss_i2t + loss_t2i) / 2 * image_emb.size(0)


def center_alignment_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
) -> torch.Tensor:
    """Center alignment loss between image and text embeddings.

    Forces the mean of image embeddings to match the mean of text embeddings
    in the joint embedding space. This helps align the two modalities'
    distributions.

    L_align = ||μ_g - μ_f||²

    where μ_g = mean(image_emb, dim=0) and μ_f = mean(text_emb, dim=0).

    Args:
        image_emb: Image embeddings of shape ``[B, k]``.
        text_emb: Text embeddings of shape ``[B, k]`` (one per image,
            corresponding to the ground-truth class).

    Returns:
        Scalar MSE loss between the two mean vectors.
    """
    return F.mse_loss(image_emb.mean(0), text_emb.mean(0))

def embedding_mse_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Direct embedding MSE loss (Variant 2 Euclidean approach).

    Computes MSE between image embeddings and text embeddings directly,
    rather than between scores and binary targets. This is a regression/
    embedding alignment objective, not a classification objective.

    L_emb = MSE(g, f) where g is image embedding and f is text embedding.

    Note: This is fundamentally different from euclidean_loss(), which
    computes MSE(scores, targets) for classification.

    Args:
        image_emb: Image embeddings of shape ``[B, k]``.
        text_emb: Text embeddings of shape ``[B, k]`` (one per image).
        reduction: Reduction method - 'mean', 'sum', or 'none' (default: 'mean').

    Returns:
        Scalar MSE loss between image and text embeddings.
    """
    return F.mse_loss(image_emb, text_emb, reduction=reduction)