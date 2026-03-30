"""
Predict convolutional classifier weights from text (Ba et al. ICCV 2015 Sec 3.3).
Paper: from 300-unit hidden layer of ft(·) predict K' filters of size 3x3; K'=5.
Score: global average pool over conv(predicted_filters, g'_v(a)).
"""
import torch
import torch.nn as nn


class ConvWeightPredictor(nn.Module):
    """Predict convolutional filters from text hidden representation.

    Paper Sec 3.3: Predicts K' filters of size 3x3 from the 300-d hidden layer
    of the text encoder ft(·). Default: K'=5 filters.

    Attributes:
        k_prime: Number of filters to predict (K' in paper).
        filter_size: Size of each filter (default 3x3).
        fc: Linear layer that maps hidden_dim to filter weights.
    """

    def __init__(self, hidden_dim: int = 300, k_prime: int = 5, filter_size: int = 3):
        """Initialize the ConvWeightPredictor.

        Args:
            hidden_dim: Dimension of text hidden representation (default 300).
            k_prime: Number of filters to predict (K' in paper, default 5).
            filter_size: Size of each filter (default 3 for 3x3).
        """
        super().__init__()
        self.k_prime = k_prime
        self.filter_size = filter_size
        self.out_dim = k_prime * filter_size * filter_size
        self.fc = nn.Linear(hidden_dim, self.out_dim)

        # Small initialization: predicts conv filters, should start small
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """Predict conv filters from text hidden representation.

        Args:
            text_hidden: Text hidden features of shape [*, 300].

        Returns:
            Predicted filters of shape [*, K', 3, 3].
        """
        out = self.fc(text_hidden)
        return out.view(*out.shape[:-1], self.k_prime, self.filter_size, self.filter_size)
