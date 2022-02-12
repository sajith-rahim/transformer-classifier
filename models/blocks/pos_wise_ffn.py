import torch
import torch.nn as nn

from base import BaseModel


class PositionWiseFFN(BaseModel):
    r"""
    Position wise Feed Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2

    In the paper;
    The dimensionality of input and output is dmodel = 512,
    and the inner-layer has dimensionality dff = 2048.
    """

    def __init__(self, dim: int, hidden_size: int = 2048, dropout_rate: float = 0.1):
        r"""
        :param dim: model dimension
        :param hidden_size: ffn hidden layer size
        :param dropout_rate: drop out ratio
        """
        super(PositionWiseFFN, self).__init__()

        self.W_1 = nn.Linear(dim, hidden_size)
        self.W_2 = nn.Linear(hidden_size, dim)

        self.layer_norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        r"""
        :param x: (batch_size, q_len, dim)
        """
        out = self.W_1(x)
        out= self.relu(out)

        out = self.W_2(out)
        out = self.dropout(out)

        out = out + x  # skip
        out = self.layer_norm(out)

        return out
