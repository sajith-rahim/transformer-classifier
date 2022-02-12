import torch
import torch.nn as nn

from base import BaseModel
from utils import get_device


class PositionalEncoding(BaseModel):
    r"""
    PE
    The positional encodings have the same dimension dim,
    as the embeddings.
    P E(pos,2i) = sin(pos/10000^(2i/dmodel))
    P E(pos,2i+1) = cos(pos/10000^(2i/dmodel))
    """

    def __init__(self, dim: int, q_len: int, dropout_rate:float = 0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        exps = [(i * 2.0 / dim) for i in range(dim)]  # (2i/dmodel)

        self.pe = torch.tensor([
            [pos / (10000.0 ** e_term) for e_term in exps]
            for pos in range(q_len)
        ])

        # 2i
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        # 2i + 1
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])


    def forward(self, x:torch.Tensor):
        r"""
        :param x: embeddings
        :return: position encoded embeddings
        """
        x = x + nn.Parameter(self.pe, requires_grad=False).to(get_device())
        x = self.dropout(x)
        return x