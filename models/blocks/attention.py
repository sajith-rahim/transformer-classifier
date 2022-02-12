from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class MultiHeadScaledDotProductAttention(BaseModel):
    r"""
    Multi-head Scaled Dot Product Attention with Mask.
    """

    def __init__(self, dim: int, n_heads: int, dropout_rate: float = 0.1):
        r"""
        :param dim:  output dim of model
        :param n_heads: number of heads
        :param dropout_rate: dropout rate
        """
        super(MultiHeadScaledDotProductAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads

        assert dim % n_heads == 0, "Dimension has to divisible by n_heads iorder to split!"

        # d_k = d_v: Dimensionality of queries and keys
        self.dim_k = dim // n_heads

        self.W_q = nn.Linear(dim, n_heads * self.dim_k)
        self.W_k = nn.Linear(dim, n_heads * self.dim_k)
        self.W_v = nn.Linear(dim, n_heads * self.dim_k)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(dim)

        self.fc = nn.Linear(n_heads * self.dim_k, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, q_len, dim = x.shape

        Q = self.W_q(x)  # (batch_size, q_len, dim) (dim, n_heads * self.dim_k) -> (batch_size, q_len, n_heads * dim_k)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, -1, self.n_heads, self.dim_k)  # (batch_size, q_len, n_heads, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.dim_k)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, q_len, n_heads, dim_k) ->  (batch_size,n_heads, q_len, dim_k)
        K = K.permute(0, 2, 3, 1)  # (batch_size, q_len, n_heads, dim_k) ->  (batch_size,n_heads, dim_k, q_len )

        # for n_heads axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, dim_k)

        # context, att = self.attention(Q, K, V, mask=mask)  # (batch_size, n_heads, word_pad_len, d_k)

        # Attention(Q, K, V ) = softmax(QK^T/√dim_k)V
        # Q·K^T / sqrt(d_k)
        scale_factor = self.dim_k ** 0.5
        attn = torch.matmul((Q / scale_factor), K)  # (batch_size, n_heads, q_len, q_len)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(self.softmax(attn))

        V = V.view(batch_size, -1, self.n_heads, self.dim_k)
        V = V.permute(0, 2, 1, 3)  # (batch_size, q_len, n_heads, dim_k) ->  (batch_size,n_heads, q_len, dim_k )

        vals_ = torch.matmul(attn, V)  # (batch_size, n_heads, q_len, dim_k)

        vals_ = vals_.permute(0, 2, 1, 3).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.dim_k)  # (batch_size, q_len , n_heads * dim_k)
        out = self.fc(vals_)  # (batch_size, q_len , dim)
        out = self.dropout2(out)

        out = out + x  # skip
        out = self.layer_norm(out)
        return out


# Test
if __name__ == "__main__":
    sample = torch.randn(1, 5, 512)
    attn = MultiHeadScaledDotProductAttention(512, 4)
    out = attn(sample)
    print(out.shape)
