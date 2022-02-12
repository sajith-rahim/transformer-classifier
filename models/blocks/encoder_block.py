from typing import Optional
import torch

from base import BaseModel
from models.blocks.attention import MultiHeadScaledDotProductAttention
from models.blocks.pos_wise_ffn import PositionWiseFFN


class TransformerEncoderBlock(BaseModel):
    r"""
    Transformer Encoder Block.
    """
    def __init__(self, dim:int, n_heads:int, ffn_hidden_layer_size:int = 2048, dropout_rate:float = 0.1):

        super(TransformerEncoderBlock, self).__init__()

        self.attn = MultiHeadScaledDotProductAttention(dim, n_heads, dropout_rate)
        self.feed_forward = PositionWiseFFN(dim, ffn_hidden_layer_size, dropout_rate)

    def forward(self, x:torch.Tensor, mask: Optional[torch.Tensor] = None):
        out = self.attn(x, mask=mask)  # (batch_size, q_len, dim)
        out = self.feed_forward(out)  # (batch_size, q_len, dim)
        return out
