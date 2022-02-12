import copy

import torch
import torch.nn as nn



from base import BaseModel
from models.blocks.encoder_block import TransformerEncoderBlock
from models.blocks.positional_encoding import PositionalEncoding


class TransformerClassifier(BaseModel):
    r"""

    """

    def __init__(
            self,
            n_classes: int,
            vocab_size: int,
            embeddings: torch.Tensor,
            dim: torch.Tensor,
            q_len: int,
            ffn_hidden_layer_size: int,
            n_heads: int,
            n_encoder: int,
            classfifier_hidden_layer_size: int = 2048,
            dropout: float = 0.5):
        r"""
        :param n_classes:
        :param vocab_size:
        :param embeddings:
        :param dim:
        :param q_len: padded query length
        :param hidden_size:
        :param n_heads:
        :param n_encoder:
        :param dropout:
        """
        super(TransformerClassifier, self).__init__()

        # Embedding
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.set_embeddings(embeddings, fine_tune=True)

        # PE
        self.postional_encoding = PositionalEncoding(dim, q_len, dropout)

        # Encoder
        self.encoder = TransformerEncoderBlock(dim, n_heads, ffn_hidden_layer_size, dropout)

        # replicate n_encoder times
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) for _ in range(n_encoder)
        ])

        # Classifier
        self.fc = nn.Linear(q_len * dim, classfifier_hidden_layer_size)
        self.fc2 = nn.Linear(classfifier_hidden_layer_size, n_classes)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune:bool = True) -> None:
        """
        Set weights for embedding layer
        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings
        """
        if embeddings is None:
            # initialize embedding layer uniform
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor):

        # get padding mask
        mask = self.get_padding_mask(text)

        # word embedding
        embeddings = self.embeddings(text)  # (batch_size, q_len, dim)
        # PE
        embeddings = self.postional_encoding(embeddings)

        encoder_out = embeddings
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, mask=mask)  # (batch_size, q_len , dim)

        encoder_out = encoder_out.view(encoder_out.size(0), -1)  # (batch_size, q_len * dim)

        out = self.fc(encoder_out)
        out = self.fc2(out)  # (batch_size, n_classes)

        return out

    @staticmethod
    def get_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        r"""

        :param seq:  (batch_size, word_pad_len)
        :param pad_idx: index of '<pad>'
        :return: mask : torch.Tensor (batch_size, 1, q_len)
        """
        mask = (seq != pad_idx).unsqueeze(-2)  # (batch_size, 1, word_pad_len)
        return mask
