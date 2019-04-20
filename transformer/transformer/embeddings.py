import math

import torch
from torch import Tensor, nn


class Embeddings(nn.Module):
    lut: nn.Embedding
    d_model: int

    def __init__(self, d_model: int, vocab: int) -> None:
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):  # TODO read again
    dropout: nn.Dropout
    pe: Tensor

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        lhs = torch.arange(0, d_model, 2, dtype=torch.float)
        rhs = -(math.log(10000.0) / d_model)
        div_term = torch.exp(lhs * rhs)
        pe[:, 0::2] = torch.sin(position * div_term)  # TODO Tensor -> FloatTensor
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)].detach()  # TODO check (Rauf 22.02.19)
        return self.dropout(x)
