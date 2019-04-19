from torch import Tensor, nn
from torch.nn import LayerNorm

from transformer.attention import MultiHeadedAttention
from transformer.utils import PositionwiseFeedForward, SublayerConnection, clones


class EncoderLayer(nn.Module):
    def __init__(
        self, size: int, self_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # TODO mask type
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, n: int) -> None:  # TODO type
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # TODO mask type
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
