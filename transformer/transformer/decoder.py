from torch import Tensor, nn
from torch.nn import LayerNorm

from transformer.attention import MultiHeadedAttention
from transformer.utils import PositionwiseFeedForward, SublayerConnection, clones


class DecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        m = memory
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, tgt_mask))
        x = self.sublayer[1](x, lambda t: self.src_attn(t, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, n: int) -> None:
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        normalised = self.norm(x)
        return normalised
