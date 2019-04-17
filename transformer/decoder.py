from torch import nn
from torch.nn import LayerNorm

from transformer.utils import clones, SublayerConnection


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  # TODO what is layer?

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        normalised = self.norm(x)
        return normalised


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, tgt_mask))
        x = self.sublayer[1](x, lambda t: self.src_attn(t, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
