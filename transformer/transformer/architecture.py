import torch.nn.functional as F  # TODO rename file
from torch import Tensor, nn

from transformer.decoder import Decoder
from transformer.encoder import Encoder


class Generator(nn.Module):
    proj: nn.Linear

    def __init__(self, d_model: int, vocab: int) -> None:  # TODO bad names
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    _encoder: Encoder
    _decoder: Decoder
    src_embed: nn.Module
    _tgt_embed: nn.Module
    generator: Generator

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        # TODO type for embeds
        tgt_embed: nn.Module,
        generator: Generator,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self.src_embed = src_embed
        self._tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, src_mask, tgt, tgt_mask)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self._encoder(self.src_embed(src), src_mask)

    def decode(self, memory: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        embedded = self._tgt_embed(tgt)
        decoded = self._decoder(embedded, memory, src_mask, tgt_mask)
        return decoded
