import copy
import time
from typing import Callable, Iterator

import torch
from torch import Tensor, nn, tensor

from transformer.architecture import EncoderDecoder, Generator
from transformer.attention import MultiHeadedAttention
from transformer.decoder import Decoder, DecoderLayer
from transformer.embeddings import Embeddings, PositionalEncoding
from transformer.encoder import Encoder, EncoderLayer
from transformer.utils import Batch, PositionwiseFeedForward


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    ).to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def run_epoch(
    data_iter: Iterator[Batch], model: EncoderDecoder, loss_compute: Callable[[Tensor, Tensor, Tensor], Tensor]
) -> Tensor:  # TODO SimpleLossCompute really ?
    start = time.time()
    total_tokens = tensor(0)
    total_loss = tensor(0.0)
    tokens = tensor(0)
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            ntokens = batch.ntokens.float()
            print(f"Epoch Step: {i} Loss: {loss / ntokens}" f" Tokens per Sec: {tokens.float() / elapsed}")
            start = time.time()
            tokens = 0
    assert total_tokens > 0
    res = total_loss / total_tokens.float()
    return res  # TODO


# global max_src_in_batch, max_tgt_in_batch


# def batch_size_fn(new, count, sofar):
#     """Keep augmenting batch and calculate total number of tokens + padding."""
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch, len(new.src))
#     max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
#     src_elements = count * max_src_in_batch
#     tgt_elements = count * max_tgt_in_batch
#     return max(src_elements, tgt_elements)
