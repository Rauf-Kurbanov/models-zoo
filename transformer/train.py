import copy
import time

from torch import nn

from transformer.architecture import EncoderDecoder, Generator
from transformer.attention import MultiHeadedAttention
from transformer.decoder import Decoder, DecoderLayer
from transformer.embeddings import PositionalEncoding, Embeddings
from transformer.encoder import Encoder, EncoderLayer
from transformer.utils import PositionwiseFeedForward


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            assert elapsed != 0
            ntokens = batch.ntokens.float()
            assert ntokens != 0
            print(f"Epoch Step: {i} Loss: {loss / ntokens}"
                  f" Tokens per Sec: {tokens / elapsed}")
            start = time.time()
            tokens = 0
    # return total_loss / total_tokens.float()  # TODO
    assert total_tokens > 0
    print("total_loss", total_loss)
    print("total_tokens", total_tokens)
    res = total_loss / total_tokens.float()
    return res  # TODO


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
