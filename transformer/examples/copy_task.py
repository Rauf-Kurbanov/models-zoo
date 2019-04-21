from typing import Iterator, Optional

import numpy as np
import torch
from torch import LongTensor, Tensor, nn
from tqdm.auto import trange

from transformer.architecture import EncoderDecoder, Generator
from transformer.optimizer import NoamOpt
from transformer.regularisation import LabelSmoothing
from transformer.train import make_model, run_epoch
from transformer.utils import Batch, LongTensorType, subsequent_mask


def data_gen(
    v: int, batch: int, nbatches: int, device: torch.device = torch.device("cpu")
) -> Iterator[Batch]:  # TODO bad name
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = np.random.randint(1, v, size=(batch, 10))
        data[:, 0] = 1
        src: LongTensorType = torch.from_numpy(data)
        tgt: LongTensorType = torch.from_numpy(data)
        src, tgt = src.to(device), tgt.to(device)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    generator: Generator
    criterion: nn.Module
    opt: Optional[NoamOpt]

    def __init__(
        self, generator: Generator, criterion: nn.Module, opt: Optional[NoamOpt] = None
    ) -> None:  # TODO more general type for opt
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: Tensor, y: Tensor, norm: Tensor) -> Tensor:
        x = self.generator(x)
        lhs = x.contiguous().view(-1, x.size(-1))
        rhs = y.contiguous().view(-1)
        loss = self.criterion(lhs, rhs)
        norm = norm.float()  # TODO why did I need to add cast?
        assert norm != 0
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm  # TODO check if tensor


def greedy_decode(
    model: EncoderDecoder, src: LongTensor, src_mask: LongTensor, max_len: int, start_symbol: int
) -> Tensor:
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == "__main__":
    # Train the simple copy task.
    V = 11
    device = torch.device("cuda")
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, n=2, device=device)
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    for epoch in trange(10):
        model.train()
        run_epoch(data_gen(V, 30, 20, device=device), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5, device=device), model, SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
