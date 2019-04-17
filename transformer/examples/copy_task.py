import torch
import numpy as np

from transformer.optimizer import NoamOpt
from transformer.regularisation import LabelSmoothing
from transformer.train import make_model, run_epoch
from transformer.utils import Batch


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = np.random.randint(1, V, size=(batch, 10))
        data[:, 0] = 1
        src = torch.from_numpy(data)
        tgt = torch.from_numpy(data)
        # src = torch.Tensor(data)
        # tgt = torch.Tensor(data)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
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
        return loss.item() * norm


if __name__ == '__main__':
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        # if not model.training:
        #     model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        print("end of epoch", epoch)
        # model.eval()
        # print(run_epoch(data_gen(V, 30, 5), model,
        #                 SimpleLossCompute(model.generator, criterion, None)))