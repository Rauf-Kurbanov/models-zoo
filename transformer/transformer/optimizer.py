from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from transformer.architecture import EncoderDecoder


class NoamOpt:
    """Optim wrapper that implements rate."""

    optimizer: Optimizer
    _step: int
    warmup: int
    factor: float
    model_size: int
    _rate: float

    def __init__(self, model_size: int, factor: float, warmup: int, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0.0

    def step(self) -> None:
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model: EncoderDecoder) -> NoamOpt:
    return NoamOpt(
        model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )


def visualize_lr_settings() -> None:  # TODO move
    """Three settings of the lrate hyperparameters."""

    opts = [NoamOpt(512, 1, 4000, None), NoamOpt(512, 1, 8000, None), NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
