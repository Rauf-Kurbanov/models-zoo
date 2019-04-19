import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    _criterion: nn.Module
    _padding_idx: int
    _confidence: float
    _smoothing: float
    _size: int

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0, reduce: bool = True) -> None:
        super().__init__()
        self._criterion = nn.KLDivLoss(size_average=False, reduce=reduce)
        self._padding_idx = padding_idx
        self._confidence = 1.0 - smoothing
        self._smoothing = smoothing
        self._size = size

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert x.size(1) == self._size
        true_dist = torch.zeros_like(x.data)
        true_dist.fill_(self._smoothing / (self._size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self._confidence)
        true_dist[:, self._padding_idx] = 0
        mask = torch.nonzero(target.data == self._padding_idx)
        if mask.dim() > 0 and mask.nelement() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self._criterion(x, true_dist)


def example_1() -> None:
    # Example of label smoothing.
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]])
    crit(predict.log(), torch.LongTensor([2, 1, 0]))
    print(crit.true_dist)
    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist)
    plt.show()


def example_2() -> None:
    crit = LabelSmoothing(5, 0, 0.1, reduce=False)

    def loss(x: np.ndarray) -> Tensor:
        d = x + 3 * 1
        predict = np.column_stack([np.zeros_like(x), x / d, 1 / d, 1 / d, 1 / d])
        predict = torch.Tensor(predict)
        target = torch.ones(x.shape, dtype=torch.long)
        return crit(predict.log(), target)

    xs = np.arange(1, 100)
    probs = xs / (xs + 3)
    ys = loss(xs)[:, 1]
    plt.plot(xs, ys.numpy())
    plt.plot(xs, probs)
    plt.show()


def example_3() -> None:
    crit = LabelSmoothing(5, 0, 0.3)

    def loss(x: int) -> Tensor:
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
        # print(predict)
        v = crit(predict.log(), torch.LongTensor([1]))
        return v.data[0]

    xs = np.arange(1, 100)
    probs = xs / (xs + 3)
    # plt.plot(xs, probs)
    plt.plot(probs, [loss(x) for x in range(1, 100)])
    plt.show()


if __name__ == "__main__":
    example_2()
