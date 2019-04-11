import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0, reduce=True):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=reduce)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone().detach()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and mask.nelement() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, true_dist)


def example_1():
    # Example of label smoothing.
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(predict.log(),
             torch.LongTensor([2, 1, 0]))
    print(crit.true_dist)
    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist)
    plt.show()


def example_2_bad():
    crit = LabelSmoothing(5, 0, 0.1, reduce=False)

    def loss(x):
        d = x + 3 * 1
        predict = np.column_stack([np.zeros_like(x),
                                  x / d, 1 / d, 1 / d, 1 / d])
        predict = torch.Tensor(predict)
        target = torch.ones(x.shape, dtype=torch.long)
        return crit(predict.log(), target)

    xs = np.arange(1, 100)
    probs = xs / (xs + 3)
    print(probs)
    ys = loss(xs)[:, 1]
    plt.plot(xs, ys.numpy())
    plt.plot(xs, probs)
    plt.show()


def example_3():
    crit = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                     ])
        # print(predict)
        return crit(predict.log(),
                    torch.LongTensor([1])).data[0]

    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()


if __name__ == '__main__':
    example_3()
