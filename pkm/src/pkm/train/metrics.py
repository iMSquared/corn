#!/usr/bin/env python3

import torch as th
nn = th.nn
F = nn.functional


def sigmoid_inverse(x: th.Tensor) -> th.Tensor:
    """ Inverse of the sigmoid function. """
    x = th.as_tensor(x)
    return th.log(x / (1 - x))


class IoU(nn.Module):
    """
    Pointwise IoU (intersection over union) metric.
    """

    def __init__(self,
                 min_logit: float = 0.0,
                 eps: float = 1e-6):
        """
        Args:
            min_logit: Minimum logit value to be considered occupied.
                Usually computed via inverse_sigmoid(x).
        """
        super().__init__()
        self.min_logit = min_logit
        self.eps = eps

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        """
        Args:
            input: prediction (logits, >=`min_logit` is considered positive.)
            target: target (binary or float, >=0.5 is considered positive.)

        Returns:
            iou: The computed IoU metric.
        """
        x = (input >= self.min_logit)  # bool
        y = (target >= (0.5 + self.eps))  # bool
        ixn = th.logical_and(x, y)
        uxn = th.logical_or(x, y)
        iou = th.div(th.count_nonzero(ixn), th.count_nonzero(uxn))
        return iou
