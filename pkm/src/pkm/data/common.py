#!/usr/bin/env python3

from torch.utils.data import default_collate
from typing import Dict, FrozenSet, Iterable, Tuple
import torch as th
import numpy as np


def collate_dicts(
        batch: Iterable[Dict[str, th.Tensor]],
        select: FrozenSet[str],
        passthrough: FrozenSet[str]):
    """ Collate dictionaries.

    Wrapper around default_collate
    for lists of dictionaries, which allows:
    * selecting subset of tensors.
    * outputting lists without stacking.
    """
    elem = batch[0]
    out = {}
    for key in elem:
        if key not in select:
            continue
        value = [d[key] for d in batch]
        if key in passthrough:
            # Try to avoid the issue:
            # https://github.com/pytorch/pytorch/issues/13246
            # which seems to cause memory leaks.
            # value = np.array(value, dtype=object)
            pass
        else:
            value = default_collate(value)
        out[key] = value
    return out
