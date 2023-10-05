#!/usr/bin/env python3

from typing import Iterable
from pathlib import Path
from tqdm.auto import tqdm
import itertools
import numpy as np

from pkm.data.util import split_files


def glob_path(root: str,
              pattern: str,
              max_count: int):
    if max_count > 0:
        g = Path(root).rglob(pattern)
        g = list(tqdm(itertools.islice(g, max_count), desc='glob'))
        data_path = np.array(g)
    else:
        g = tqdm(Path(root).rglob(pattern))
        g = list(sorted(g))
        data_path = np.array(g)
    return data_path


def tv_split(
        paths: Iterable[str],
        seed: int,
        valid_ratio: float,
        split: str):
    (valid_data, train_data) = split_files(paths,
                                           np.random.default_rng(seed),
                                           fraction=valid_ratio)
    if split == 'train':
        out = train_data
    elif split == 'valid':
        out = valid_data
    else:
        raise ValueError(F'Unknown split = {split}')
    return out
