#!/usr/bin/env python3

from dataclasses import dataclass
from pkm.util.config import ConfigBase
from pathlib import Path
import pickle
from typing import (List, Tuple, Dict, Callable, Optional)
from tqdm.auto import tqdm
import torch as th


class CachedDataset(th.utils.data.Dataset):
    """Class that memoizes a subset of the given dataset, and retrieves them
    during runtime from the filesystem cache."""
    @dataclass
    class Config(ConfigBase):
        cache_dir: str = '~/.cache/pkm/'
        # NOTE: How many samples to fetch.
        num_samples: Optional[int] = None
        force_rebuild: bool = False

    def __init__(self,
                 cfg: Config,
                 dataset_fn: Callable[[None], th.utils.data.Dataset],
                 name: str,
                 transform=None):
        self.cfg = cfg
        self.dataset_fn = dataset_fn
        self.name = name
        self.data = self._build()
        self.xfm = transform

    def _build(self):
        samples_cache = F'{self.cfg.cache_dir}/{self.name}.pkl'
        samples_path = Path(samples_cache).expanduser()

        if (self.cfg.force_rebuild) or (not samples_path.exists()):
            # Download data from scratch ...
            dataset = self.dataset_fn()
            if self.cfg.num_samples is None:
                num_samples = len(dataset)
            else:
                num_samples = self.cfg.num_samples
            samples = []
            for i, data in tqdm(
                    enumerate(dataset),
                    total=num_samples):
                samples.append(data)
                if i >= num_samples:
                    break
            samples_path.parent.mkdir(
                parents=True, exist_ok=True)
            with open(str(samples_path), 'wb') as f:
                pickle.dump(samples, f)
        else:
            with open(str(samples_path), 'rb') as f:
                samples = pickle.load(f)
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        out = self.data[index % len(self.data)]
        if self.xfm is not None:
            out = self.xfm(out)
        return out
