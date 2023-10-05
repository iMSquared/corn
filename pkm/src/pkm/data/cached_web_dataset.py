#!/usr/bin/env python3

from dataclasses import dataclass
from pkm.util.config import ConfigBase
from pathlib import Path
import pickle
from typing import (List, Tuple, Dict, Callable, Optional, Union,
                    Iterable, Any)
from tqdm.auto import tqdm
import torch as th
import json
import webdataset as wds
from os import PathLike
from functools import partial


def _build_wds(
        cache_dir: Union[str, PathLike],
        dataset_fn: Callable[[None], th.utils.data.Dataset],
        num_samples: Optional[int] = None,
        shard_size: Union[int, None, str] = None,
        max_size: Optional[int] = None,
        ext: Optional[Dict[str, str]] = None
):
    """ Optionally build webdataset shards. """
    if ext is None:
        ext = {}

    # Download data from scratch ...
    dataset = dataset_fn()
    if num_samples is None:
        try:
            num_samples = len(dataset)
        except TypeError:
            num_samples = None
    else:
        num_samples = num_samples

    # Automatically configure the number of samples
    # stored in a single shard.
    # if shard_size is None:
    #     shard_size = max(1, num_samples // 128)
    kwds = {}
    if isinstance(shard_size, str) and shard_size == 'auto':
        kwds['maxcount'] = max(1, num_samples // 128)

    if max_size is not None:
        kwds['maxsize'] = max_size

    # Write shards.
    cache_dir.mkdir(
        parents=True, exist_ok=True)
    keys: Tuple[str, ...] = None
    with wds.ShardWriter(
            str(cache_dir / 'shard-%06d.tar'),
            **kwds) as fp:
        count: int = 0
        for data in tqdm(dataset, leave=False, desc='wds.shard'):
            # assert isinstance(data, Dict[str, np.ndarray])
            fp.write({
                '__key__': F'{count:06d}',
                **{k + ext.get(k, '.pyd'): v for
                    k, v in data.items()}
            })
            if keys is None:
                keys = list(data.keys())
            count += 1
            if (num_samples is not None) and count >= num_samples:
                break

    # Write metadata.
    with open(cache_dir / 'metadata.json', 'w') as fp:
        json.dump({
            'num_data': count,
            'keys': keys
        }, fp)


def to_dict(keys: Iterable[str], x):
    """ tuple() -> dict() """
    return {k: v for k, v in zip(keys, x)}


def as_wds(
        cache_dir: Union[str, PathLike],
        dataset_fn: Callable[[None], th.utils.data.Dataset],
        num_samples: Optional[int] = None,
        shard_size: Optional[int] = None,
        shuffle: bool = False,
        ext: Optional[Dict[str, str]] = None,
        batch_size: int = None,
        as_dict: bool = True,
) -> wds.WebDataset:
    """
    Convert dataset into a WebDataset format, cached in
    local filesystem.

    Args:
        cache_dir: Directly to store/load shards.
        dataset_fn: Function to instantiate the dataset with,
            while caching shards.
        num_samples: Number of samples to load/store.
        shard_size: Number of samples per each shard.
        shuffle: Whether to trigger shardshuffle+shuffle.

    Returns:
        wds.WebDataset version of dataset_fn().
    """
    if ext is None:
        ext = {}

    cache_dir = Path(cache_dir).expanduser()
    if not (cache_dir / 'metadata.json').exists():
        _build_wds(cache_dir, dataset_fn,
                   num_samples, shard_size, ext=ext)
    shards = list(map(str, sorted(cache_dir.glob('shard-*.tar'))))
    with open(cache_dir / 'metadata.json', 'r') as fp:
        metadata = json.load(fp)
    num_data: int = metadata['num_data']
    keys: Tuple[str, ...] = metadata['keys']

    _to_dict = partial(to_dict, keys)

    if shuffle:
        # TODO: `shuffle` size should be
        # determined based on ... something
        dataset = (wds.WebDataset(
            shards, shardshuffle=True)
            .with_length(num_data)
            .shuffle(1024)
            .decode()
            .to_tuple(*[k + ext.get(k, '.pyd') for k in keys])
        )
        if batch_size is not None:
            dataset = dataset.batched(batch_size)
        if as_dict:
            dataset = dataset.map(_to_dict)
    else:
        dataset = (wds.WebDataset(shards,
                                  shardshuffle=False)
                   .with_length(num_data)
                   .decode()
                   .to_tuple(*[k + ext.get(k, '.pyd') for k in keys]))
        if batch_size is not None:
            dataset = dataset.batched(batch_size)
        if as_dict:
            dataset = dataset.map(_to_dict)

    if as_dict:
        return dataset
    else:
        return (dataset, _to_dict)
