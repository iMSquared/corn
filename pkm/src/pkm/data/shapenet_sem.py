#!/usr/bin/env bash

from pathlib import Path
import numpy as np
import pandas as pd


def load_mug_ids(filename: str):
    path = Path(filename)
    df = pd.read_csv(path)
    df['fullId'] = df['fullId'].apply(lambda s: s.split('.', maxsplit=1)[1])
    df['wnsynset'] = df['wnsynset'].apply(lambda s: (
        s[1:] if (isinstance(s, str) and s.startswith('n')) else s))
    # wordnet.synset('
    # print(len(df[df['wnsynset'] == '3802912']))
    mask = df['category'].str.contains(
        'DrinkingUtensil', case=False)
    mask = mask.fillna(False)
    # 112 objects total
    return df['fullId'][mask].to_numpy()


def main():
    filename = '/opt/datasets/ShapeNetSem/metadata.csv'
    load_mug_ids(filename)


if __name__ == '__main__':
    main()
