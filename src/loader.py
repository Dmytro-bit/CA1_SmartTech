import pickle
import re
from pathlib import Path

import numpy as np


def merge_batches(batches: list[dict]) -> dict:
    if not batches:
        return {}

    keys = batches[0].keys()
    if any(batch.keys() != keys for batch in batches[1:]):
        raise ValueError("all batches must contain the same keys")

    merged = {}
    for key in keys:
        values = [batch[key] for batch in batches]

        if all(isinstance(value, np.ndarray) for value in values):
            merged[key] = np.concatenate(values, axis=0)
        elif all(isinstance(value, list) for value in values):
            merged[key] = [item for value in values for item in value]
        elif all(isinstance(value, tuple) for value in values):
            merged[key] = tuple(item for value in values for item in value)
        else:
            merged[key] = values

    return merged


def loads(path: Path, /, pattern: str) -> dict:
    path = Path(path)
    files = sorted(
        file
        for file in path.iterdir()
        if file.is_file() and re.fullmatch(pattern, file.name)
    )

    batches = []
    for file in files:
        with file.open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        batches.append(batch)

    return merge_batches(batches)


def load_c10(path):
    with open(path / 'cifar-10-batches-py/batches.meta', 'rb') as f:
        metadata = pickle.load(f, encoding='bytes')
    with open(path / 'cifar-10-batches-py/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')

    batch = loads(path / 'cifar-10-batches-py', r"data_batch_\d+")
    return metadata, batch, test_data

def load_c100(path):
    with (path / "cifar-100-python" / "meta").open("rb") as f:
        metadata = pickle.load(f, encoding="bytes")

    with (path / "cifar-100-python" / "train").open("rb") as f:
        batch = pickle.load(f, encoding="bytes")

    with (path / "cifar-100-python" / "test").open("rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    return metadata, batch, test_data