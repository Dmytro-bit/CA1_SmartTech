import pickle
import re
from pathlib import Path

from utils import merge_batches


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


def load_c10_test(path):
    with open(path / 'cifar-10-batches-py/batches.meta', 'rb') as f:
        metadata = pickle.load(f, encoding='bytes')
    with open(path / 'cifar-10-batches-py/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    return metadata, test_data


def load_c10(path):
    metadata, test_data = load_c10_test(path)

    batch = loads(path / 'cifar-10-batches-py', r"data_batch_\d+")
    return metadata, batch, test_data


def load_c100_test(path):
    with (path / "cifar-100-python" / "meta").open("rb") as f:
        metadata = pickle.load(f, encoding="bytes")
    with (path / "cifar-100-python" / "test").open("rb") as f:
        test_data = pickle.load(f, encoding="bytes")
    return metadata, test_data


def load_c100(path):
    metadata, test_data = load_c100_test(path)

    with (path / "cifar-100-python" / "train").open("rb") as f:
        batch = pickle.load(f, encoding="bytes")

    return metadata, batch, test_data
