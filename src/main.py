from itertools import chain
from pathlib import Path

import loader
from utils import process_data

CIFAR_10_LABELS = ["automobile", "bird", "cat", "deer", "dog", "horse", "truck"]
CIFAR_100_LABELS = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "bicycle",
                    "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]
CIFAR_100_SUPER_LABELS = ["trees"]

DATAPATH = Path(__file__).resolve().parent.parent / "data"

ALL_LABELS = chain(CIFAR_10_LABELS, CIFAR_100_LABELS, CIFAR_100_SUPER_LABELS)
LABEL_INDEXES = {label: index for index, label in enumerate(ALL_LABELS)}


def main():
    metadata_c10, batch_c10, test_c10 = loader.load_c10(DATAPATH)
    metadata_c100, batch_c100, test_data_c100 = loader.load_c100(DATAPATH)

    c10_labels, c10_images = process_data(
        metadata=metadata_c10,
        metadata_key="label_names",
        batch=batch_c10,
        label_key="labels",
        images_key="data",
        selected_labels=CIFAR_10_LABELS,
        unified_indexes=LABEL_INDEXES,
    )

    c100_labels, c100_images = process_data(
        metadata=metadata_c100,
        metadata_key="fine_label_names",
        batch=batch_c100,
        label_key="fine_labels",
        images_key="data",
        selected_labels=CIFAR_100_LABELS,
        unified_indexes=LABEL_INDEXES,
    )

    return

if __name__ == "__main__":
    main()
