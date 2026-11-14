from itertools import chain
from pathlib import Path

import loader

CIRAF_10_LABELS = ["automobile", "bird", "cat", "deer", "dog", "horse", "truck"]
CIRAF_100_LABELS = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "trees", "bicycle",
                    "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]

DATAPATH = Path(__file__).resolve().parent.parent / "data"

LABEL_INDEXES = {label: index for label, index in enumerate(chain(CIRAF_10_LABELS, CIRAF_100_LABELS))}


def main():
    metadata_c10, batch_c10, test_c10 = loader.load_c10(DATAPATH)
    metadata_c100, batch_c100, test_data_c100 = loader.load_c100(DATAPATH)

    return


if __name__ == "__main__":
    main()
