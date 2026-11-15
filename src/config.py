import logging
import sys
from itertools import chain
from pathlib import Path

CIFAR_10_LABELS = ["automobile", "bird", "cat", "deer", "dog", "horse", "truck"]
CIFAR_100_LABELS = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "bicycle",
                    "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]
CIFAR_100_SUPER_LABELS = ["trees"]
NUM_OF_CLASSES = 24

DATAPATH = Path(__file__).resolve().parent.parent / "data"

ALL_LABELS = list(dict.fromkeys(chain(CIFAR_10_LABELS, CIFAR_100_LABELS, CIFAR_100_SUPER_LABELS)))
LABEL_INDEXES = {label: index for index, label in enumerate(ALL_LABELS)}
INDEXES_LABELS = {index: label for label, index in LABEL_INDEXES.items()}


def set_logger_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler(DATAPATH / "logs/app.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
