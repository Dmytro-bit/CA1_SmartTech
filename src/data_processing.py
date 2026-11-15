import logging

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from config import CIFAR_10_LABELS, CIFAR_100_LABELS, LABEL_INDEXES

logger = logging.getLogger(__name__)
b_ut8 = lambda x: bytes(x, "utf-8")  # definitely unsign long


def split_data(images, labels):
    X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.1, random_state=77)
    logger.info(f"Training samples {len(X_train)}, Validation samples {len(X_valid)}")
    return X_train, X_valid, y_train, y_valid


# get filtered labels and images
def process_data(metadata: dict, metadata_key: str, batch: dict, label_key: str, images_key: str,
                 selected_labels: list[str], unified_indexes: dict[str, int]):
    source_names = [
        name.decode("utf-8")
        for name in metadata[metadata_key.encode()]
    ]

    source_to_unified = {
        source_names.index(name): unified_indexes[name]
        for name in selected_labels
    }

    labels = np.asarray(batch[b_ut8(label_key)])
    mask = np.isin(labels, list(source_to_unified))

    res_images = batch[b_ut8(images_key)][mask]
    res_labels = labels[mask]

    unified_labels = np.asarray([
        source_to_unified[int(label)]
        for label in res_labels
    ])

    return unified_labels, res_images


def process_c10_data(metadata, batch):
    return process_data(
        metadata=metadata,
        metadata_key="label_names",
        batch=batch,
        label_key="labels",
        images_key="data",
        selected_labels=CIFAR_10_LABELS,
        unified_indexes=LABEL_INDEXES,
    )


def process_c100_data(metadata, batch):
    return process_data(
        metadata=metadata,
        metadata_key="fine_label_names",
        batch=batch,
        label_key="fine_labels",
        images_key="data",
        selected_labels=CIFAR_100_LABELS,
        unified_indexes=LABEL_INDEXES,
    )


def apply_preprocessing(X_train, X_val, X_test):
    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))
    return X_train, X_val, X_test


def preprocessing(img):
    img = reshape_cifar_image(img)
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def reshape_cifar_image(img):
    img = np.asarray(img)
    if img.shape == (3 * 32 * 32,):
        return img.reshape(3, 32, 32).transpose(1, 2, 0)
    if img.shape == (3, 32, 32):
        return img.transpose(1, 2, 0)
    if img.shape == (32, 32, 3):
        return img
    raise ValueError(
        f"Expected image with shape (3072,), (3, 32, 32), or "
        f"(32, 32, 3); got {img.shape}"
    )


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def equalize(img):
    # This only works for grayscale images
    img = cv2.equalizeHist(img)
    return img
