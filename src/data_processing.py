import numpy as np

from config import CIFAR_10_LABELS, CIFAR_100_LABELS, LABEL_INDEXES

b_ut8 = lambda x: bytes(x, "utf-8")  # definitely unsign long


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
