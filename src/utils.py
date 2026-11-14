import numpy as np

b_ut8 = lambda x: bytes(x, "utf-8")  # definitely unsign long


# convert text label into index
def get_label_number(metadata: dict, metadata_key: str, label: str):
    try:
        return metadata[metadata_key].index(b_ut8(label))
    except IndexError:
        raise IndexError(f"Metadata key error, {label} not found in metadata[{metadata_key}]")


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
