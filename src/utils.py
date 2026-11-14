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

def merge_two_np_arrays(p1, p2):
    return np.concatenate([p1, p2], axis=0)
