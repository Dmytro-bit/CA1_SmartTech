import numpy as np

b_ut8 = lambda x: bytes(x, "utf-8") #definitely unsign long

# convert text label into index
def get_label_number(metadata: dict, metadata_key: str, label: str):
    try:
        return metadata[metadata_key].index(b_ut8(label))
    except IndexError:
        raise IndexError(f"Metadata key error, {label} not found in metadata[{metadata_key}]")

# get filtered labels and images
def process_data(metadata, metadata_key, batch, label_key, images_key, selected_labels: list[str]):
    wanted_labels = [
        get_label_number(metadata, metadata_key, name)
        for name in selected_labels
    ]
    labels = np.asarray(batch[b_ut8(label_key)])
    mask = np.isin(labels, wanted_labels)

    res_images = batch[b_ut8(images_key)][mask]
    res_labels = labels[mask]

    return res_labels, res_images

