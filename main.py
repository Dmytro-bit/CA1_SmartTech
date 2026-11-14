import pickle
import numpy as np

CIRAF_10_LABELS = ["automobile", "bird", "cat", "deer", "dog", "horse", "truck"]
CIRAF_100_LABELS = []

def unpickle():
    with open('data/cifar-10-batches-py/batches.meta', 'rb') as f:
        metadata = pickle.load(f, encoding='bytes')
    with open('data/cifar-10-batches-py/data_batch_1', 'rb') as f:
        batch_1 = pickle.load(f, encoding='bytes')
    with open('data/cifar-10-batches-py/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    return metadata, batch_1, test_data


def get_label_number(metadata, str_label):
    label_names = [
        name.decode('utf-8')
        for name in metadata[b'label_names']
    ]

    return label_names.index(str_label.lower())

def process_data(metadata, batch, selected_labels: list[str]):
    wanted_labels = [
        get_label_number(metadata, name)
        for name in selected_labels
    ]
    labels = np.asarray(batch[b'labels'])
    mask = np.isin(labels, wanted_labels)

    res_images = batch[b'data'][mask]
    res_labels = labels[mask]

    return res_labels, res_images



def main():
    metadata , batch_1, test_data = unpickle()
    res_l_1, res_i_1 = process_data(metadata, batch_1, CIRAF_10_LABELS)


    return res_l_1, res_i_1

if __name__ == "__main__":
    main()
