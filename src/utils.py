import matplotlib.pyplot as plt
import numpy as np


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


def show_training_samples(num_classes, data: dict[int, str], X_train, y_train):
    images = np.asarray(X_train)
    labels = np.asarray(y_train)

    classes = list(data.items())[:num_classes]
    cols = 5
    figure, axes = plt.subplots(
        nrows=num_classes,
        ncols=cols,
        figsize=(10, max(2, num_classes * 2)),
        squeeze=False,
    )
    rng = np.random.default_rng()
    sample_counts = []

    for row_index, (class_index, class_name) in enumerate(classes):
        selected = images[labels == class_index]
        sample_counts.append(len(selected))

        axes[row_index, 0].set_ylabel(class_name, rotation=0, labelpad=35)
        if len(selected) == 0:
            continue
        else:
            sample_indexes = rng.choice(
                len(selected),
                size=cols,
                replace=len(selected) < cols,
            )
            for column_index, sample_index in enumerate(sample_indexes):
                image = selected[sample_index]
                cmap = "gray" if image.ndim == 2 else None
                axes[row_index, column_index].imshow(image, cmap=cmap)

        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.tight_layout()
    plt.show()
    return sample_counts


def explore_datagen(datagen, X_train, y_train):
    datagen.fit(X_train)
    batches = datagen.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()
    for i in range(15):
        axs[i].imshow(np.clip(X_batch[i], 0.0, 1.0))
        axs[i].axis('off')
    plt.show()
