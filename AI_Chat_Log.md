# AI Assistance Log: `src/model.py`

This document records the AI-assisted changes made to `src/model.py`, why they
were introduced, and how they affect training and evaluation. The suggestions
were reviewed and applied incrementally while testing the CIFAR-10/CIFAR-100
image-classification pipeline.

## 1. RGB model input

The model input was changed to:

```python
Input(shape=(32, 32, 3))
```

CIFAR images are 32 by 32 pixels with three colour channels. Earlier processing
converted them to grayscale, which discarded colour information that can help
distinguish visually similar classes. The current model therefore accepts the
RGB arrays produced by preprocessing, with values normalized to the range
`0.0` to `1.0`.

This architecture is not compatible with weights produced by the earlier
single-channel model. After changing the input shape, the model must be trained
again from newly initialized weights.

## 2. Convolutional architecture redesign

The original network used two 60-filter convolutions followed by two 30-filter
convolutions. Reducing the number of filters deeper in the network limited its
ability to represent increasingly complex visual features. It also used large
5x5 kernels and flattened the final feature maps into a large dense input.

The replacement architecture has three convolutional blocks:

```text
Input: 32x32x3

Block 1:
Conv2D 64 -> BatchNormalization -> ReLU
Conv2D 64 -> BatchNormalization -> ReLU
MaxPooling2D -> Dropout(0.25)

Block 2:
Conv2D 128 -> BatchNormalization -> ReLU
Conv2D 128 -> BatchNormalization -> ReLU
MaxPooling2D -> Dropout(0.30)

Block 3:
Conv2D 256 -> BatchNormalization -> ReLU
Conv2D 256 -> BatchNormalization -> ReLU
MaxPooling2D -> Dropout(0.40)

Classification head:
GlobalAveragePooling2D
Dense 256 -> BatchNormalization -> ReLU
Dropout(0.50)
Dense num_classes -> Softmax
```

The filter count now increases from 64 to 128 to 256 as the spatial resolution
decreases. Early layers learn edges, colours, and textures; middle layers learn
object parts; deeper layers can represent more class-specific patterns.

All convolutions use 3x3 kernels with `padding="same"`. Stacking small kernels
provides multiple nonlinear transformations while preserving the feature-map
size until pooling is applied.

## 3. Batch normalization

`BatchNormalization` was inserted after each convolution and before the ReLU
activation:

```python
Conv2D(..., use_bias=False)
BatchNormalization()
Activation("relu")
```

Batch normalization stabilizes the distribution of intermediate activations.
This commonly makes optimization more stable and can help the network converge
faster. The convolution layers use `use_bias=False` because batch normalization
already learns an offset, making a separate convolution bias redundant.

Batch normalization was also added after the 256-unit dense layer.

## 4. Progressive dropout

Dropout was added after each convolutional block, with its rate increasing as
the feature representation becomes deeper:

```text
Block 1: 0.25
Block 2: 0.30
Block 3: 0.40
Dense head: 0.50
```

Dropout randomly suppresses activations during training. This discourages the
network from depending too strongly on individual features and helps reduce
overfitting. Dropout is automatically disabled during validation, evaluation,
and prediction.

## 5. Global average pooling

The original `Flatten()` layer was replaced with `GlobalAveragePooling2D()`.
Flattening retains every spatial value and can create a parameter-heavy dense
layer. Global average pooling instead summarizes each final feature map with
one value. This reduces the size of the classification head and encourages each
feature map to represent meaningful high-level evidence.

## 6. Output layer and compilation

The final layer remains:

```python
Dense(num_classes, activation="softmax")
```

Softmax produces one probability per mutually exclusive class. The model is
compiled using Adam with a learning rate of `0.001`, categorical cross-entropy,
and accuracy:

```python
model.compile(
    Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

Categorical cross-entropy is appropriate because labels are converted to
one-hot vectors by `one_hot_encode()`.

## 7. Training callbacks

A `create_callbacks()` function was added to configure three callbacks for each
training run.

### ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_delta=0.001,
    min_lr=1e-6,
)
```

If validation loss does not improve sufficiently for three epochs, the learning
rate is halved. This allows training to make smaller parameter updates after
progress begins to plateau.

### EarlyStopping

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    start_from_epoch=10,
)
```

Early stopping begins monitoring after a ten-epoch warm-up. Training stops if
validation loss then fails to improve for ten epochs. `restore_best_weights`
ensures that the in-memory model returns to the best observed validation-loss
weights rather than retaining a later, potentially overfitted epoch.

### ModelCheckpoint

```python
ModelCheckpoint(
    filepath=model_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)
```

Only the model with the best validation loss is saved. The `.keras` file stores
the architecture, learned weights, optimizer state, and compile configuration.
It can later be restored with `tensorflow.keras.models.load_model()`.
