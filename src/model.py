import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file, img_to_array, load_img

from utils import save_figure
from config import DATAPATH, INDEXES_LABELS

logger = logging.getLogger(__name__)

def create_callbacks():
    model_directory = DATAPATH / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = model_directory / f"best_model_{timestamp}.keras"
    logger.info("Best model will be saved to %s", model_path)

    return [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_delta=0.001,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            start_from_epoch=10,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]


def create_data_generator():
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
                                 rotation_range=10)
    return datagen


def one_hot_encode(num_classes, y_train, y_val, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_val, y_test



def modified_model(num_classes):  # Holly AI
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    # Block 1: low-level edges, textures, and colour features.
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2: larger object parts.
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))

    # Block 3: high-level class-specific features.
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    datagen,
):
    callbacks = create_callbacks()
    history = model.fit(datagen.flow(X_train, y_train, batch_size=50), epochs=100, validation_data=(X_val, y_val),
                        verbose=1, callbacks=callbacks)

    accuracy_figure, accuracy_axis = plt.subplots()
    accuracy_axis.plot(history.history['accuracy'])
    accuracy_axis.plot(history.history['val_accuracy'])
    accuracy_axis.legend(['training', 'validation'])
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    save_figure(accuracy_figure, "accuracy")
    plt.show()
    loss_figure, loss_axis = plt.subplots()
    loss_axis.plot(history.history['loss'])
    loss_axis.plot(history.history['val_loss'])
    loss_axis.legend(['training', 'validation'])
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    save_figure(loss_figure, "loss")
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=1)
    logger.info("Test loss: %.4f", score[0])
    logger.info("Test accuracy: %.4f", score[1])


def predict_image_url(model, image_url, top_k=5):
    # Downloads into the Keras cache directory.
    image_path = get_file(origin=image_url)

    image = load_img(
        image_path,
        color_mode="rgb",
        target_size=(32, 32),
    )

    image_array = img_to_array(image).astype(np.float32) / 255.0

    # Change (32, 32, 3) into a batch: (1, 32, 32, 3).
    image_batch = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_batch, verbose=0)[0]
    best_indexes = np.argsort(probabilities)[-top_k:][::-1]

    return [
        {
            "label": INDEXES_LABELS[int(index)],
            "probability": round(float(probabilities[index]) * 100, 2),
        }
        for index in best_indexes
    ]