import logging

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)


def create_data_generator():
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
                                 rotation_range=10)
    return datagen


def one_hot_encode(num_classes, y_train, y_val, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_val, y_test



def modified_model(num_classes):
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen):
    datagen.fit(X_train)
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
