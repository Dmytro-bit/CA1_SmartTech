import argparse

from tensorflow.keras.models import load_model

import loader
from config import *
from data_processing import (
    apply_preprocessing,
    process_c10_data,
    process_c100_data,
    split_data,
)
from model import modified_model, one_hot_encode, create_data_generator, evaluate_model, predict_image_url
from utils import merge_two_np_arrays, show_training_samples, explore_datagen

set_logger_config()
logger = logging.getLogger(__name__)


def run_train_model():
    # Load pikle objects
    metadata_c10, batch_c10, test_c10 = loader.load_c10(DATAPATH)
    metadata_c100, batch_c100, test_c100 = loader.load_c100(DATAPATH)

    # GEt labels and images for Cifar10 dataset (train, test)
    c10_labels, c10_images = process_c10_data(metadata_c10, batch_c10)
    test_c10_labels, test_c10_images = process_c10_data(metadata_c10, test_c10)

    # Get labels and images for Cifar100 dataset (train, test)
    c100_labels, c100_images = process_c100_data(metadata_c100, batch_c100)
    test_c100_labels, test_c100_images = process_c100_data(metadata_c100, test_c100)

    # Merge two datasets in one (train, test)
    merged_train_labels = merge_two_np_arrays(c10_labels, c100_labels)
    merged_train_images = merge_two_np_arrays(c10_images, c100_images)
    merged_test_labels = merge_two_np_arrays(test_c10_labels, test_c100_labels)
    merged_test_images = merge_two_np_arrays(test_c10_images, test_c100_images)

    X_train, X_valid, y_train, y_valid = split_data(merged_train_images, merged_train_labels)

    X_train, X_val, X_test = apply_preprocessing(X_train, X_valid, merged_test_images)
    num_of_each_type = show_training_samples(NUM_OF_CLASSES, INDEXES_LABELS, X_train, y_train)

    y_train, y_val, y_test = one_hot_encode(NUM_OF_CLASSES, y_train, y_valid, merged_test_labels)

    model = modified_model(NUM_OF_CLASSES)

    model.summary(print_fn=logger.info)

    datagen = create_data_generator()
    explore_datagen(datagen, X_train, y_train)

    evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen)

    return


def run_prediction(model_path, image_url):
    model = load_model(model_path)
    logger.info(f"Loaded : {model_path}")

    return predict_image_url(model, image_url)


def run_model_test(model_path):
    metadata_c10, test_c10 = loader.load_c10_test(DATAPATH)
    metadata_c100, test_c100 = loader.load_c100_test(DATAPATH)

    test_c10_labels, test_c10_images = process_c10_data(metadata_c10, test_c10)
    test_c100_labels, test_c100_images = process_c100_data(metadata_c100, test_c100)

    test_labels = merge_two_np_arrays(test_c10_labels, test_c100_labels)
    test_images = merge_two_np_arrays(test_c10_images, test_c100_images)
    _, _, X_test = apply_preprocessing([], [], test_images)
    _, _, y_test = one_hot_encode(NUM_OF_CLASSES, [], [], test_labels)

    model = load_model(model_path)
    logger.info("Loaded: %s", model_path)

    results = model.evaluate(X_test, y_test, verbose=1, return_dict=True)
    for metric, value in results.items():
        logger.info("Test %s: %.4f", metric, value)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        help="Run model training",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Saved model to test, or to use for URL prediction",
        type=str,
        required=False
    )
    parser.add_argument(
        "--url",
        help="Optional image URL; without it, evaluate the model on the test dataset",
        type=str,
        required=False
    )
    args = parser.parse_args()

    if args.train is True:
        run_train_model()
    else:
        if args.model:
            if args.url:
                print(run_prediction(args.model, args.url))
            else:
                print(run_model_test(args.model))
        else:
            parser.error("-m/--model is required when -t/--train is absent")
