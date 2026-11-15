import loader
from config import *
from data_processing import process_c10_data, process_c100_data, apply_preprocessing, split_data
from model import modified_model, reshape_for_cnn, one_hot_encode, create_data_generator, evaluate_model
from utils import merge_two_np_arrays, show_training_samples, explore_datagen

set_logger_config()
logger = logging.getLogger(__name__)


def main():
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

    X_train, X_val, X_test = reshape_for_cnn(X_train, X_val, X_test)
    y_train, y_val, y_test = one_hot_encode(NUM_OF_CLASSES, y_train, y_valid, merged_test_labels)

    model = modified_model(NUM_OF_CLASSES)

    model.summary(print_fn=logger.info)

    datagen = create_data_generator()
    explore_datagen(datagen, X_train, y_train)

    evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen)

    return


if __name__ == "__main__":
    main()
