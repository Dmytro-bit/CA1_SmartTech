import loader
from utils import process_data, merge_two_np_arrays
from config import *
from data_processing import process_c10_data, process_c100_data

set_logger_config()
logger = logging.getLogger(__name__)


def main():
    #Load pikle objects
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


    return


if __name__ == "__main__":
    main()
