import argparse
import csv
import pickle
import sys
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from data_augmentation import visualize_data_set
from networks.LeNet import LeNet
from networks.NvidiaCNN import NvidiaCNN
from networks.VGG import VGG


# general setup
CFG_DATASET_PATH = './data' # Path to dataset

# hyperparameters
VALIDATION_SET_SIZE = 0.2   # proportion of full dataset used for the test set
NB_EPOCHS = 10              # default number of training epochs
BATCH_SIZE = 128            # default training batch size (number of images per batch)
CROP_IMAGE_TOP = 60         # number of pixels the image shall be cropped at top row
CROP_IMAGE_BOTTOM = 20      # number of pixels the image shall be cropped at bottom row

# augmentation
STEERING_ANGLE_CORRECTION = 3.7 # Steering angle correction for left and right images in degree


def show_configuration(dataset_csv_filename):
    """ Shows a summary of the actual configuration.
    
    :param dataset_csv_filename: Dataset CSV filename.
    """

    print('General Configuration')
    print('----------------------------------------------------------')
    print(' Path to dataset:           {:s}'.format(CFG_DATASET_PATH))
    print(' Dataset CSV file:          {:s}'.format(dataset_csv_filename))
    print('')
    print('Data pre-processing')
    print('----------------------------------------------------------')
    print(' Crop image at top:         {:d} pixels'.format(CROP_IMAGE_TOP))
    print(' Crop image at bottom:      {:d} pixels'.format(CROP_IMAGE_BOTTOM))
    print('')
    print('Hyperparameters')
    print('----------------------------------------------------------')
    print(' Validation set size:       {:.2f}'.format(VALIDATION_SET_SIZE))
    print(' Batch size:                {:d}'.format(BATCH_SIZE))
    print(' Number of epochs:          {:d}'.format(NB_EPOCHS))
    print('')
    print('Augmentation')
    print('----------------------------------------------------------')
    print(' Steering angle correction: {:.2f}°'.format(STEERING_ANGLE_CORRECTION))
    print('')


def prepare_datasets(csv_filename, validation_set_proportion=0.0):
    """ Prepares the training and validation datasets (images and measurements) from driving log cvs file.
    
    :param csv_filename:              Path and filename of CVS file.
    :param validation_set_proportion: Proportion of the full dataset used for the validation set. If set to 0.0 only 
                                      one sample array is returned. Default = 0.0
    
    :return: Returns the train_samples and validation_samples dataset.
    """

    # open and read content of csv file
    samples = []
    with open(csv_filename) as csv_file:
        reader = csv.reader(csv_file)

        # skip the csv header
        next(reader, None)

        for line in reader:
            samples.append(line)

    if validation_set_proportion == 0:
        return samples
    else:
        train_samples, validation_samples = train_test_split(samples, test_size=validation_set_proportion)

        return train_samples, validation_samples


def plot_training_statistics(history):
    """ Plots the fit history statistics like training and validation loss.
    
    :param history: History of model training ['loss', 'val_loss'].
    """
    plt.plot(history['loss'], 'x-')
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.grid()
    print('Close the figures to continue...')
    plt.show()


def train_model(model, dataset_csv_filename):
    """ Initializes and trains the selected network.
    
    :param model:           Supported models are
                             - LeNet5
                             - NvidiaCNN
                             - VGG16
    :dataset_csv_filename:  Path to dataset csv filename.
    """

    supported_models = ['LeNet5', 'NvidiaCNN', 'VGG16']

    # check for valid model
    matching = [s for s in supported_models if model in s]

    if len(matching) == 0:
        print('Not support model. Check help for supported models.')
        exit(-1)

    # prepare data sets
    print('Preparing training and validation datasets...', end='', flush=True)
    train_samples, validation_samples = prepare_datasets(dataset_csv_filename, VALIDATION_SET_SIZE)
    print('done')

    print('Number of training samples:   {:5d}'.format(len(train_samples)))
    print('Number of validation samples: {:5d}'.format(len(validation_samples)))

    global BATCH_SIZE
    global NB_EPOCHS

    if model == 'LeNet5':
        # setup LeNet-5 model architecture
        network = LeNet(input_depth=3, input_height=160, input_width=320, regression=True, nb_classes=1,
                        crop_top=CROP_IMAGE_TOP, crop_bottom=CROP_IMAGE_BOTTOM,
                        steering_angle_correction=STEERING_ANGLE_CORRECTION)
    elif model == 'NvidiaCNN':
        # setup NVIDIA CNN model architecture
        network = NvidiaCNN(input_depth=3, input_height=160, input_width=320, regression=True, nb_classes=1,
                            crop_top=CROP_IMAGE_TOP, crop_bottom=CROP_IMAGE_BOTTOM,
                            steering_angle_correction=STEERING_ANGLE_CORRECTION)
        NB_EPOCHS = 7
    elif model == 'VGG16':
        # setup VGG-16 model architecture
        network = VGG(input_depth=3, input_height=160, input_width=320, regression=True, nb_classes=1,
                      crop_top=CROP_IMAGE_TOP, crop_bottom=CROP_IMAGE_BOTTOM,
                      steering_angle_correction=STEERING_ANGLE_CORRECTION)

        # reduced batch size due to memory limitation on AWS and optimized number of epochs
        BATCH_SIZE = 16
        NB_EPOCHS = 4
    else:
        print('Not support model. Check help for supported models.')
        exit(-1)

    show_configuration(dataset_csv_filename)

    # setup training and validation generators
    network.setup_training_validation_generators(train_samples, validation_samples, CFG_DATASET_PATH, BATCH_SIZE)

    # train the model
    network.train(NB_EPOCHS)

    # save the training results
    network.save_history(model + '_history.obj')
    network.safe_model(model + '_model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training and Evaluation')

    parser.add_argument(
        '-t', '--train',
        help='Trains the model. Supported MODELS=\'LeNet5\', \'NvidiaCNN\', \'VGG16\'.',
        dest='train_model',
        metavar='MODEL'
    )

    parser.add_argument(
        '-d', '--dataset',
        help='Path to dataset CSV file.',
        dest='dataset_filename',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-vd', '--visualize-datasets',
        help='Visualizes random dataset samples.',
        dest='visualize_dataset_csv',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-th', '--show_training_history',
        help='Plots the training history (train loss and validation loss).',
        dest='history_filename',
        metavar="FILE"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(1)

    if args.train_model:
        # train the model
        if args.dataset_filename is None:
            print('Use -d CSV_FILE parameter to specify the training/validation dataset.')
        elif os.path.exists(args.dataset_filename):
            train_model(args.train_model, args.dataset_filename)
        else:
            print('Dataset CSV file \'{:s}\' not found.'.format(args.dataset_filename))
    elif args.visualize_dataset_csv:
        # Prepare data sets and show random training and validation sample
        print('Preparing training and validation datasets...', end='', flush=True)
        samples = prepare_datasets(args.visualize_dataset_csv, 0.0)
        print('done')

        print('Show random dataset samples:')
        visualize_data_set(samples, title='Dataset', dataset_path=CFG_DATASET_PATH)
    elif args.history_filename:
        # unpickle history object and plot training and validation loss
        if os.path.isfile(args.history_filename):
            with open(args.history_filename, 'rb') as file:
                history = pickle.load(file)
                plot_training_statistics(history)
        else:
            print('History object file \"{:s}\" not found.'.format(args.history_filename))
