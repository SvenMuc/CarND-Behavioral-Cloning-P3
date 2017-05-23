import argparse
import csv
import pickle
import sys
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from networks.LeNet import LeNet
from networks.NvidiaFull import NvidiaFull
from networks.NvidiaLight import NvidiaLight
from networks.VGG import VGG


# general setup
CFG_DATASET_PATH = './data'          # Path to dataset

# hyperparameters
VALIDATION_SET_SIZE = 0.2            # proportion of full dataset used for the test set
BATCH_SIZE = 256                     # default training batch size (number of images per batch)
ROI = [20, 60, 320 - 20, 160 - 22]   # Region if interest
IMAGE_WIDTH = 64                     # Model input image width
IMAGE_HEIGHT = 64                    # Model input image height

# augmentation
STEERING_ANGLE_CORRECTION = 6.25     # Steering angle correction for left and right images in degree


def show_configuration(dataset_csv_filename, nb_epochs=None, learning_rate=None):
    """ Shows a summary of the actual configuration.

    :param dataset_csv_filename: Dataset CSV filename.
    :param nb_epochs:            Number of training epochs.
    :param learning_rate:        The optimizer's learning rate.
    """

    print('General Configuration')
    print('----------------------------------------------------------')
    print(' Path to dataset:           {:s}'.format(CFG_DATASET_PATH))
    print(' Dataset CSV file:          {:s}'.format(dataset_csv_filename))
    print('')
    print('Data pre-processing')
    print('----------------------------------------------------------')
    print(' ROI:                       [{:d}, {:d}, {:d}, {:d}] pixels'.format(ROI[0], ROI[1], ROI[2], ROI[3]))
    print(' Image width:               {:d}'.format(IMAGE_WIDTH))
    print(' Image height:              {:d}'.format(IMAGE_HEIGHT))
    print('')
    print('Hyperparameters')
    print('----------------------------------------------------------')
    print(' Validation set size:       {:.2f}'.format(VALIDATION_SET_SIZE))
    print(' Batch size:                {:d}'.format(BATCH_SIZE))
    if nb_epochs is not None:
        print(' Number of epochs:          {:d}'.format(nb_epochs))
    if learning_rate is not None:
        print(' Learning rate:             {:f}'.format(learning_rate))
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
        return shuffle(samples)
    else:
        train_samples, validation_samples = train_test_split(shuffle(samples), test_size=validation_set_proportion)

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


def train_model(model, dataset_csv_filename, trained_model=None, nb_epochs=7, nb_samples_per_epoch=20.000, learning_rate=0.001):
    """ Initializes and trains the selected network.

    :param model:           Supported models are
                             - LeNet5
                             - NvidiaFull
                             - NvidiaLight (lightweight NVIDIA architecture)
                             - VGG16
    :param dataset_csv_filename:  Path to dataset csv filename.
    :param trained_model:         Trained model file (*.h5). If set, the model will be retrained by the given dataset.
    :param nb_epochs:             Number training epochs. Default = 7
    :param nb_samples_per_epoch:  Number of samples which will be train in each epoch. Default = 20.000
    :param learning_rate:         The optimizer's learning rate. Default = 0.001
    """

    supported_models = ['LeNet5', 'NvidiaFull', 'NvidiaLight', 'VGG16']

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

    if model == 'LeNet5':
        # setup LeNet-5 model architecture
        network = LeNet(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                        regression=True, nb_classes=1,
                        roi=ROI,
                        steering_angle_correction=STEERING_ANGLE_CORRECTION,
                        weights_path=trained_model)
    elif model == 'NvidiaFull':
        # setup Full NVIDIA CNN model architecture
        network = NvidiaFull(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                             regression=True, nb_classes=1,
                             roi=ROI,
                             steering_angle_correction=STEERING_ANGLE_CORRECTION,
                             weights_path=trained_model)
    elif model == 'NvidiaLight':
        # setup lightweight NVIDIA CNN model architecture
        network = NvidiaLight(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                              regression=True, nb_classes=1,
                              roi=ROI,
                              steering_angle_correction=STEERING_ANGLE_CORRECTION,
                              weights_path=trained_model)
    elif model == 'VGG16':
        # setup VGG-16 model architecture
        network = VGG(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                      regression=True, nb_classes=1,
                      roi=ROI,
                      steering_angle_correction=STEERING_ANGLE_CORRECTION,
                      weights_path=trained_model)

        # reduced batch size due to memory limitation on AWS and optimized number of epochs
        BATCH_SIZE = 16
    else:
        print('Not support model. Check help for supported models.')
        exit(-1)

    show_configuration(dataset_csv_filename, learning_rate=learning_rate, nb_epochs=nb_epochs)
    print('{:s} Network Summary'.format(network.model_name))
    network.summary()
    # network.verbose = 1

    # setup training and validation generators
    network.setup_training_validation_generators(train_samples, validation_samples, CFG_DATASET_PATH, BATCH_SIZE)

    # train the model
    if trained_model is None:
        print('Train model:')
    else:
        print('Retrain model \'{:s}\':'.format(trained_model))

    network.train(nb_epochs=nb_epochs, nb_samples_per_epoch=nb_samples_per_epoch, learning_rate=learning_rate)

    # save the training results
    network.save_history(model + '_history.obj')
    network.safe_model(model + '_model.h5')


def save_model_graph(model, filename):
    """ Saves the model graph to a PNG file.

    :param model:     Supported models are
                        - LeNet5
                        - NvidiaFull
                        - NvidiaLight (lightweight NVIDIA architecture)
                        - VGG16
    :param filename:  Filename of PNG file.
    """

    supported_models = ['LeNet5', 'NvidiaFull', 'NvidiaLight', 'VGG16']

    # check for valid model
    matching = [s for s in supported_models if model in s]

    if len(matching) == 0:
        print('Not support model. Check help for supported models.')
        exit(-1)

    if model == 'LeNet5':
        # setup LeNet-5 model architecture
        network = LeNet(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                        regression=True, nb_classes=1,
                        roi=ROI,
                        steering_angle_correction=STEERING_ANGLE_CORRECTION,
                        weights_path=None)
    elif model == 'NvidiaFull':
        # setup Full NVIDIA CNN model architecture
        network = NvidiaFull(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                             regression=True, nb_classes=1,
                             roi=ROI,
                             steering_angle_correction=STEERING_ANGLE_CORRECTION,
                             weights_path=None)
    elif model == 'NvidiaLight':
        # setup lightweight NVIDIA CNN model architecture
        network = NvidiaLight(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                              regression=True, nb_classes=1,
                              roi=ROI,
                              steering_angle_correction=STEERING_ANGLE_CORRECTION,
                              weights_path=None)
    elif model == 'VGG16':
        # setup VGG-16 model architecture
        network = VGG(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                      regression=True, nb_classes=1,
                      roi=ROI,
                      steering_angle_correction=STEERING_ANGLE_CORRECTION,
                      weights_path=None)
    else:
        print('Not support model. Check help for supported models.')
        exit(-1)

    print('Safe model graph...', end='', flush=True)
    network.save_model_graph(filename)
    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training and Evaluation')

    parser.add_argument(
        '-t', '--train',
        help='Trains the model. Supported MODELS=\'LeNet5\', \'NvidiaFull\', \'NvidiaLight\', \'VGG16\'.',
        dest='train_model',
        metavar='MODEL'
    )

    parser.add_argument(
        '-m', '--model',
        help='Retrains the given model. Select model by -t.',
        dest='retrain_model',
        metavar='MODEL_FILE'
    )

    parser.add_argument(
        '-d', '--dataset',
        help='Path to dataset CSV file.',
        dest='dataset_filename',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-ne', '--number-epochs',
        help='Number of training epochs (default = 7).',
        dest='number_epochs',
        metavar='EPOCHS'
    )

    parser.add_argument(
        '-ns', '--number-samples_per_epoch',
        help='Number samples trained each epoch (default = 20.000).',
        dest='number_samples_epoch',
        metavar='SAMPLES'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        help='Learning rate of the optimizer (default = 0.001).',
        dest='learning_rate',
        metavar='RATE'
    )

    parser.add_argument(
        '-th', '--show_training_history',
        help='Plots the training history (train loss and validation loss).',
        dest='history_filename',
        metavar='FILE'
    )

    parser.add_argument(
        '-smg', '--safe-model-graph',
        help='Saves the model graph to a PNG file (1=model name, 2=filename).',
        dest='safe_model_graph',
        nargs=2,
        metavar='ARG'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(1)

    if args.train_model:
        # train the model
        if args.dataset_filename is None:
            print('ERROR: Use -d CSV_FILE parameter to specify the training/validation dataset.')
        elif os.path.exists(args.dataset_filename):
            if args.learning_rate is None:
                lr = 0.001
                print('Using default learning rate {:f}'.format(lr))
            else:
                lr = float(args.learning_rate)

            if args.number_epochs is None:
                print('ERROR: Use -ne parameter to specify the number training epochs.')
                exit(-1)

            if args.number_samples_epoch is None:
                print('ERROR: Use -ns parameter to specify the number samples per epoch.')
                exit(-1)

            if args.retrain_model:
                train_model(model=args.train_model,
                            dataset_csv_filename=args.dataset_filename,
                            trained_model=args.retrain_model,
                            nb_epochs=int(args.number_epochs),
                            nb_samples_per_epoch=int(args.number_samples_epoch),
                            learning_rate=lr)
            else:
                train_model(model=args.train_model,
                            dataset_csv_filename=args.dataset_filename,
                            nb_epochs=int(args.number_epochs),
                            nb_samples_per_epoch=int(args.number_samples_epoch),
                            learning_rate=lr)
        else:
            print('Dataset CSV file \'{:s}\' not found.'.format(args.dataset_filename))

    elif args.history_filename:
        # unpickle history object and plot training and validation loss
        if os.path.isfile(args.history_filename):
            with open(args.history_filename, 'rb') as file:
                history = pickle.load(file)
                plot_training_statistics(history)
        else:
            print('History object file \"{:s}\" not found.'.format(args.history_filename))

    elif args.safe_model_graph:
        # safes the model graph to a PNG file
        save_model_graph(args.safe_model_graph[0], args.safe_model_graph[1])

