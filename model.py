import argparse
import csv
import cv2
import pickle
import sys
from random import randint
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from networks.LeNet import LeNet

# general setup
CFG_DATA_IMAGE_PATH = './data/IMG/'  # Path to image data

# hyperparameters
VALIDATION_SET_SIZE = 0.2   # proportion of full dataset used for the test set
NB_EPOCHS = 10              # number of training epochs
BATCH_SIZE = 128            # Training batch size (number of images per batch)
CROP_IMAGE_TOP = 60         # number of pixels the image shall be cropped at top row
CROP_IMAGE_BOTTOM = 20      # number of pixels the image shall be cropped at bottom row


def show_configuration():
    """ Shows a summary of the actual configuration. """

    print(" Configuration")
    print("----------------------------------------------------------")
    print(" Image data path:        {:s}".format(CFG_DATA_IMAGE_PATH))
    print("")
    print(" Data pre-processing")
    print("----------------------------------------------------------")
    print(" Crop image at top:      {:d} pixels".format(CROP_IMAGE_TOP))
    print(" Crop image at bottom:   {:d} pixels".format(CROP_IMAGE_BOTTOM))
    print("")
    print(" Hyperparameters")
    print("----------------------------------------------------------")
    print(" Validation set size:    {:.2f}".format(VALIDATION_SET_SIZE))
    print(" Batch size:             {:d}".format(BATCH_SIZE))
    print(" Number of epochs:       {:d}".format(NB_EPOCHS))
    print("")


def prepare_datasets(csv_filename, validation_set_proportion):
    """ Prepares the training and validation datasets (images and measurements) from driving log cvs file.
    
    :param csv_filename:              Path and filename of CVS file.
    :param validation_set_proportion: Proportion of the full dataset used for the validation set.
    
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

    train_samples, validation_samples = train_test_split(samples, test_size=validation_set_proportion)

    return train_samples, validation_samples


def visualize_data_set(samples):
    """ Visualize the data set (random image) and ground truth data.
    
    REMARK:
    In order to enable plotting in PyCharm add environment variable 'DISPLAY = True'.
    
    :param samples: Samples in format [center, left, right, steering, throttle, brake, speed].
    """

    print("Plotting diagrams...", end='', flush=True)

    #
    # Plot center, left and right random image
    #
    nb_samples = len(samples)
    idx = randint(0, nb_samples)

    # load random center, left and right image
    filename = samples[idx][0].split('/')[-1]
    image_path = CFG_DATA_IMAGE_PATH + filename
    center_image = cv2.imread(image_path)

    filename = samples[idx][1].split('/')[-1]
    image_path = CFG_DATA_IMAGE_PATH + filename
    left_image = cv2.imread(image_path)

    filename = samples[idx][2].split('/')[-1]
    image_path = CFG_DATA_IMAGE_PATH + filename
    right_image = cv2.imread(image_path)

    # all images have the same width and height
    height = center_image.shape[0]
    width = center_image.shape[1]

    # show random images
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    title = 'image set {:d} (w={:d}, h={:d}) with ROI = [0, {:d}, {:d}, {:d}]'.format(idx, width, height,
                                                                                      CROP_IMAGE_TOP, width,
                                                                                      height - CROP_IMAGE_BOTTOM)
    fig.suptitle(title)

    # prepare images and show ROI
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(left_image, (0, CROP_IMAGE_TOP), (width - 1, height - CROP_IMAGE_BOTTOM), (255, 0, 0), 1)
    ax1.imshow(left_image)
    ax1.set_title('left image')

    center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(center_image, (0, CROP_IMAGE_TOP), (width - 1, height - CROP_IMAGE_BOTTOM), (255, 0, 0), 1)
    ax2.imshow(center_image)
    ax2.set_title('center image')

    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(right_image, (0, CROP_IMAGE_TOP), (width - 1, height - CROP_IMAGE_BOTTOM), (255, 0, 0), 1)
    ax3.imshow(right_image)
    ax3.set_title('right image')

    plt.subplots_adjust(left=0.04, right=0.98, top=0.75)

    #
    # plot steering angle histogram
    #
    center_angles = []

    for sample in samples:
        center_angles.append(float(sample[3]))

    fig = plt.figure()
    n, bins, patches = plt.hist(center_angles, 100, normed=1, facecolor='green', edgecolor='black', alpha=0.75,
                                histtype='bar', rwidth=0.85)
    plt.xlabel('center steering angle [degree]')
    plt.ylabel('frequency')
    plt.grid(True)

    # add a line showing the expected distribution
    (mu, sigma) = norm.fit(center_angles)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=1.5)

    fig.suptitle(r'$\mathrm{Histogram\ of\ center\ steering\ angle}\ (\mu=%.4f,\ \sigma=%.4f)$' % (mu, sigma))

    print('done')
    print('Close the figures to continue...')
    plt.show()


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


def train_model(model):
    """ Initializes and trains the selected network.
    
    :param model: Supported models are
                  - LeNet5
                  - VGG16
    """

    supported_models = ['LeNet5', 'VGG16']

    # check for valid model
    matching = [s for s in supported_models if model in s]

    if len(matching) == 0:
        print('Not support model. Check help for supported models.')
        exit(-1)

    show_configuration()

    # prepare data sets
    print('Preparing training and validation datasets...', end='', flush=True)
    train_samples, validation_samples = prepare_datasets('./data/driving_log.csv', VALIDATION_SET_SIZE)
    print('done')

    print('Number of training samples:   {:5d}'.format(len(train_samples)))
    print('Number of validation samples: {:5d}'.format(len(validation_samples)))

    if model == 'LeNet5':
        # setup LeNet-5 model architecture
        network = LeNet(input_depth=3, input_height=160, input_width=320, regression=True, nb_classes=1,
                        crop_top=CROP_IMAGE_TOP, crop_bottom=CROP_IMAGE_BOTTOM)
    else:
        print('Not support model. Check help for supported models.')
        exit(-1)

    # setup training and validation generators
    network.setup_training_validation_generators(train_samples, validation_samples, CFG_DATA_IMAGE_PATH, BATCH_SIZE)

    # train the model
    network.train(NB_EPOCHS)

    # save the training results
    network.save_history(model + '_history.obj')
    network.safe_model(model + '_model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training and Evaluation')

    parser.add_argument(
        '-t', '--train',
        help='Trains the model. Supported MODELS=\'LeNet5\', \'VGG16\'.',
        dest='train_model',
        metavar='MODEL'
    )

    parser.add_argument(
        '-sc', '--show-configuration',
        help='Shows the model configuration.',
        dest='show_configuration',
        action='store_true'
    )

    parser.add_argument(
        '-vds', '--visualize-datasets',
        help='Visualizes random training and validation sample.',
        dest='visualize_datasets',
        action='store_true'
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
        train_model(args.train_model)
    elif args.show_configuration:
        # show actual configuration
        show_configuration()
    elif args.visualize_datasets:
        # Prepare data sets and show random training and validation sample
        print('Preparing training and validation datasets...', end='', flush=True)
        train_samples, validation_samples = prepare_datasets('./data/driving_log.csv', VALIDATION_SET_SIZE)
        print('done')
        print('Show random training sample...')
        visualize_data_set(train_samples)
        print('Show random validation sample...')
        visualize_data_set(validation_samples)
    elif args.history_filename:
        # unpickle history object and plot training and validation loss
        if os.path.isfile(args.history_filename):
            with open(args.history_filename, 'rb') as file:
                history = pickle.load(file)
                plot_training_statistics(history)
        else:
            print('History object file \"{:s}\" not found.'.format(args.history_filename))
