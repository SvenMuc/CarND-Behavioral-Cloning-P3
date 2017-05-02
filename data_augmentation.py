import argparse
import csv
import cv2
import sys
import math
import os
from random import randint
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split


# general setup
CFG_DATASET_PATH = './data' # Path to dataset

# hyperparameters
VALIDATION_SET_SIZE = 0.2   # proportion of full dataset used for the test set
NB_EPOCHS = 10              # default number of training epochs
BATCH_SIZE = 128            # default training batch size (number of images per batch)
CROP_IMAGE_TOP = 60         # number of pixels the image shall be cropped at top row
CROP_IMAGE_BOTTOM = 20      # number of pixels the image shall be cropped at bottom row


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


def plot_steering_angle_histogram(steering_angles, title='Histogram of steering angle', show=True):
    """ Plot steering angle histogram.
    
    :param steering_angles: Array containing steering angles in radiant.
    :param title:           Title of the histogram.
    :param show:            If true, `plot.show()` is called at the end.
    """
    fig = plt.figure()
    n, bins, patches = plt.hist(steering_angles, 100, normed=1, facecolor='green', edgecolor='black', alpha=0.75,
                                histtype='bar', rwidth=0.85)
    plt.xlabel('steering angle [degree]')
    plt.ylabel('frequency')
    plt.grid(True)

    # add a line showing the expected distribution
    (mu, sigma) = norm.fit(steering_angles)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=1.5)

    fig.suptitle(r'$\mathrm{%s}\ (\mu=%.4f,\ \sigma=%.4f)$' % (title.replace(' ', '\ '), mu, sigma))

    if show:
        plt.show()


def visualize_data_set(samples, title='', dataset_path=''):
    """ Visualize the data set (random image) and ground truth data.
    
    REMARK:
    In order to enable plotting in PyCharm add environment variable 'DISPLAY = True'.
    
    :param samples:   Samples in format [center, left, right, steering, throttle, brake, speed].
    :param title:     Title shown on the image figure.
    :param dataset_path: Path to data directory without '/' at the end.
    """

    print("Plotting diagrams...", end='', flush=True)

    #
    # Plot center, left and right random image
    #
    nb_shown_images = 5
    nb_samples = len(samples)

    fig, axarr = plt.subplots(nb_shown_images, 3, figsize=(10, 8))

    for i in range(nb_shown_images):
        idx = randint(0, nb_samples)

        # load random center, left and right image
        center_image = cv2.imread(dataset_path + '/' + samples[idx][0].lstrip())
        left_image = cv2.imread(dataset_path + '/' + samples[idx][1].lstrip())
        right_image = cv2.imread(dataset_path + '/' + samples[idx][2].lstrip())

        # all images have the same width and height
        height = center_image.shape[0]
        width = center_image.shape[1]

        # prepare images and show ROI
        left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(left_image, (0, CROP_IMAGE_TOP), (width - 1, height - CROP_IMAGE_BOTTOM), (255, 0, 0), 1)
        axarr[i-1, 0].imshow(left_image)
        axarr[i-1, 0].axis('off')

        center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(center_image, (0, CROP_IMAGE_TOP), (width - 1, height - CROP_IMAGE_BOTTOM), (255, 0, 0), 1)
        axarr[i-1, 1].imshow(center_image)
        axarr[i-1, 1].axis('off')

        right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(right_image, (0, CROP_IMAGE_TOP), (width - 1, height - CROP_IMAGE_BOTTOM), (255, 0, 0), 1)
        axarr[i-1, 2].imshow(right_image)
        axarr[i-1, 2].axis('off')

    # set titles
    axarr[0, 0].set_title('left image')
    axarr[0, 1].set_title('center image')
    axarr[0, 2].set_title('right image')
    title = '{:s} (w={:d}, h={:d}) with ROI = [0, {:d}, {:d}, {:d}]'.format(title, width, height,
                                                                            CROP_IMAGE_TOP, width,
                                                                            height - CROP_IMAGE_BOTTOM)
    fig.suptitle(title)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05)

    #
    # plot steering angle histogram
    #
    center_angles = []

    for sample in samples:
        center_angles.append(math.degrees(float(sample[3])))

    plot_steering_angle_histogram(center_angles, title='Histogram of center steering angle', show=False)

    print('done')
    print('Close the figures to continue...')
    plt.show()


def augment_dataset(source_directory, destination_directory):
    """ Augments the data set.
     
     Methods:
      - Flip images with curves (steering angle >= threshold) horizontally.
    
    :param source_directory:      Source path of data to augment without a \ at the end.        
    :param destination_directory: Destination path without a \ at the end to store augmented data.
    """

    # check if all source and destination directories exist
    source_image_directory = source_directory + '/IMG'
    source_csv_log_file = source_directory + '/driving_log.csv'
    augmented_image_sub_directory = 'IMG_augmented'
    augmented_image_directory = destination_directory + '/' + augmented_image_sub_directory
    augmented_csv_log_file = destination_directory + '/augmented_log.csv'

    if not os.path.exists(destination_directory):
        print('Source directory \'{:s}\' does not exists.'.format(source_directory))
        exit(-1)

    if os.path.exists(destination_directory):
        if os.path.exists(augmented_image_directory):
            print('All source and destination directories exist. Ready to augment data.')
        else:
            os.makedirs(augmented_image_directory)
            print('Created destination directory \'{:s}\' for augmented images.'.format(augmented_image_directory))
    else:
        os.makedirs(destination_directory)
        print('Created destination directory \'{:s}\' for CSV file'.format(destination_directory))
        os.makedirs(augmented_image_directory)
        print('Created destination directory \'{:s}\' for augmented images.'.format(augmented_image_directory))

    # get full dataset which shall be augmented
    samples = prepare_datasets(source_csv_log_file, 0.0)

    # find images with curves by |steering angle| >= threshold
    samples_curves = []
    steering_angle_threshold = 6.0

    print('Number of samples:        {:d}'.format(len(samples)))
    print('Source CVS file:          {:s}'.format(source_csv_log_file))
    print('Source images:            {:s}'.format(source_image_directory))
    print('Augmented CVS file:       {:s}'.format(augmented_csv_log_file))
    print('Augmented images:         {:s}'.format(augmented_image_directory))
    print('Steering angle threshold: {:.3f}°'.format(steering_angle_threshold))
    print('')
    print('Augment dataset...', end='', flush=True)

    # augment dataset
    csv_file = open(augmented_csv_log_file, 'w')
    fieldnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for sample in samples:
        if abs(float(sample[3])) >= math.radians(steering_angle_threshold):
            samples_curves.append(sample)

            # load and flip center, left and right image horizontally
            center_image_filename = sample[0].split('/')[-1]
            center_image_path = source_image_directory + '/' + center_image_filename
            center_image = cv2.imread(center_image_path)

            if center_image is None:
                print('Image {:s} not found.'.format(center_image_path))
                exit(-1)
            else:
                cv2.imwrite(augmented_image_directory + '/flipped_' + center_image_filename, cv2.flip(center_image, 1))

            left_image_filename = sample[1].split('/')[-1]
            left_image_path = source_image_directory + '/' + left_image_filename
            left_image = cv2.imread(left_image_path)

            if left_image is None:
                print('Image {:s} not found.'.format(center_image_path))
                exit(-1)
            else:
                cv2.imwrite(augmented_image_directory + '/flipped_' + left_image_filename, cv2.flip(left_image, 1))

            right_image_filename = sample[2].split('/')[-1]
            right_image_path = source_image_directory + '/' + right_image_filename
            right_image = cv2.imread(right_image_path)

            if right_image is None:
                print('Image {:s} not found.'.format(center_image_path))
                exit(-1)
            else:
                cv2.imwrite(augmented_image_directory + '/flipped_' + right_image_filename, cv2.flip(right_image, 1))

            # invert steering angle
            steering_angle_flipped = -float(sample[3])

            # write augmented data to CSV file
            # Header = center, left, right, steering, throttle, brake, speed
            writer.writerow({'center': augmented_image_sub_directory + '/flipped_' + center_image_filename,
                             'left': augmented_image_sub_directory + '/flipped_' + left_image_filename,
                             'right': augmented_image_sub_directory + '/flipped_' + right_image_filename,
                             'steering': str(steering_angle_flipped),
                             'throttle': sample[4],
                             'brake': sample[5],
                             'speed': sample[6]})

        # write not augmented data to CSV file
        writer.writerow({'center': sample[0],
                         'left': sample[1],
                         'right': sample[2],
                         'steering': sample[3],
                         'throttle': sample[4],
                         'brake': sample[5],
                         'speed': sample[6]})
    print('done')
    print('')

    #
    # Plot augmented steering angle histogram
    #
    augmented_samples = prepare_datasets(augmented_csv_log_file, 0.0)

    augmented_steering_angles = []
    steering_angles = []

    for sample in augmented_samples:
        augmented_steering_angles.append(math.degrees(float(sample[3])))

    plot_steering_angle_histogram(augmented_steering_angles, title='Augmented steering angles', show=False)

    for sample in samples:
        steering_angles.append(math.degrees(float(sample[3])))

    plot_steering_angle_histogram(steering_angles, title='Before augmentation of steering angles', show=False)

    print('Found images with curves (|steering angles| >= {:.1f}°): {:d}'.format(steering_angle_threshold, len(samples_curves)))
    visualize_data_set(samples_curves, title='Images with |steering angle| >= {:.1f}°'.format(steering_angle_threshold), dataset_path=CFG_DATASET_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training and Evaluation')

    parser.add_argument(
        '-vd', '--visualize-datasets',
        help='Visualizes random dataset samples.',
        dest='visualize_dataset_csv',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-vsd', '--visualize-splitted.datasets',
        help='Visualizes random splitted training and validation dataset samples.',
        dest='visualize_splitted_dataset_csv',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-a', '--augment-dataset',
        help='Augments the dataset by flipping images with curves horizontally.',
        dest='augment_dataset',
        action='store_true'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(1)

    if args.visualize_dataset_csv:
        # Prepare data sets and show random training and validation sample
        print('Preparing training and validation datasets...', end='', flush=True)
        samples = prepare_datasets(args.visualize_dataset_csv, 0.0)
        print('done')

        print('Show random dataset samples:')
        visualize_data_set(samples, title='Dataset', dataset_path=CFG_DATASET_PATH)

    if args.visualize_splitted_dataset_csv:
        # Prepare data sets and show random training and validation samples
        print('Preparing training and validation datasets...', end='', flush=True)
        train_samples, validation_samples = prepare_datasets(args.visualize_splitted_dataset_csv, VALIDATION_SET_SIZE)
        print('done')

        print('Show random training samples:')
        visualize_data_set(train_samples, title='Training Dataset', dataset_path=CFG_DATASET_PATH)
        print('Show random validation samples:')
        visualize_data_set(validation_samples, title='Validation Dataset', dataset_path=CFG_DATASET_PATH)

    elif args.augment_dataset:
        # augment dataset
        augment_dataset('./data', './data')
