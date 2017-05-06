import argparse
import csv
import cv2
import sys
import os
import math
from random import randint
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import signal
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
    fig, ax1 = plt.subplots()
    n, bins, patches = ax1.hist(steering_angles, 100, normed=True, facecolor='green', edgecolor='black', alpha=0.75,
                                histtype='bar', rwidth=0.85, label='steering angles')
    ax1.set_xlabel('steering angle [degree]')
    ax1.set_ylabel('frequency')
    ax1.grid(True)

    # add a line showing the expected distribution
    (mu, sigma) = norm.fit(steering_angles)
    y = mlab.normpdf(bins, mu, sigma)
    # ax2 = ax1.twinx()
    ax1.plot(bins, y, 'r--', linewidth=1.5, label='norm')
    # ax2.set_ylim(ymin=0)

    ax1.legend()
    fig.suptitle(r'$\mathrm{%s}\ (\mu=%.4f,\ \sigma=%.4f)$' % (title.replace(' ', '\ '), mu, sigma))

    if show:
        plt.show()

def plot_odometry(samples, title='', show=True):
    """ Plot odometry (steering angle, throttle, brake and speed) diagram.
    
    :param samples: Samples in format [center, left, right, steering, throttle, brake, speed].
    :param title:   Title of the diagram
    :param show:    If true, `plot.show()` is called at the end.
    """

    # prepare odometry data
    angles = []
    throttle = []
    brake = []
    speed = []

    for sample in samples:
        angles.append(float(sample[3]) * 25.)
        throttle.append(float(sample[4]))
        brake.append(float(sample[5]))
        speed.append(float(sample[6]))

    fig, axarr = plt.subplots(4, figsize=[10, 8], sharex=True)

    # plot steering angle in degree
    axarr[0].plot(angles, label='steering angle')
    axarr[0].set_ylabel('steering angle [°]')
    axarr[0].grid(True)

    # plot throttle
    axarr[1].plot(throttle, label='throttle')
    axarr[1].set_ylabel('throttle')
    axarr[1].grid(True)

    # plot brake
    axarr[2].plot(brake, label='brake')
    axarr[2].set_ylabel('brake')
    axarr[2].grid(True)

    # plot speed
    axarr[3].plot(speed, label='speed')
    axarr[3].set_xlabel('#sample')
    axarr[3].set_ylabel('speed [mp/h]')
    axarr[3].grid(True)

    fig.suptitle(title)

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

    #
    # Plot center, left and right random image
    #
    nb_shown_images = 5
    nb_samples = len(samples)

    print('Title:             {:s}'.format(title))
    print('Dataset CSV file:  {:s}'.format(dataset_path))
    print('Number of samples: {:d}'.format(nb_samples))
    print('')
    print("Plotting diagrams...", end='', flush=True)

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
        center_angles.append(float(sample[3]) * 25.)

    plot_steering_angle_histogram(center_angles, title='Histogram of center steering angle', show=False)

    print('done')
    print('Close the figures to continue...')
    plt.show()


def smooth_steering_angle(csv_file):
    """ Smooths the steering angle.
    
    :param csv_file: CSV file containing the measurements (steering angle shall be at position 3).
    """

    print('Preparing training and validation datasets...', end='', flush=True)
    samples = prepare_datasets(csv_file)
    print('done')

    filename = csv_file[0:csv_file.rfind('.')] + '_smoothed.csv'
    print('Smooth steering angles...', end='', flush=True)

    angles = np.empty(0)

    for sample in samples:
        angles = np.append(angles, float(sample[3]))

    # interpolate data
    x = np.arange(len(angles))
    window = signal.hann(15)
    angles_filtered = signal.convolve(angles, window, mode='same') / sum(window)

    # write filtered steering angle to csv file
    file = open(filename, 'w')
    fieldnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for idx, sample in enumerate(samples):
        writer.writerow({'center': sample[0],
                         'left': sample[1],
                         'right': sample[2],
                         'steering': str(angles_filtered[idx]),
                         'throttle': sample[4],
                         'brake': sample[5],
                         'speed': sample[6]})

    print('done')
    print('Close the figures to continue...')

    # plot steering angle in degree
    fig = plt.figure(figsize=(10, 3))
    plt.plot(angles * 25., 'b', label='raw steering angle')
    plt.plot(angles_filtered * 25., 'r', label='interpolated')
    plt.ylabel('steering angle [°]')
    plt.grid(True)
    plt.title('Smoothed Steering Angle')
    plt.legend()
    plt.show()


def merge_csv_files(source_file_1, source_file_2, merged_file):
    """ Merges two CSV files.
    
    :param source_file_1: Path and name of first CSV file.
    :param source_file_2: Path and name of second CSV file.
    :param merged_file:   Path and name of merged CSV file.
    """

    # check for valid path and filenames
    if not os.path.isfile(source_file_1):
        print('ERROR: Source file 1 \'{:s}\' not found.'.format(source_file_1))
        exit(-1)

    if not os.path.isfile(source_file_2):
        print('ERROR: Source file 2 \'{:s}\' not found.'.format(source_file_2))
        exit(-1)

    if os.path.isfile(merged_file):
        print('ERROR: Merged file \'{:s}\' already exists. Rename merge file.'.format(merged_file))

    # merge files
    print('Merging CSV files: {:s} + {:s} ==> {:s} ...'.format(source_file_1, source_file_2, merged_file), end='', flush=True)
    header_saved = False

    with open(merged_file, 'wb') as f_merged:
        for filename in [source_file_1, source_file_2]:
            with open(filename, 'rb') as f_in:
                header = next(f_in)
                if not header_saved:
                    f_merged.write(header)
                    header_saved = True
                for line in f_in:
                    f_merged.write(line)
    print('done')


def augment_dataset(csv_file, source_directory, destination_directory,
                    reduce_zero_steering_angles=0.,
                    steering_angle_threshold=-1.,
                    write_images=True):
    """ Augments the data set.
     
     Methods:
      - Flip images with curves (|steering angle| >= threshold) horizontally.
      - Reduce total number of images with 0° steering angle
    
    :param csv_file:                 CSV file describing the dataset to augment.
    :param source_directory:         Source path of data to augment without a \ at the end.        
    :param destination_directory:    Destination path without a \ at the end to store augmented data.
    :param reduce_zero_steering_angles: Reduces total amount of samples with 0° steering angle by given percentage. 
                                        (0.0 = no reduction, 1.0 = remove all)
    :param steering_angle_threshold: Flip images with |steering angle| >= threshold [0°..25°].
    :param write_images;             If false, the augmented images won't be saved. Only relevant for statistics!
    """

    # check if all source and destination directories exist
    augmented_image_sub_directory = 'IMG_augmented'
    augmented_image_directory = destination_directory + '/' + augmented_image_sub_directory
    augmented_csv_log_file = destination_directory + '/augmented_log.csv'
    steering_angle_threshold_normed = steering_angle_threshold / 25.        # steering angles are normalized: -1 .. 1 = -25° .. 25°

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
    samples = np.array(prepare_datasets(csv_file, 0.0))

    # find images with curves by |steering angle| >= threshold
    samples_curves = []

    print('Number of samples:            {:d}'.format(len(samples)))
    print('Source CVS file:              {:s}'.format(csv_file))
    print('Source directory:             {:s}'.format(source_directory))
    print('Augmented CVS file:           {:s}'.format(augmented_csv_log_file))
    print('Augmented images:             {:s}'.format(augmented_image_directory))

    if reduce_zero_steering_angles < 1.:
        print('Reduce 0° steering angles by: {:.2f}%'.format(reduce_zero_steering_angles * 100.))
    else:
        print('Reduce 0° steering angles by: deactivated')

    if steering_angle_threshold >= 0.:
        print('Steering angle threshold:     {:.2f} ({:.2f}°)'.format(steering_angle_threshold_normed, steering_angle_threshold))
    else:
        print('Steering angle threshold:     deactivated')

    print('')
    print('Augment dataset...', end='', flush=True)

    # augment dataset
    csv_file = open(augmented_csv_log_file, 'w')
    fieldnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    #
    # Reduce total amount of images with 0° steering angle. Skip if 1.
    #
    zero_angle_index = np.empty(0, dtype='int32')

    if reduce_zero_steering_angles < 1.:
        for idx, sample in enumerate(samples):
            if float(sample[3]) == 0.:
                zero_angle_index = np.append(zero_angle_index, idx)

    nb_samples_to_delete = math.ceil(len(zero_angle_index) * reduce_zero_steering_angles)
    idx = np.arange(nb_samples_to_delete)
    np.random.shuffle(idx)
    samples_reduced = np.delete(samples, zero_angle_index[idx], 0)

    #
    # Flip images with steering angles >= threshold. Skip if < 0.
    #
    for sample in samples_reduced:
        if abs(float(sample[3])) >= steering_angle_threshold_normed and steering_angle_threshold >= 0.0:
            samples_curves.append(sample)

            # load and flip center, left and right image horizontally
            center_image_filename = sample[0].split('/')[-1]
            center_image_path = source_directory + '/' + sample[0]
            center_image = cv2.imread(center_image_path)

            if center_image is None:
                print('Center image {:s} not found.'.format(center_image_path))
                exit(-1)
            elif write_images:
                cv2.imwrite(augmented_image_directory + '/flipped_' + center_image_filename, cv2.flip(center_image, 1))

            left_image_filename = sample[1].split('/')[-1]
            left_image_path = source_directory + '/' + sample[1]
            left_image = cv2.imread(left_image_path)

            if left_image is None:
                print('Left image {:s} not found.'.format(sample[1]))
                exit(-1)
            elif write_images:
                cv2.imwrite(augmented_image_directory + '/flipped_' + left_image_filename, cv2.flip(left_image, 1))

            right_image_filename = sample[2].split('/')[-1]
            right_image_path = source_directory + '/' + sample[2]
            right_image = cv2.imread(right_image_path)

            if right_image is None:
                print('Right image {:s} not found.'.format(sample[2]))
                exit(-1)
            elif write_images:
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

    print('Number of samples before augmentation:                    {:5d}'.format(len(samples)))

    if reduce_zero_steering_angles < 1.:
        print('Number of removed 0° steering angle samples:               {:d}'.format(nb_samples_to_delete))

    if steering_angle_threshold >= 0.:
        print('Number of augmented samples(|steering angles| >= {:.1f}°): {:5d}'.format(steering_angle_threshold,
                                                                                        len(samples_curves)))

    print('Number of samples after augmentation:                     {:5d}'.format(len(augmented_samples)))
    print('')

    augmented_steering_angles = []
    steering_angles = []

    for sample in augmented_samples:
        augmented_steering_angles.append(float(sample[3]) * 25.)

    plot_steering_angle_histogram(augmented_steering_angles, title='Augmented steering angles', show=False)

    for sample in samples:
        steering_angles.append(float(sample[3]) * 25.)

    plot_steering_angle_histogram(steering_angles, title='Before augmentation of steering angles', show=False)

    visualize_data_set(samples_curves, title='Images with |steering angle| >= {:.1f}°'
                       .format(steering_angle_threshold), dataset_path=CFG_DATASET_PATH)


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
        '-vo', '--visualize-odometry',
        help='Visualizes odometry data (angle, speed, brake, throttle) in a diagram.',
        dest='visualize_odometry_dataset_csv',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-s', '--smooth-steering_angles',
        help='Smoothes the steering angles.',
        dest='smoth_steering_angle_csv',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-m', '--merge-data-sets',
        help='Merges two dataset CSV files (file-1, file-2, file-out).',
        dest='merge_dataset_csv',
        nargs=3,
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-a', '--augment-dataset',
        help='Augments the dataset (val0 = CVS file, val1 = reduce 0° steering angles [perc], val2 = flip steering angle threshold [°]), val3 = write image files',
        dest='augment_dataset',
        nargs=4,
        metavar='VALUE'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(1)

    elif args.visualize_dataset_csv:
        # Prepare data sets and show random training and validation sample
        print('Preparing training and validation datasets...', end='', flush=True)
        samples = prepare_datasets(args.visualize_dataset_csv, 0.0)
        print('done')

        visualize_data_set(samples, title='Original Dataset', dataset_path=CFG_DATASET_PATH)

    elif args.visualize_odometry_dataset_csv:
        # plot steering angle, speed, throttle, brake and speed
        print('Preparing datasets...', end='', flush=True)
        samples = prepare_datasets(args.visualize_odometry_dataset_csv, 0.0)
        print('done')
        print('Close the figures to continue...')

        plot_odometry(samples, title='Odometry Data')

    elif args.smoth_steering_angle_csv:
        # smooth steering angles
        smooth_steering_angle(args.smoth_steering_angle_csv)

    elif args.visualize_splitted_dataset_csv:
        # Prepare data sets and show random training and validation samples
        print('Preparing training and validation datasets...', end='', flush=True)
        train_samples, validation_samples = prepare_datasets(args.visualize_splitted_dataset_csv, VALIDATION_SET_SIZE)
        print('done')

        print('Show random training samples:')
        visualize_data_set(train_samples, title='Training Dataset', dataset_path=CFG_DATASET_PATH)
        print('Show random validation samples:')
        visualize_data_set(validation_samples, title='Validation Dataset', dataset_path=CFG_DATASET_PATH)

    elif args.merge_dataset_csv:
        # merge two CVS files
        merge_csv_files(args.merge_dataset_csv[0], args.merge_dataset_csv[1], args.merge_dataset_csv[2])

    elif args.augment_dataset:
        # augment dataset
        augment_dataset(args.augment_dataset[0], './data', './data',
                        reduce_zero_steering_angles=float(args.augment_dataset[1]),
                        steering_angle_threshold=float(args.augment_dataset[2]),
                        write_images=args.augment_dataset[3])
