import argparse
import csv
import cv2
import sys
import os
import math
from random import randint, shuffle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import signal
from sklearn.model_selection import train_test_split
from keras.models import load_model
from networks.BaseNetwork import BaseNetwork
from networks.NvidiaFull import NvidiaFull
from Filter import Filter
from model import CFG_DATASET_PATH, VALIDATION_SET_SIZE, ROI, IMAGE_WIDTH, IMAGE_HEIGHT
from model import STEERING_ANGLE_CORRECTION

np.random.seed()

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


def plot_steering_angle_histogram(steering_angles, title='Histogram of steering angle', nb_bins=200, show=True):
    """ Plot steering angle histogram.
    
    :param steering_angles: Array containing steering angles in degree.
    :param title:           Title of the histogram.
    :param nb_bins:         Number of bins.
    :param show:            If true, `plot.show()` is called at the end.
    """
    fig, ax1 = plt.subplots()
    n, bins, patches = ax1.hist(steering_angles, nb_bins, normed=False, facecolor='green', edgecolor='black', alpha=0.75,
                                histtype='bar', rwidth=0.85, label='steering angles')
    ax1.set_xlabel('steering angle [degree]')
    ax1.set_ylabel('frequency')
    ax1.grid(True)
    ax1.legend()
    fig.suptitle(title)

    if show:
        plt.show()


def plot_normed_steering_angle_histogram(steering_angles, title='Histogram of steering angle', nb_bins=200, show=True):
    """ Plot normed steering angle histogram with expected distribution.
    
    :param steering_angles: Array containing steering angles in degree.
    :param title:           Title of the histogram.
    :param nb_bins:         Number of bins.
    :param show:            If true, `plot.show()` is called at the end.
    """
    fig, ax1 = plt.subplots()
    n, bins, patches = ax1.hist(steering_angles, nb_bins, normed=True, facecolor='green', edgecolor='black', alpha=0.75,
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
    math_title = r'$\mathrm{%s}$' % title.replace(' ', '\ ')
    math_title += '\n'
    math_title += r'$(\mu=%.4f,\ \sigma=%.4f,\ N=%d)$' % (mu, sigma, len(steering_angles))
    fig.suptitle(math_title)

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


def visualize_generator_data_set(samples, title='', verbose=0, show=True):
    """ Visualizes the dataset generated by the generator.
    
    REMARK:
    In order to enable plotting in PyCharm add environment variable 'DISPLAY = True'.
    
    :param samples: Samples in format [center, left, right, steering, throttle, brake, speed].
    :param title:   Title shown on the image figure.
    :param verbose: 0 = no debug info, 1 = verbose mode (prints internal debug info), 2 = show generator images
    :param show:    If true, `plot.show()` is called at the end.
    """

    angles = []
    angles_generator = np.array([], dtype=np.float64)

    for sample in samples:
        angles.append(float(sample[3]) * 25.)

    # setup full NVIDIA CNN model architecture
    network = NvidiaFull(input_depth=3, input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH, regression=True,
                         nb_classes=1, roi=ROI, steering_angle_correction=STEERING_ANGLE_CORRECTION)

    network.path_to_image_data = './data'
    network.verbose = verbose

    generator = network.generator(samples)
    nb_samples = len(samples)

    print("Preparing dataset by generator...")

    for X_train, y_train in generator:
        angles_generator = np.append(angles_generator, y_train * 25.)

        if len(angles_generator) >= nb_samples:
            break

    print('Number of samples in dataset: {:d}'.format(len(angles)))
    print('Number of samples generated:  {:d}'.format(len(angles_generator)))
    print('Close the figures to continue...')

    plot_normed_steering_angle_histogram(angles, title='Normed histogram of steering angles in dataset', show=False)
    plot_normed_steering_angle_histogram(angles_generator, title='Normed histogram of steering angles from generator', show=False)
    plot_steering_angle_histogram(angles, title='Histogram of steering angles in dataset', show=False)
    plot_steering_angle_histogram(angles_generator, title='Histogram of steering angles from generator', show=show)


def compare_model_prediction(samples, model, dataset_path='', show=True):
    """ Compares the model steering angle prediction with the real steering angle (ground truth measurement data).
    
    :param samples:      Samples in format [center, left, right, steering, throttle, brake, speed].
    :param model:        Trained model.
    :param dataset_path: Path to data directory without '/' at the end.
    :param show:         If true, `plot.show()` is called at the end.
    """

    # prepare odometry data
    angles_center = []
    angles_left = []
    angles_right = []
    angles_center_predicted = []
    angles_left_predicted = []
    angles_right_predicted = []
    angles_center_predicted_filtered = []
    angles_left_predicted_filtered = []
    angles_right_predicted_filtered = []

    filter_center = Filter()
    filter_left = Filter()
    filter_right = Filter()

    print('Predicting steering angles...', end='', flush=True)

    for sample in samples:
        angles_center.append(float(sample[3]) * 25.)
        angles_left.append(float(sample[3]) * 25. + 4.6)
        angles_right.append(float(sample[3]) * 25. - 4.6)

        # load random center, left and right image
        center_image = cv2.imread(dataset_path + '/' + sample[0].lstrip())
        left_image = cv2.imread(dataset_path + '/' + sample[1].lstrip())
        right_image = cv2.imread(dataset_path + '/' + sample[2].lstrip())

        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        # resize and crop images to match model input size
        center_image = BaseNetwork.preprocess_image(center_image, IMAGE_WIDTH, IMAGE_HEIGHT, ROI)
        left_image = BaseNetwork.preprocess_image(left_image, IMAGE_WIDTH, IMAGE_HEIGHT, ROI)
        right_image = BaseNetwork.preprocess_image(right_image, IMAGE_WIDTH, IMAGE_HEIGHT, ROI)

        # predict steering angles
        angles_center_predicted.append(float(model.predict(center_image[None, :, :, :], batch_size=1)) * 25.)
        angles_left_predicted.append(float(model.predict(left_image[None, :, :, :], batch_size=1)) * 25.)
        angles_right_predicted.append(float(model.predict(right_image[None, :, :, :], batch_size=1)) * 25.)

        # filter predicted steering angles
        angles_center_predicted_filtered.append(filter_center.moving_average(angles_center_predicted[-1], 8))
        angles_left_predicted_filtered.append(filter_left.moving_average(angles_left_predicted[-1], 8))
        angles_right_predicted_filtered.append(filter_right.moving_average(angles_right_predicted[-1], 8))

    print('done')
    print('Close the figures to continue...')

    fig, axarr = plt.subplots(3, figsize=[10, 8], sharex=True)

    # plot center steering angle in degree
    axarr[0].plot(angles_center, label='steering angle')
    axarr[0].plot(angles_center_predicted, label='steering angle predicted')
    axarr[0].plot(angles_center_predicted_filtered, label='steering angle predicted filtered')
    axarr[0].set_ylabel('center steering angle [°]')
    axarr[0].grid(True)
    axarr[0].legend()

    # plot left steering angle in degree
    axarr[1].plot(angles_left, label='steering angle')
    axarr[1].plot(angles_left_predicted, label='steering angle predicted')
    axarr[1].plot(angles_left_predicted_filtered, label='steering angle predicted filtered')
    axarr[1].set_ylabel('left steering angle [°]')
    axarr[1].grid(True)
    axarr[1].legend()

    # plot right steering angle in degree
    axarr[2].plot(angles_right, label='steering angle')
    axarr[2].plot(angles_right_predicted, label='steering angle predicted')
    axarr[2].plot(angles_right_predicted_filtered, label='steering angle predicted filtered')
    axarr[2].set_ylabel('right steering angle [°]')
    axarr[2].grid(True)
    axarr[2].legend()

    fig.suptitle('Comparison of trained and predicted steering angle')

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
        cv2.rectangle(left_image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)
        axarr[i-1, 0].imshow(left_image)
        axarr[i-1, 0].axis('off')

        center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(center_image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)
        axarr[i-1, 1].imshow(center_image)
        axarr[i-1, 1].axis('off')

        right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(right_image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)
        axarr[i-1, 2].imshow(right_image)
        axarr[i-1, 2].axis('off')

    # set titles
    axarr[0, 0].set_title('left image')
    axarr[0, 1].set_title('center image')
    axarr[0, 2].set_title('right image')
    title = '{:s} (w={:d}, h={:d}) with ROI = [{:d}, {:d}, {:d}, {:d}]'.format(title, width, height,
                                                                               ROI[0], ROI[1], ROI[2], ROI[3])
    fig.suptitle(title)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05)

    #
    # plot steering angle histogram
    #
    center_angles = []

    for sample in samples:
        center_angles.append(float(sample[3]) * 25.)

    plot_normed_steering_angle_histogram(center_angles, title='Normed histogram of center steering angle', show=False)
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

    # check for valid path and filename
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


def reduce_samples_with_steering_angle(source_csv_file, destination_csv_file, min_angle, max_angle, skip_rate):
    """ Reduces the total number of samples in the specified range of angels by the skip rate.
         
    :param source_csv_file:        Source CSV file describing the dataset to reduce.
    :param destination_csv_file:   Destination CSV file containing the reduced dataset.
    :param min_angle:              Min angle of range in degree.
    :param max_angle:              Max angle of range in degree.
    :param skip_rate:              Reduction rate in percentage of total amount of samples within specified steering
                                   range (0.0 = no reduction, 1.0 = remove all).
    """

    # check for valid source and destination files
    if not os.path.exists(source_csv_file):
        print('ERROR: Source CSV file \'{:s}\' does not exists.'.format(source_csv_file))
        exit(-1)

    if os.path.exists(destination_csv_file):
        print('ERROR: Destination CSV file \'{:s}\' already exists.'.format(destination_csv_file))
        exit(-1)

    print('Reading dataset...', end='', flush=True)
    samples = np.array(prepare_datasets(source_csv_file, 0.0))
    print('done')

    print('Number of samples:      {:d}'.format(len(samples)))
    print('Source CVS file:        {:s}'.format(source_csv_file))
    print('Destination CVS file:   {:s}'.format(destination_csv_file))
    print('Angle range:            [{:f}, {:f}]'.format(min_angle, max_angle))
    print('Skip rate:              {:.2f}'.format(skip_rate))
    print('')
    print('Reduce dataset...', end='', flush=True)

    #
    # Reduce randomly total amount of samples within steering angle range
    #
    csv_file = open(destination_csv_file, 'w')
    fieldnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    skip_angle_index = np.empty(0, dtype='int32')

    for idx, sample in enumerate(samples):
        angle = float(sample[3]) * 25.
        if min_angle <= angle <= max_angle:
            skip_angle_index = np.append(skip_angle_index, idx)

    nb_samples_to_delete = math.ceil(len(skip_angle_index) * skip_rate)
    idx = np.arange(nb_samples_to_delete)
    np.random.shuffle(idx)
    samples_reduced = np.delete(samples, skip_angle_index[idx], 0)

    # write reduced samples into destination CSV file
    for sample in samples_reduced:
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
    # Plot reduced steering angle histogram
    #
    print('Number of samples within range:     {:5d}'.format(len(skip_angle_index)))
    print('Number of skipped samples:          {:5d}'.format(nb_samples_to_delete))
    print('Number of samples before reduction: {:5d}'.format(len(samples)))
    print('Number of samples after reduction:  {:5d}'.format(len(samples_reduced)))
    print('')
    print('Close the figures to continue...')

    steering_angles = []
    steering_angles_reduced = []

    for sample in samples_reduced:
        steering_angles_reduced.append(float(sample[3]) * 25.)

    plot_normed_steering_angle_histogram(steering_angles_reduced, title='Reduced steering angles', show=False)
    # plot_steering_angle_histogram(steering_angles_reduced, title='Reduced steering angles', show=False)

    for sample in samples:
        steering_angles.append(float(sample[3]) * 25.)

    plot_normed_steering_angle_histogram(steering_angles, title='Before augmentation of steering angles', show=True)
    # plot_steering_angle_histogram(steering_angles, title='Before augmentation of steering angles', show=True)


def balance_steering_angles(source_csv_file, destination_csv_file, max_samples_per_bin=200, bin_size=0.1):
    """ Balances the steering angles by histogram analysis.

    :param source_csv_file:       Source CSV file describing the dataset to reduce.
    :param destination_csv_file:  Destination CSV file containing the balanced dataset.
    :param max_samples_per_bin:   Max number of samples in each bin.
    :param bin_size:              Bin size in degree.
    """

    # check for valid source and destination files
    if not os.path.exists(source_csv_file):
        print('ERROR: Source CSV file \'{:s}\' does not exists.'.format(source_csv_file))
        exit(-1)

    if os.path.exists(destination_csv_file):
        print('ERROR: Destination CSV file \'{:s}\' already exists.'.format(destination_csv_file))
        exit(-1)

    print('Reading dataset...', end='', flush=True)
    samples = np.array(prepare_datasets(source_csv_file, 0.0))
    print('done')

    nb_bins = int(50./bin_size)

    print('Number of samples:      {:d}'.format(len(samples)))
    print('Source CVS file:        {:s}'.format(source_csv_file))
    print('Destination CVS file:   {:s}'.format(destination_csv_file))
    print('Max samples / bin:      {:d}'.format(max_samples_per_bin))
    print('Bin size:               {:.2f}'.format(bin_size))
    print('Number of bins:         {:d}'.format(nb_bins))
    print('')
    print('Balance dataset...', end='', flush=True)

    angles = np.array([])

    # get all steering angles
    for sample in samples:
        angles = np.append(angles,float(sample[3]) * 25.)

    # calculate histogram and sort angles according assigned bin
    hist, bin_edges = np.histogram(angles, bins=nb_bins)
    bin_idx = np.digitize(angles, bins=bin_edges) - 1
    angle_idx = np.arange(0, len(angles), 1)
    bin_idx_sorted = bin_idx.argsort(axis=0)
    bin_idx = bin_idx[bin_idx_sorted]
    angle_idx = angle_idx[bin_idx_sorted]

    bin_angle_idx_map = [[]] * (nb_bins + 1)         # contains the mapping between bin and angle idx
    buf = []
    last_idx = bin_idx[0]

    for i, idx in enumerate(bin_idx):
        if last_idx == idx:
            buf.append(angle_idx[i])
        else:
            bin_angle_idx_map[last_idx] = buf
            buf = []
            buf.append(angle_idx[i])
            last_idx = idx

    bin_angle_idx_map[last_idx] = buf

    # limit number of samples per bin (if > max_samples_per_bin then take random samples)
    remaining_angle_idx = []

    for angle_idx_list in bin_angle_idx_map:
        if len(angle_idx_list) > max_samples_per_bin:
            shuffle(angle_idx_list)
            remaining_angle_idx.append(angle_idx_list[0:max_samples_per_bin])
        elif len(angle_idx_list) > 0:
            remaining_angle_idx.append(angle_idx_list)

    remaining_angle_idx = sum(remaining_angle_idx, [])
    samples_balanced = samples[remaining_angle_idx]

    print('done')

    # write balances dataset into destination CSV file
    print('Save balanced dataset...', end='', flush=True)

    csv_file = open(destination_csv_file, 'w')
    fieldnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for sample in samples_balanced:
        writer.writerow({'center': sample[0],
                         'left': sample[1],
                         'right': sample[2],
                         'steering': sample[3],
                         'throttle': sample[4],
                         'brake': sample[5],
                         'speed': sample[6]})

    print('done')
    print('Close the figures to continue...')

    plot_steering_angle_histogram(angles, title='Steering angle histogram in original dataset (N={:d})'.format(len(angles)),
                                  nb_bins=nb_bins, show=False)
    plot_steering_angle_histogram(angles[remaining_angle_idx],
                                  title='Steering angle histogram in balanced dataset (N={:d})'.format(len(angles[remaining_angle_idx])),
                                  nb_bins=nb_bins, show=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation and model analysis toolbox')

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
        '-vg', '--visualize-generator',
        help='Visualizes the generator steering angle distribution.',
        dest='visualize_generator_dataset_csv',
        metavar='CSV_FILE'
    )

    parser.add_argument(
        '-v', '--verbose',
        help='Verbose (0=none, 1=console output, 2=show images)',
        dest='verbose',
        metavar='N'
    )

    parser.add_argument(
        '-cp', '--compare-prediction',
        help='Compares the model prediction with the ground truth data (H5-FILE CSV-FILE)',
        dest='compare_prediction_model_file',
        nargs=2,
        metavar='FILE'
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
        '-ra', '--reduce_angles',
        help='Reduces zero angles randomly (source CSV filer, destination CSV file, min angle, max angle, skip rate)',
        dest='reduce_angles',
        nargs=5,
        metavar='VALUE'
    )

    parser.add_argument(
        '-b', '--balance',
        help='Balances the steering angles (source CSV filer, destination CSV file, max number samples per bin, bin size)',
        dest='balance',
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

    elif args.visualize_generator_dataset_csv:
        # Visualizes the steering angle distribution provided by the generator.
        print('Preparing datasets...', end='', flush=True)
        samples = prepare_datasets(args.visualize_generator_dataset_csv, 0.0)
        print('done')

        visualize_generator_data_set(samples, title='Generator Steering Angle Histogram', verbose=int(args.verbose))

    elif args.compare_prediction_model_file:
        # Compares the model steering angle prediction with the ground truth data
        print('Preparing datasets...', end='', flush=True)
        samples = prepare_datasets(args.compare_prediction_model_file[1], 0.0)
        print('done')

        model = load_model(args.compare_prediction_model_file[0])
        compare_model_prediction(samples, model=model, dataset_path='./data')

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

    elif args.reduce_angles:
        # reduce randomly angles within range by skip rate
        reduce_samples_with_steering_angle(source_csv_file=args.reduce_angles[0],
                                           destination_csv_file=args.reduce_angles[1],
                                           min_angle=float(args.reduce_angles[2]),
                                           max_angle=float(args.reduce_angles[3]),
                                           skip_rate=float(args.reduce_angles[4]))

    elif args.balance:
        # balance steering angles by histogram analysis
        balance_steering_angles(source_csv_file=args.balance[0],
                                destination_csv_file=args.balance[1],
                                max_samples_per_bin=int(args.balance[2]),
                                bin_size=float(args.balance[3]))
