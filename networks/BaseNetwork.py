import pickle
import cv2
import math
import numpy as np
from sklearn.utils import shuffle


class BaseNetwork:
    """ Provides the base routines to train a network architecture and to store the training results. """

    input_width = 0                  # Width of the input layer
    input_height = 0                 # Height of the input layer
    input_depth = 0                  # Depth of the input layer (e.g. number of channels of an image)
    nb_classes = 1                   # Number of output classes resp. number of regression values
    regression = False               # If true, the network is setup as regression problem. Otherwise for classification.
    crop_top = 0                     # Number of pixels the image is cropped from top row.
    crop_bottom = 0                  # Number of pixels the image is cropped from bottom row.
    steering_angle_correction = 0.0  # Correction of steering angles for left and right images in degrees.
    weights_path = ''                # Path to trained model weights.
    model = None                     # Keras model
    batch_size = 128                 # Batch size for training
    train_generator = None           # Generator for training date
    validation_generator = None      # Generator for validation data
    train_samples = None             # Training dataset (links to images only!)
    validation_samples = None        # Validation dataset (links to images only!)
    path_to_image_data = ''          # Path to image data (replaces the path in the train/valid samples)
    history_object = None            # History object contains the loss and val_loss data after network training

    def __init__(self, input_width, input_height, input_depth, nb_classes, regression=False,
                 crop_top=0, crop_bottom=0, steering_angle_correction=0.0, weights_path=None):
        """ Initializes the base network.
        
        :param input_width:   Width of the input image.
        :param input_height:  Height if the input image.
        :param input_depth:   Depth of the input image (e.g. number of channels).
        :param nb_classes:    Number of unique classes (class labels) in the dataset. In case of a regression set, the 
                              number of regression outputs.
        :param regression:    If true the output layer is configured for a regression problem. If false the output
                              is configured with a softmax function.
        :param crop_top:      If >0 the image will be cropped from top row by given number of pixels.
        :param crop_bottom:   If >0 the image will be cropped from bottom by given number of pixels.
        :param steering_angle_correction: Correction for left and right image steering angles in degree.
        :param weights_path:  Path to trained model parameters. If set, the model will be initialized by these parameters.
        """

        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.nb_classes = nb_classes
        self.regression = regression
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.steering_angle_correction = steering_angle_correction
        self.weights_path = weights_path

    def generator(self, samples, batch_size=128):
        """ Generator
         
        :param samples:    Samples which shall be loaded into memory. 
        :param batch_size: Batch size for actual run.
        
        :return: Returns x_train and y_train.
        """

        nb_samples = len(samples)

        while 1:  # loop forever so the generator never terminates
            shuffle(samples)

            for offset in range(0, nb_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []

                for batch_sample in batch_samples:
                    # load images
                    center_image = cv2.imread(self.path_to_image_data + '/' + batch_sample[0].lstrip())
                    left_image = cv2.imread(self.path_to_image_data + '/' + batch_sample[1].lstrip())
                    right_image = cv2.imread(self.path_to_image_data + '/' + batch_sample[2].lstrip())

                    # adjust steering angles for left and right images
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle + math.radians(self.steering_angle_correction)
                    right_angle = center_angle - math.radians(self.steering_angle_correction)

                    images.append(center_image)
                    images.append(left_image)
                    images.append(right_image)

                    angles.append(center_angle)
                    angles.append(left_angle)
                    angles.append(right_angle)

                # convert to numpy arrays
                x_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(x_train, y_train)

    def setup_training_validation_generators(self, train_samples, validation_samples, path_to_image_data,
                                             batch_size=128):
        """ Setup the generators for training and validation samples.
        
        :param train_samples:       Training samples reference (not loaded into memory!).
        :param validation_samples:  Validation samples reference (not loaded into memory!).
        :param path_to_image_data:  Path to image data. This replaces the path in the train/valid sample list.
        :param batch_size:          Size of each batch (number of samples). Default = 128 samples.
        """
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.batch_size = batch_size
        self.path_to_image_data = path_to_image_data

        self.train_generator = self.generator(self.train_samples, batch_size=self.batch_size)
        self.validation_generator = self.generator(self.validation_samples, batch_size=self.batch_size)

    def train(self, nb_epochs, verbose=1):
        """
        Trains the network with given number of epochs.
        
        :param nb_epochs: Number of epochs the model will be trained.
        :param verbose:   0 = no logging, 1 = progress bar, 2 = one log line per epoch, Default = 1
        
        :return: Returns the training history data ['loss', 'val_loss'].
        """
        self.model.compile(optimizer='adam', loss='mse')
        self.history_object = self.model.fit_generator(self.train_generator, samples_per_epoch=len(self.train_samples),
                                                       validation_data=self.validation_generator,
                                                       nb_val_samples=len(self.validation_samples),
                                                       nb_epoch=nb_epochs, verbose=verbose)

        return self.history_object.history

    def save_history(self, filename='history.obj'):
        """ Saves the history ['loss', 'val_loss'] to a pickled file.
        
        :param filename: Path and filename to store pickled history data in.
        """
        fp = open(filename, 'wb')
        pickle.dump(self.history_object.history, fp)
        fp.close()
        print('History saved to {:s}'.format(filename))

    def safe_model(self, filename='model.h5'):
        """ Saves the trained model.
        
        :param filename: Path and filename to store the trained model.
        """
        self.model.save(filename)
        print('Model saved to {:s}'.format(filename))
