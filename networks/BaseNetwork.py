import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
import DataAugmentation as da
from keras.optimizers import adam
# TODO: from keras.utils.visualize_util import plot

class BaseNetwork:
    """ Provides the base routines to train a network architecture and to store the training results. """

    model_name = ''                  # Name of the model (e.g. NVIDIA_CNN or LeNet5)
    input_width = 0                  # Width of the input layer
    input_height = 0                 # Height of the input layer
    input_depth = 0                  # Depth of the input layer (e.g. number of channels of an image)
    nb_classes = 1                   # Number of output classes resp. number of regression values
    regression = False               # If true, the network is setup as regression problem. Otherwise for classification
    steering_angle_correction = 0.0  # Correction of steering angles for left and right images
    skip_rate_zero_angles = 0.0      # Reduces zero angles data sets by skip rate [0..1]
    angle_threshold = 0.0            # Take left, right and flip images with |steering angle| >= threshold [0°..25°]
    weights_path = ''                # Path to trained model weights
    model = None                     # Keras model
    batch_size = 128                 # Batch size for training
    train_generator = None           # Generator for training date
    validation_generator = None      # Generator for validation data
    train_samples = None             # Training dataset (links to images only!)
    validation_samples = None        # Validation dataset (links to images only!)
    nb_train_samples = 0             # Total number if training samples after augmentation (produced by generator)
    nb_validation_samples = 0        # Total number if validation samples after augmentation (produced by generator)
    path_to_image_data = ''          # Path to image data (replaces the path in the train/valid samples)
    history_object = None            # History object contains the loss and val_loss data after network training
    roi = None                       # Region of interesst which will be cropped before feeded into the first model layer
    verbose = 0                      # If >0 verbose mode (prints internal debug information)


    def __init__(self, model_name, input_width, input_height, input_depth, nb_classes, regression=False,
                 roi=None, steering_angle_correction=0.0, skip_rate_zero_angles=0.0, angle_threshold=0.0, weights_path=None):
        """ Initializes the base network.
        
        :param model_name:    Name of the model (e.g NvidiaCNN, LeNet5, etc).
        :param input_width:   Width of the input image. If smaller than original image, the image will be resized.
        :param input_height:  Height of the input image. If smaller than original image, the image will be resized.
        :param input_depth:   Depth of the input image (e.g. number of channels).
        :param nb_classes:    Number of unique classes (class labels) in the dataset. In case of a regression set, the 
                              number of regression outputs.
        :param regression:    If true the output layer is configured for a regression problem. If false the output
                              is configured with a softmax function.
        :param roi:           Region of interest which will be cropped [x0, y0, x1, y1].
        :param steering_angle_correction: Correction for left and right image steering angles in degree.
        :param skip_rate_zero_angles: Reduces total amount of samples with 0° steering angle by given percentage. 
                                      (0.0 = no reduction, 1.0 = remove all)
        :param angle_threshold: Take left, right and flipe images with |steering angle| >= threshold [0°..25°].
        :param weights_path:  Path to trained model parameters. If set, the model will be initialized by these parameters.
        """

        self.model_name = model_name
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.nb_classes = nb_classes
        self.regression = regression
        self.roi = roi
        self.steering_angle_correction = steering_angle_correction / 25.    # normalize angle from -25°..25° to -1..1
        self.skip_rate_zero_angles = skip_rate_zero_angles
        self.angle_threshold = angle_threshold / 25.                        # normalize angle from -25°..25° to -1..1
        self.weights_path = weights_path

    @staticmethod
    def preprocess_image(image, width, height, roi):
        """ Pre-processing pipeline for model input image. The method applies the following steps:
                
        :param image:  Image which to be preprocessed (Input format: RGB coded image).
        :param width:  Width of the model input image. If smaller than original image, the image will be resized.
        :param height: Height of the model input image. If smaller than original image, the image will be resized.
        :param roi:    Region of interest which will be cropped [x0, y0, x1, y1].
        
        :return: Returns the pre-processed image.
        """

        return da.DataAugmentation.crop_image(image, roi, (width, height))

    def get_number_genertor_samples(self, samples, reduce_zero_angles=True):
        """ Determines the total number of samples the generator produces.
        
        ATTENTION: 
        Make sure the 'skip_rate_zero_angles' and 'angle_threshold' is set correctly. 
        
        :param samples:             Training or validation samples.
        :param reduce_zero_angles: If true, zereo angles will be reduced before counting.
        
        :return: Returns the total number of samples the generator produces base on its input samples.
        """

        if reduce_zero_angles:
            samples_reduced = da.DataAugmentation.reduce_zero_steering_angles(samples, self.skip_rate_zero_angles)
        else:
            samples_reduced = samples

        nb_angle_theshold = 0
        for sample in samples_reduced:
            if abs(float(sample[3])) >= self.angle_threshold:
                nb_angle_theshold += 1

        return len(samples_reduced) * 2 + (nb_angle_theshold * 5)

    def generator(self, samples, batch_size=128):
        """ Generator
         
        :param samples:    Samples which shall be loaded into memory. 
        :param batch_size: Batch size for actual run.
        
        :return: Returns x_train and y_train.
        """

        # reduce total amount of scenes with 0° steering angles in order to avoid bias towards straights
        samples_reduced = da.DataAugmentation.reduce_zero_steering_angles(samples, self.skip_rate_zero_angles)
        shuffle(samples_reduced)

        nb_samples = len(samples_reduced)
        nb_gen_samples = self.get_number_genertor_samples(samples_reduced, reduce_zero_angles=False)
        nb_total_samples = 0

        while 1:  # loop forever so the generator never terminates
            for offset in range(0, nb_samples, batch_size):
                batch_samples = samples_reduced[offset:offset + batch_size]

                nb_batch_samples = 0
                images = []
                angles = []
                crop_size = (self.input_width, self.input_height)

                for batch_sample in batch_samples:
                    center_angle = float(batch_sample[3])

                    # cv2.imread returns BGR images, convert to RGB because simulator delivers RGB images
                    center_image = cv2.imread(self.path_to_image_data + '/' + batch_sample[0].lstrip())
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    images.append(da.DataAugmentation.crop_image(center_image, self.roi, crop_size))
                    angles.append(center_angle)
                    nb_batch_samples += 1

                    # take left and right images with |center steering angle| >= threshold only
                    if abs(center_angle) >= self.angle_threshold:
                        # add left and right image
                        left_image = cv2.imread(self.path_to_image_data + '/' + batch_sample[1].lstrip())
                        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                        images.append(da.DataAugmentation.crop_image(left_image, self.roi, crop_size))
                        left_angle = center_angle + self.steering_angle_correction
                        angles.append(left_angle)
                        nb_batch_samples += 1

                        right_image = cv2.imread(self.path_to_image_data + '/' + batch_sample[2].lstrip())
                        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                        images.append(da.DataAugmentation.crop_image(right_image, self.roi, crop_size))
                        right_angle = center_angle - self.steering_angle_correction
                        angles.append(right_angle)
                        nb_batch_samples += 1

                        # flip center, left and right image
                        flipped_image, flipped_angle = da.DataAugmentation.flip_image_horizontally(center_image, center_angle)
                        images.append(da.DataAugmentation.crop_image(flipped_image, self.roi, crop_size))
                        angles.append(flipped_angle)
                        nb_batch_samples += 1

                        flipped_image, flipped_angle = da.DataAugmentation.flip_image_horizontally(left_image, left_angle)
                        images.append(da.DataAugmentation.crop_image(flipped_image, self.roi, crop_size))
                        angles.append(flipped_angle)
                        nb_batch_samples += 1

                        flipped_image, flipped_angle = da.DataAugmentation.flip_image_horizontally(right_image, right_angle)
                        images.append(da.DataAugmentation.crop_image(flipped_image, self.roi, crop_size))
                        angles.append(flipped_angle)
                        nb_batch_samples += 1

                    # add randomly translated, rotated and perspective transformed center images
                    augmentation = np.random.randint(low=0, high=1)

                    if augmentation == 0:
                        new_center_image, new_center_angle = da.DataAugmentation.augment_translation(center_image, center_angle, [20, 20])
                    elif augmentation == 1:
                        new_center_image, new_center_angle = da.DataAugmentation.augment_rotation(center_image, center_angle, 10.)
                    else:
                        new_center_image, new_center_angle = da.DataAugmentation.augment_perpective_transformation(center_image, center_angle, [40, 50])

                    images.append(da.DataAugmentation.crop_image(new_center_image, self.roi, crop_size))
                    angles.append(new_center_angle)
                    nb_batch_samples += 1

                nb_total_samples += nb_batch_samples

                if self.verbose > 0:
                    print(' Generator: nb_samples: {:d} nb_batch_samples: {:d} nb_total_samples: {:d}/{:d}'.format(nb_samples, nb_batch_samples, nb_total_samples, nb_gen_samples))

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

        self.nb_train_samples = self.get_number_genertor_samples(train_samples)
        self.nb_validation_samples = self.get_number_genertor_samples(validation_samples)

        self.train_generator = self.generator(self.train_samples, batch_size=self.batch_size)
        self.validation_generator = self.generator(self.validation_samples, batch_size=self.batch_size)

    def train(self, nb_epochs, learning_rate=0.001, verbose=1):
        """
        Trains the network with given number of epochs.
        
        :param nb_epochs:     Number of epochs the model will be trained.
        :param learning_rate: Learning rate. Default = 0.001
        :param verbose:       0 = no logging, 1 = progress bar, 2 = one log line per epoch, Default = 1
        
        :return: Returns the training history ['loss', 'val_loss'].
        """
        optimizer = adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        self.history_object = self.model.fit_generator(self.train_generator,
                                                       samples_per_epoch=self.nb_train_samples,
                                                       validation_data=self.validation_generator,
                                                       nb_val_samples=self.nb_validation_samples,
                                                       nb_epoch=nb_epochs,
                                                       verbose=verbose)
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

    def summary(self):
        """ Print a model representation summary."""
        self.model.summary()

    # def save_model_graph(self, filename):
    #     """ Saves the model graph to a PNG file.
    #
    #     :param filename: Output filename (*.png).
    #     """
    #     plot(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
