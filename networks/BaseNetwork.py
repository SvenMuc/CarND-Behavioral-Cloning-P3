import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
import DataAugmentation as da
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
    angle_threshold = 0.0            # Take left, right and flip images with |steering angle| >= threshold [0°..25°]
    weights_path = ''                # Path to trained model weights
    model = None                     # Keras model
    batch_size = 128                 # Batch size for training
    train_generator = None           # Generator for training date
    validation_generator = None      # Generator for validation data
    train_samples = None             # Training dataset (links to images only!)
    validation_samples = None        # Validation dataset (links to images only!)
    nb_samples_per_epoch = 0         # Number of sample trained per epoch
    nb_train_samples = 0             # Total number if training samples after augmentation (produced by generator)
    nb_validation_samples = 0        # Total number if validation samples after augmentation (produced by generator)
    path_to_image_data = ''          # Path to image data (replaces the path in the train/valid samples)
    history_object = None            # History object contains the loss and val_loss data after network training
    roi = None                       # Region of interesst which will be cropped before feeded into the first model layer
    verbose = 0                      # 1 = verbose mode (prints internal debug information), 2 = show generator images

    def __init__(self, model_name, input_width, input_height, input_depth, nb_classes, regression=False,
                 roi=None, steering_angle_correction=0.0, angle_threshold=0.0, weights_path=None):
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
        :param angle_threshold: Take left, right and flip images with |steering angle| >= threshold [0°..25°].
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
        self.angle_threshold = angle_threshold / 25.                        # normalize angle from -25°..25° to -1..1
        self.weights_path = weights_path

    @staticmethod
    def preprocess_image(image, width, height, roi):
        """ Pre-processing pipeline for model input image. The method applies the following steps:

            - crop and resize image
                
        :param image:  Image which shall be preprocessed (Input format: RGB coded image!).
        :param width:  Width of the model input image. If smaller than original image, the image will be resized.
        :param height: Height of the model input image. If smaller than original image, the image will be resized.
        :param roi:    Region of interest which will be cropped [x0, y0, x1, y1].
        
        :return: Returns the pre-processed image.
        """

        # TODO: image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return da.DataAugmentation.crop_image(image, roi, (width, height))

    def generator(self, samples, batch_size=128, augment=True):
        """ Generator
         
        :param samples:    Samples which shall be loaded into memory. 
        :param batch_size: Batch size for actual run.
        :param augment:    If true the dataset will be randomly augmented.
        
        :return: Returns x_train and y_train.
        """

        samples = shuffle(samples)

        nb_samples = len(samples)
        nb_total_samples = 0
        color_scheme = cv2.COLOR_BGR2RGB

        while 1:  # loop forever so the generator never terminates
            for offset in range(0, nb_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                nb_batch_samples = 0
                images = []
                angles = []
                crop_size = (self.input_width, self.input_height)
                i = 0

                for batch_sample in batch_samples:
                    angle_center = float(batch_sample[3])

                    # add 60% augmented images and 40% recorded images
                    if augment and np.random.rand() <= 0.6:
                        # add random center, left or right image
                        clr = np.random.randint(low=0, high=3)

                        if clr == 0:
                            # add center image
                            image = cv2.imread(self.path_to_image_data + '/' + batch_sample[0].lstrip())
                            image = cv2.cvtColor(image, color_scheme)
                            angle = angle_center
                        elif clr == 1:
                            # add left image
                            image = cv2.imread(self.path_to_image_data + '/' + batch_sample[1].lstrip())
                            image = cv2.cvtColor(image, color_scheme)
                            angle = angle_center + self.steering_angle_correction
                        elif clr == 2:
                            # add right image
                            image = cv2.imread(self.path_to_image_data + '/' + batch_sample[2].lstrip())
                            image = cv2.cvtColor(image, color_scheme)
                            angle = angle_center - self.steering_angle_correction

                        # apply random translation
                        # TODO: image, angle = da.DataAugmentation.random_translation(image, angle, [30, 30], probability=0.5)

                        # apply random perspective transformation
                        image, angle = da.DataAugmentation.random_perspective_transformation(image, angle, [40, 50], probability=0.5)

                        # apply shadow augmentation
                        # TODO: image = da.DataAugmentation.random_shadow(image, probability=0.5)

                        # crop and resize image
                        image = da.DataAugmentation.crop_image(image, self.roi, crop_size)

                        # apply random flip, lr_bias = 0.0 (no left/right bias correction of dataset)
                        image, angle = da.DataAugmentation.flip_image_horizontally(image, angle, probability=0.5, lr_bias=0.0)

                        # apply random brightness
                        # TODO: image = da.DataAugmentation.random_brightness(image, probability=0.5)
                    else:
                        # add center image
                        image = cv2.imread(self.path_to_image_data + '/' + batch_sample[0].lstrip())
                        image = cv2.cvtColor(image, color_scheme)
                        image = da.DataAugmentation.crop_image(image, self.roi, crop_size)
                        angle = angle_center

                    images.append(image)
                    angles.append(angle)
                    nb_batch_samples += 1
                    i += 1

                    if self.verbose == 2:
                        # show generated images in a separate window
                        image_gen = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        da.DataAugmentation.draw_steering_angles(image_gen, steering_angle=angle)
                        cv2.imshow('Generator output', image_gen)
                        cv2.waitKey(100)

                nb_total_samples += nb_batch_samples

                if self.verbose > 0:
                    print(' Generator: nb_batch_samples: {:4d} nb_total_samples: {:5d}/{:5d}'.format(nb_batch_samples, nb_total_samples, nb_samples))

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

        self.nb_train_samples = len(train_samples)
        self.nb_validation_samples = len(validation_samples)

        self.train_generator = self.generator(self.train_samples, batch_size=self.batch_size, augment=True)
        self.validation_generator = self.generator(self.validation_samples, batch_size=self.batch_size, augment=False)

    def train(self, nb_epochs, nb_samples_per_epoch, learning_rate=0.001, verbose=1):
        """
        Trains the network with given number of epochs.
        
        :param nb_epochs:            Number of epochs the model will be trained.
        :param nb_samples_per_epoch: Number of samples which will be train in each epoch.
        :param learning_rate:        Learning rate. Default = 0.001
        :param verbose:              0 = no logging, 1 = progress bar, 2 = one log line per epoch, Default = 1
        
        :return: Returns the training history ['loss', 'val_loss'].
        """

        filepath = self.model_name + '_checkpoint_{epoch:02d}_{val_loss:.4f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=True)

        optimizer = adam(lr=learning_rate)
        self.nb_samples_per_epoch = nb_samples_per_epoch
        self.model.compile(optimizer=optimizer, loss='mse')
        self.history_object = self.model.fit_generator(self.train_generator,
                                                       samples_per_epoch=nb_samples_per_epoch,
                                                       validation_data=self.validation_generator,
                                                       nb_val_samples=self.nb_validation_samples,
                                                       nb_epoch=nb_epochs,
                                                       verbose=verbose,
                                                       callbacks=[checkpoint, early_stopping])
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
