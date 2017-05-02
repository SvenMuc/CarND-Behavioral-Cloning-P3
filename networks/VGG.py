from networks.BaseNetwork import BaseNetwork
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout


class VGG(BaseNetwork):
    """ VGG-16 (16 layers) network which can be used either for a classification or regression problem."""

    def __init__(self, input_width, input_height, input_depth, nb_classes, regression=False,
                 crop_top=0, crop_bottom=0, steering_angle_correction=0.0, weights_path=None):
        """ Constructs the VGG-16 network architecture.
        
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

        super(VGG, self).__init__(input_width, input_height, input_depth, nb_classes, regression,
                                  crop_top, crop_bottom, steering_angle_correction, weights_path)

        print('VGG-16 Configuration:')
        print(' Input Layer: w={:d}, h={:d}, d={:d}'.format(self.input_width, self.input_height, self.input_depth))
        print(' Output Layer: {:d}, {:s}'.format(self.nb_classes, 'regression' if self.regression else 'softmax'))

        # setup-up network architecture
        self.model = self.setup_network_architecture

    @property
    def setup_network_architecture(self):
        """ Constructs the VGG-16 network architecture.
         
        :return: Returns the initialized network model. 
        """

        # initialize the model
        self.model = Sequential()

        # normalize and mean center images
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.input_height, self.input_width, self.input_depth)))

        # crop images at top and bottom
        if self.crop_top > 0 or self.crop_bottom > 0:
            self.model.add(Cropping2D(cropping=((self.crop_top, self.crop_bottom), (0, 0))))

        # CONV --> RELU --> CONV --> RELU --> POOL
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # CONV --> RELU --> CONV --> RELU --> POOL
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # CONV --> RELU --> CONV --> RELU --> CONV --> RELU --> POOL
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # CONV --> RELU --> CONV --> RELU --> CONV --> RELU --> POOL
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # CONV --> RELU --> CONV --> RELU --> CONV --> RELU --> POOL
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # FC --> DROPOUT --> FC --> DROPOUT
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))

        # output layer
        if self.regression:
            self.model.add(Dense(self.nb_classes))
        else:
            # add softmax activation in case of classification setup
            self.model.add(Dense(self.nb_classes, activation='softmax'))

        # if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if self.weights_path is not None:
            self.model.load_weights(self.weights_path)

        # return the constructed network architecture
        return self.model