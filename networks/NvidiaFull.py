from networks.BaseNetwork import BaseNetwork
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout


class NvidiaFull(BaseNetwork):
    """ Full NVIDIA CNN network which can be used either for a classification or regression problem."""

    def __init__(self, input_width, input_height, input_depth, nb_classes, regression=False,
                 roi=None, steering_angle_correction=0.0, angle_threshold=0.0, weights_path=None):
        """ Constructs the NVIDIA CNN network architecture.
        
        :param input_width:   Width of the input image.
        :param input_height:  Height if the input image.
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

        super(NvidiaFull, self).__init__('NvidiaFull', input_width, input_height, input_depth, nb_classes, regression,
                                         roi, steering_angle_correction, angle_threshold, weights_path)

        print('NVIDIA Full CNN Configuration:')
        print(' Input Layer: w={:d}, h={:d}, d={:d}'.format(self.input_width, self.input_height, self.input_depth))
        print(' Output Layer: {:d}, {:s}'.format(self.nb_classes, 'regression' if self.regression else 'softmax'))

        # setup-up network architecture
        self.model = self.setup_network_architecture

    @property
    def setup_network_architecture(self):
        """ Constructs the NVIDIA CNN network architecture.
         
        :return: Returns the initialized network model. 
        """

        # initialize the model
        self.model = Sequential()

        # normalize and mean center images
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.input_height, self.input_width, self.input_depth)))

        self.model.add(Convolution2D(3, 5, 5,  border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Convolution2D(48, 3, 3, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1164, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='relu'))

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
