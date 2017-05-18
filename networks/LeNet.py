from networks.BaseNetwork import BaseNetwork
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten


class LeNet(BaseNetwork):
    """ Standard LeNet-5 network which can be used either for a classification or regression problem."""

    def __init__(self, input_width, input_height, input_depth, nb_classes, regression=False,
                 roi=None, steering_angle_correction=0.0, angle_threshold=0.0, weights_path=None):
        """ Constructs the LeNet-5 network architecture.
        
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

        super(LeNet, self).__init__('LeNet5', input_width, input_height, input_depth, nb_classes, regression,
                                    roi, steering_angle_correction, angle_threshold, weights_path)

        print('LeNet Configuration:')
        print(' Input Layer: w={:d}, h={:d}, d={:d}'.format(self.input_width, self.input_height, self.input_depth))
        print(' Output Layer: {:d}, {:s}'.format(self.nb_classes, 'regression' if self.regression else 'softmax'))

        # setup-up network architecture
        self.model = self.setup_network_architecture

    @property
    def setup_network_architecture(self):
        """ Constructs the LeNet-5 network architecture.
         
        :return: Returns the initialized network model. 
        """

        # initialize the model
        self.model = Sequential()

        # normalize and mean center images
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.input_height, self.input_width, self.input_depth)))

        # 1. layer: CONV --> POOL --> RELU
        # 6 convolutions with 5x5 filter
        # max pooling with 2x2 pool
        self.model.add(Convolution2D(6, 5, 5, border_mode='valid'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))

        # 2. layer: CONV --> POOL --> RELU
        # 16 convolutions with 5x5 filter
        # max pooling with 2x2 pool
        self.model.add(Convolution2D(16, 5, 5, border_mode="valid"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation("relu"))

        # 1. fully connected layer
        self.model.add(Flatten())
        self.model.add(Dense(120))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        # 2. fully connected layer
        self.model.add(Dense(84))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.4))

        # output layer
        self.model.add(Dense(self.nb_classes))

        if not self.regression:
            # add softmax activation in case of classification setup
            self.model.add(Activation("softmax"))

        # if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if self.weights_path is not None:
            self.model.load_weights(self.weights_path)

        # return the constructed network architecture
        return self.model
