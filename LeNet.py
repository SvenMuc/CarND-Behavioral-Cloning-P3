from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Lambda, Cropping2D


class LeNet:
    """ Standard LeNet-5 network which can be used either for a classification or regression problem."""

    width = 0               # Input width
    height = 0              # Input height
    depth = 0               # Input depth (e.g. number of channels of an image)
    nb_classes = 1          # Number of output classes resp. number of regression values
    regression = False      # If true the network is setup for a regression problem. Otherwise for classification.
    crop_top = 0            # Number of pixels the image is cropped from top row.
    crop_bottom = 0         # Number of pixels the image is cropped from bottom row.
    weights_path = ''       # Path to trained model weights.
    model = None            # LeNet keras model

    def __init__(self, width, height, depth, nb_classes, regression=False,
                 crop_top=0, crop_bottom=0, weights_path=None):
        """ Constructs the LeNet-5 network architecture.

        width        -- Width of the input image.
        height       -- Height if the input image.
        depth        -- Depth of the input image (e.g. number of channels).
        nb_classes   -- Number of unique classes (class labels) in the dataset. In case of a regression set, the number
                        of regression outputs.
        regression   -- If true the output layer is configured for a regression problem. If false the output
                        is configured with a softmax function.
        crop_top     -- If >0 the image will be cropped from top row by given number of pixels.
        crop_bottom  -- If >0 the image will be cropped from bottom by given number of pixels.
        weights_path -- Path to trained model parameters. If set, the model will be initialized by these parameters.
        """

        self.width = width
        self.height = height
        self.depth = depth
        self.nb_classes = nb_classes
        self.regression = regression
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.weights_path = weights_path

        print('LeNet Configuration:')
        print(' Input Layer: w={:d}, h={:d}, d={:d}'.format(self.width, self.height, self.depth))
        print(' Output Layer: {:d}, {:s}'.format(self.nb_classes, 'regression' if self.regression else 'softmax'))

        # setup-up network architecture
        self.model = self.setup_network_architecture

    @property
    def setup_network_architecture(self):
        """ Constructs the LeNet-5 network architecture. """

        # initialize the model
        self.model = Sequential()

        # normalize and mean center images
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.height, self.width, self.depth)))

        # crop images at top and bottom
        if self.crop_top > 0 or self.crop_bottom > 0:
            self.model.add(Cropping2D(cropping=((self.crop_top, self.crop_bottom), (0, 0))))

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
