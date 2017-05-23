import numpy as np


class Filter:
    """ Real-time signal filter. """

    moving_average_array = np.array([])      # Array containing the moving average

    def __init__(self):
        """ Initializer. """

        self.moving_average_array = np.array([])

    def moving_average(self, value, window_size=5):
        """ Calculates the moving average of the input value.
    
       :param value:       Input value. 
       :param window_size: Window size of moving average.
        
       :return: Returns the moving average signal.
       """

        # initialize static attribute
        if len(self.moving_average_array) == 0:
            self.moving_average_array = np.full(window_size, value)

        self.moving_average_array = np.append(self.moving_average_array, value)
        window = np.ones(window_size) / float(window_size)

        # remove old data if array size greater window size
        if len(self.moving_average_array) > window_size:
            self.moving_average_array = np.delete(self.moving_average_array, 0)

        filtered_values = np.convolve(self.moving_average_array, window, mode='valid')

        return filtered_values[-1]

    def reset_moving_average(self):
        """ Resets the moving average filter. """

        self.moving_average_array = np.array([])
