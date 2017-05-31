import matplotlib as mpl
mpl.use('macosx', force=True)
from pylab import plt
import numpy as np


class TelemetryPlots:
    """ Dynamic telemetry plots for steering angle, speed and set-speed. """

    window_size = 30
    angle_array = np.array([])
    speed_array = np.array([])
    set_speed_array = np.array([])
    fig = None
    ax1 = None
    ax2 = None
    line_angle = None
    line_speed = None
    line_set_speed = None

    def __init__(self, steering_angle, throttle, speed, set_speed):
        """ Initializes the plots for telemetry data.

        :param steering_angle:  Actual steering angle [degree].
        :param throttle:        Actual throttle position.
        :param speed:           Actual speed [mph].
        :param set_speed:       Actual set speed [mph].
        """

        # initialize arrays
        self.angle_array = np.array([steering_angle])
        self.speed_array = np.array([speed])
        self.set_speed_array = np.array([set_speed])

        # prepare plots
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)      # steering angles
        self.ax2 = self.fig.add_subplot(212)      # speed and set-speed
        self.line_angle, = self.ax1.plot([0], self.angle_array, 'r')
        self.line_speed, = self.ax2.plot([0], self.speed_array, 'b')
        self.line_set_speed, = self.ax2.plot([0], self.set_speed_array, 'r--')

        self.ax1.set_ylim(-25., 25.)
        self.ax2.set_ylim(0., 30.)
        self.ax1.set_ylabel('steering angle [°]')
        self.ax2.set_ylabel('speed [mph]')

        plt.draw()
        plt.pause(1e-9)

    def update(self, steering_angle, throttle, speed, set_speed):
        """ Updates the telemetry plots.

        :param steering_angle:  Actual steering angle [degree].
        :param throttle:        Actual throttle position.
        :param speed:           Actual speed [mph].
        :param set_speed:       Actual set speed [mph].
        """

        self.angle_array = np.append(self.angle_array, steering_angle)
        self.speed_array = np.append(self.speed_array, speed)
        self.set_speed_array = np.append(self.set_speed_array, set_speed)
        nb_entries = len(self.angle_array)

        if nb_entries < self.window_size:
            self.line_angle.set_data(range(0, nb_entries), self.angle_array[0:nb_entries])
            self.line_speed.set_data(range(0, nb_entries), self.speed_array[0:nb_entries])
            self.line_set_speed.set_data(range(0, nb_entries), self.set_speed_array[0:nb_entries])
        else:
            self.line_angle.set_data(range(nb_entries - self.window_size, nb_entries),
                                     self.angle_array[nb_entries - self.window_size:nb_entries])
            self.ax1.set_xlim(nb_entries - self.window_size, nb_entries)

            self.line_speed.set_data(range(nb_entries - self.window_size, nb_entries),
                                     self.speed_array[nb_entries - self.window_size:nb_entries])
            self.line_set_speed.set_data(range(nb_entries - self.window_size, nb_entries),
                                         self.set_speed_array[nb_entries - self.window_size:nb_entries])
            self.ax2.set_xlim(nb_entries - self.window_size, nb_entries)

        plt.draw_all()      # redraw the canvas
        plt.pause(1e-9)