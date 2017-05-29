import argparse
import base64
from datetime import datetime
import os
import shutil
import sys

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version
from networks.BaseNetwork import BaseNetwork
# TODO: from Filter import Filter
from model import IMAGE_WIDTH, IMAGE_HEIGHT, ROI
from DataAugmentation import DataAugmentation
# TODO: import cv2


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.003) # org: 0.1, 0.002
set_speed = 15
controller.set_desired(set_speed)

recovery = 0                              # hack to avoid vehicle standstill situations in the simulation
frame = 0
# TODO: filter = Filter()


def bar(value, range=[-1., 1.], prefix='', suffix='', limit=None):
    """ Shows graph like this [-----|-----] in the console.

    :param value:  Value which shall be shown.
    :param range:  Range of graph.
    :param prefix: Text shown before the graph.
    :param suffix: Text shown behind the graph.
    :param limit:  Draws a limit bar `|` at the pos/neg limit position.
    """

    bar_len = 21
    r = float(range[1] - range[0])
    value_pos = max(min(int(bar_len * (value + r / 2.) / r), bar_len - 1), 0)

    list = ['-'] * bar_len
    list[int(bar_len / 2)] = '+'

    if limit is not None:
        threshold_pos_0 = max(min(int(bar_len * (limit + r / 2.) / r), bar_len - 1), 0)
        threshold_pos_1 = max(min(int(bar_len * (-limit + r / 2.) / r), bar_len - 1), 0)
        list[threshold_pos_0] = '|'
        list[threshold_pos_1] = '|'

    list[value_pos] = '\033[97m∆\033[00m'

    str = ''.join(list)
    sys.stdout.write('{:s}[{:s}] {:s}'.format(prefix, str, suffix))
    sys.stdout.flush()


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        # Pre-process image and predict steering angle
        input_image = BaseNetwork.preprocess_image(image_array, IMAGE_WIDTH, IMAGE_HEIGHT, ROI)
        steering_angle = float(model.predict(input_image[None, :, :, :], batch_size=1))

        # filter steering angle by moving average
        # TODO: steering_angle = filter.moving_average(steering_angle, window_size=8)

        # emergency brake to handle downhill driving
        if speed > (controller.set_point + 7):
            controller.set_desired(3)
            info_text = '\033[91m<<<< EMERGENCY BRAKE >>>>\033[00m'
        else:
            controller.set_desired(set_speed)
            info_text = ''

        throttle = controller.update(speed)

        # hack to avoid vehicle standstill in simulation
        global recovery

        if speed <= 0.01 and recovery > 0:
            recovery = max(0, recovery - 1)
            info_text += '\033[92m<<< RECOVERY MODE >>>\033[00m'
            send_control(0, -1.)
        else:    
            send_control(steering_angle, throttle)
            recovery = 3

        # show status
        bar(steering_angle, prefix='angle: ', suffix=' {:7.2f}°'.format(steering_angle * 25.))
        bar(throttle, prefix='  throttle: ', suffix=' {:5.2f}'.format(throttle))
        print('  speed: {:5.2f} / {:5.2f} mph {:s}'.format(speed, controller.set_point, info_text))

        # save frame
        if args.image_folder != '':
            global frame

            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image = Image.fromarray(DataAugmentation.draw_overlay(image_array,
                                                    frame=frame,
                                                    steering_angle=steering_angle * 25.,
                                                    speed=float(speed),
                                                    color=(0, 30, 70)))
            frame += 1
            image.save('{}.jpg'.format(image_filename))

            # TODO: show steering angle prediction in image
            # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            # cv2.imshow('Predicted steering angle', image_array)
            # cv2.waitKey(1)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
