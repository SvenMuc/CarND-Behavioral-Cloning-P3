import cv2
import csv
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt


class DataAugmentation:
    """ Provides basic method to augment the data. """

    @staticmethod
    def show_images(original_image, augmented_image):
        """ Shows the original and augmented image.
        
        :param original_image:  Original image. 
        :param augmented_image: Augmented image.
        """

        # TODO: implement show_images method.
        return False

    @staticmethod
    def draw_steering_angles(image, steering_angle=None, augmented_steering_angle=None):
        """ Visualizes the steering angle by line in the center of the image.
        
        :param image:                     Input image.
        :param steering_angle:            Steering angle in degree (green)
        :param augmented_steering_angle:  Augmented steering angle in degree (red).
        
        :return: Returns the image with visualized steering angle.
        """

        height = image.shape[0]
        width = image.shape[1]
        p0 = (int(width / 2), height)

        # draw steering angle
        if steering_angle is not None:
            #p1 = (int(width / 2) + int(height / 2 * math.tan(math.radians(steering_angle * 25.))), int(height / 2))
            p1 = (int(width / 2) + int(height / 2 * math.tan(math.radians(steering_angle * 25.))), 0)
            cv2.line(image, p0, p1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw augmented steering angle
        if augmented_steering_angle is not None:
            #p1 = (int(width / 2) + int(height / 2 * math.tan(math.radians(augmented_steering_angle * 25.))), int(height / 2))
            p1 = (int(width / 2) + int(height / 2 * math.tan(math.radians(augmented_steering_angle * 25.))), 0)
            cv2.line(image, p0, p1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return image

    @staticmethod
    def reduce_zero_steering_angles(samples, reduction_rate):
        """ Reduces total amount of images with 0° steering angle.
        
        :param samples:        Measurement samples which shall be augmented.
        :param reduction_rate: Reduction rate [0..1].
        :return: Returns the augmented samples array. If reduction rate is out of the range [0..1] the input samples
                 will be returned.
        """

        zero_angle_index = np.empty(0, dtype='int32')

        if reduction_rate < 1.:
            for idx, sample in enumerate(samples):
                if float(sample[3]) == 0.:
                    zero_angle_index = np.append(zero_angle_index, idx)

            nb_samples_to_delete = math.ceil(len(zero_angle_index) * reduction_rate)
            idx = np.arange(nb_samples_to_delete)
            np.random.shuffle(idx)
            return np.delete(samples, zero_angle_index[idx], 0)
        else:
            return samples

    @staticmethod
    def flip_image_horizontally(image, steering_angle):
        """ Flip image horizontally
        
        :param image:          Input RGB image.
        :param steering_angle: Input steering angle.
        
        :return: Returns the augmented RGB image and the flipped steering angle.
        """
        flipped_image = cv2.flip(image, 1)
        flipped_steering_angle = -steering_angle

        return flipped_image, flipped_steering_angle

    @staticmethod
    def augment_brightness(image):
        """ Augments the image by random brightness adjustment.
        
        For HSV color space see: https://en.wikipedia.org/wiki/HSL_and_HSV
        
        :param image: Input RGB image.
        
        :return:      Returns the augmented RGB image.
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = np.array(image_hsv, dtype=np.float64)
        rand_brightness = .1 + np.random.uniform()
        image_hsv[:, :, 2] = image_hsv[:, :, 2] * rand_brightness
        image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255
        image_hsv = np.array(image_hsv, dtype=np.uint8)
        image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        return image_rgb

    @staticmethod
    def augment_blurring(image):
        """ Augments the image by random blur.
        
        :param image: Input RGB image.
        
        :return:      Returns the augmented RGB image.
        """

        return cv2.blur(image, (3, 3))
        # return cv2.medianBlur(image, 5)
        # return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def augment_shadow(image):
        """ Augments the image by random shadow areas.

        :param image: Input RGB image.
        
        :return:      Returns the augmented RGB image.
        """

        # TODO: implement shadow augmentation

        return image

    @staticmethod
    def augment_translation(image, steering_angle, max_translation):
        """ Translates (shift) the image randomly horizontally and vertically.
        
        ATTENTION: This method shall only be applied, if the car is driving in the center of the road.
        
        :param image:           Input RGB image.
        :param steering_angle:  Input steering angle.
        :param max_translation: Max translation in pixel [tx_max, ty_max].
        
        :return: Returns the augmented RGB image and the augmented steering angle.
        """

        height = image.shape[0]
        width = image.shape[1]
        tx = np.random.uniform(low=-max_translation[0], high=max_translation[0])
        ty = np.random.uniform(low=0, high=max_translation[1])
        # ty = np.random.uniform(low=-max_translation[1], high=max_translation[1])
        t_angle = steering_angle + (math.degrees(math.tan(2. * tx / height)) / 25.)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        t_image = cv2.warpAffine(image, M, (width, height))  #, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return t_image, t_angle

    @staticmethod
    def augment_rotation(image, steering_angle, max_rotation):
        """ Rotates the image randomly around mid of bottom image row.
        
        ATTENTION: This method shall only be applied, if the car is driving in the center of the road.
        
        :param image:          Input RGB image.
        :param steering_angle: Input steering angle.
        :param max_rotation:   Max rotation angle in degree.
        
        :return: Returns the augmented RGB image and the augmented steering angle.
        """

        height = image.shape[0]
        width = image.shape[1]
        rot = np.random.uniform(low=-max_rotation, high=max_rotation)
        r_angle = steering_angle  # TODO: check rotation angle: - (rot / 25.)
        M = cv2.getRotationMatrix2D((height, width / 2), rot, 1)
        t_image = cv2.warpAffine(image, M, (width, height))  #, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return t_image, r_angle

    @staticmethod
    def augment_perpective_transformation(image, steering_angle, max_trans):
        """ Applies random perspective transformation to image to simulate curves.
        
        ATTENTION: This method shall only be applied, if the car is driving in the center of the road.
        
        :param image:          Input RGB image.
        :param steering_angle: Input steering angle.
        :param max_trans:      Max transformation in pixel [x, y].
        
        :return: Returns the augmented RGB image and the augmented steering angle.
        """

        height = image.shape[0]
        width = image.shape[1]
        tx = np.random.uniform(low=-max_trans[0], high=max_trans[0])
        ty = np.random.uniform(low=-max_trans[1], high=max_trans[1])
        t_angle = steering_angle + (math.degrees(math.tan(2. * tx / height)) * 0.5 / 25.)
        points1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        points2 = np.float32([[0+tx, 0+ty], [width+tx, 0+ty], [0, height], [width, height]])
        M = cv2.getPerspectiveTransform(points1, points2)
        t_image = cv2.warpPerspective(image, M, (width, height))  #, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return t_image, t_angle

    @staticmethod
    def crop_image(image, roi, resize_size=None):
        """
        
        :param image:       Input RGB image.
        :param roi:         Cropping area (region of interest) [x0, y0, x1, y1].
        :param resize_size: If not NOne the cropped image will be resized (width, height).
        
        :return: Returns the augmented RGB image and the augmented steering angle.
        """

        cropped_image = image[roi[1]:roi[3], roi[0]:roi[2]]

        if resize_size is not None:
            return cv2.resize(cropped_image, resize_size, interpolation=cv2.INTER_AREA)
        else:
            return cropped_image


def prepare_dataset(csv_filename):
    """ Prepares the training and validation datasets (images and measurements) from driving log cvs file.
    
    :param csv_filename:              Path and filename of CVS file.

    :return: Returns the train_samples and validation_samples dataset.
    """

    # open and read content of csv file
    samples = []
    with open(csv_filename) as csv_file:
        reader = csv.reader(csv_file)

        # skip the csv header
        next(reader, None)

        for line in reader:
            samples.append(line)

    return samples


if __name__ == "__main__":

    print('Image Augmentation:')
    samples = prepare_dataset('data/driving_log_track1_forwards.csv')

    #
    # Show augmented images
    #
    nb_shown_images = 5
    nb_samples = len(samples)

    fig1, axarr1 = plt.subplots(nb_shown_images, 7, figsize=(16, 9))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05, wspace=0.03, hspace=0.03)

    fig2, axarr2 = plt.subplots(nb_shown_images, 7, figsize=(16, 9))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05, wspace=0.03, hspace=0.03)


    for i in range(nb_shown_images):
        idx = randint(0, nb_samples)
        dataset_path = './data'

        # load random center image
        image = cv2.imread(dataset_path + '/' + samples[idx][0].lstrip())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        steering_angle = float(samples[idx][3])

        # augment images
        brightness_image = DataAugmentation.augment_brightness(image)
        blurred_image = DataAugmentation.augment_blurring(image)
        flipped_image, flipped_steering_angle = DataAugmentation.flip_image_horizontally(image, steering_angle)
        translated_image, translated_steering_angle = DataAugmentation.augment_translation(image, steering_angle, [20, 20])
        rotated_image, rotated_steering_angle = DataAugmentation.augment_rotation(image, steering_angle, 10.)
        transformed_image, transformed_steering_angle = DataAugmentation.augment_perpective_transformation(image, steering_angle, [40, 50])

        # crop images
        width = image.shape[1]
        height = image.shape[0]
        roi = [20, 60, width - 20, height - 22]
        size = (64, 64)

        cropped_image = DataAugmentation.crop_image(image, roi, size)
        cropped_brightness_image = DataAugmentation.crop_image(brightness_image, roi, size)
        cropped_blurred_image = DataAugmentation.crop_image(blurred_image, roi, size)
        cropped_flipped_image = DataAugmentation.crop_image(flipped_image, roi, size)
        cropped_translated_image = DataAugmentation.crop_image(translated_image, roi, size)
        cropped_rotated_image = DataAugmentation.crop_image(rotated_image, roi, size)
        cropped_transformed_image = DataAugmentation.crop_image(transformed_image, roi, size)

        # draw steering angle direction into image
        DataAugmentation.draw_steering_angles(image, steering_angle=steering_angle)
        DataAugmentation.draw_steering_angles(flipped_image, augmented_steering_angle=flipped_steering_angle)
        DataAugmentation.draw_steering_angles(translated_image, steering_angle=steering_angle, augmented_steering_angle=translated_steering_angle)
        DataAugmentation.draw_steering_angles(rotated_image, steering_angle=steering_angle, augmented_steering_angle=rotated_steering_angle)
        DataAugmentation.draw_steering_angles(transformed_image, steering_angle=steering_angle, augmented_steering_angle=transformed_steering_angle)

        DataAugmentation.draw_steering_angles(cropped_image, steering_angle=steering_angle)
        DataAugmentation.draw_steering_angles(cropped_flipped_image, augmented_steering_angle=flipped_steering_angle)
        DataAugmentation.draw_steering_angles(cropped_translated_image, steering_angle=steering_angle, augmented_steering_angle=translated_steering_angle)
        DataAugmentation.draw_steering_angles(cropped_rotated_image, steering_angle=steering_angle, augmented_steering_angle=rotated_steering_angle)
        DataAugmentation.draw_steering_angles(cropped_transformed_image, steering_angle=steering_angle, augmented_steering_angle=transformed_steering_angle)

        # show augmented images
        axarr1[i, 0].imshow(image)
        axarr1[i, 0].axis('off')
        axarr1[i, 1].imshow(brightness_image)
        axarr1[i, 1].axis('off')
        axarr1[i, 2].imshow(blurred_image)
        axarr1[i, 2].axis('off')
        axarr1[i, 3].imshow(flipped_image)
        axarr1[i, 3].axis('off')
        axarr1[i, 4].imshow(translated_image)
        axarr1[i, 4].axis('off')
        axarr1[i, 5].imshow(rotated_image)
        axarr1[i, 5].axis('off')
        axarr1[i, 6].imshow(transformed_image)
        axarr1[i, 6].axis('off')

        # show cropped images
        axarr2[i, 0].imshow(cropped_image)
        axarr2[i, 0].axis('off')
        axarr2[i, 1].imshow(cropped_brightness_image)
        axarr2[i, 1].axis('off')
        axarr2[i, 2].imshow(cropped_blurred_image)
        axarr2[i, 2].axis('off')
        axarr2[i, 3].imshow(cropped_flipped_image)
        axarr2[i, 3].axis('off')
        axarr2[i, 4].imshow(cropped_translated_image)
        axarr2[i, 4].axis('off')
        axarr2[i, 5].imshow(cropped_rotated_image)
        axarr2[i, 5].axis('off')
        axarr2[i, 6].imshow(cropped_transformed_image)
        axarr2[i, 6].axis('off')

    # set titles augmented images
    axarr1[0, 0].set_title('original')
    axarr1[0, 1].set_title('brightness')
    axarr1[0, 2].set_title('blurred')
    axarr1[0, 3].set_title('flipped')
    axarr1[0, 4].set_title('translation')
    axarr1[0, 5].set_title('rotation')
    axarr1[0, 6].set_title('transformation')

    # set titles cropped images
    axarr2[0, 0].set_title('cropped original')
    axarr2[0, 1].set_title('cropped brightness')
    axarr2[0, 2].set_title('cropped blurred')
    axarr2[0, 3].set_title('cropped flipped')
    axarr2[0, 4].set_title('cropped translation')
    axarr2[0, 5].set_title('cropped rotation')
    axarr2[0, 6].set_title('cropped transformation')

    title = 'Image Augmentation (w={:d}, h={:d})'.format(image.shape[1], image.shape[0])
    fig1.suptitle(title)

    title = 'Cropped augmented Images (w={:d}, h={:d})'.format(cropped_image.shape[1], cropped_image.shape[0])
    fig2.suptitle(title)
    plt.show()
