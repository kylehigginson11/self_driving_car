# Python imports
import sys
sys.path.append('../')
import numpy as np
import cv2
import os
import time
import logging
# Raspberry Pi imports
from picamera import PiCamera
from picamera.array import PiRGBArray
# local imports
from car_control.car import Car
from xbox_control import xbox

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car_data_collection.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")

OUTPUT_LAYER_SIZE = 3
IMAGE_PIXELS = 38400  # 320 * (240/2)
TRAINING_DIR = "training_data"


class CollectTrainingImages:

    car = Car(9, 6)
    joy = xbox.Joystick()
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    rawCapture = PiRGBArray(camera, size=(320, 240))

    def __init__(self):
        # initiate camera
        self.setup_camera()
        # call method to start image collection
        self.stream_frames()

    def setup_camera(self):
        # initialize the camera and grab a reference to the raw camera capture
        self.camera.resolution = (320, 240)
        self.camera.framerate = 32
        self.camera.rotation = 180
        time.sleep(1)

    def save_training_data_to_file(self, train_images, train_labels):
        # save training data as a numpy file and name it by current time
        file_name = time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(TRAINING_DIR):
            os.makedirs(TRAINING_DIR)
        try:
            np.savez(TRAINING_DIR + '/' + file_name + '.npz', train=train_images, train_labels=train_labels)
        except IOError:
            logging.error("Couldn't save files!")

    def stream_frames(self):
        # collect images for training
        logging.info('Start controlling car ...')

        # get current amount of ticks
        image_array = np.zeros((1, IMAGE_PIXELS))
        label_array = np.zeros((1, OUTPUT_LAYER_SIZE), 'float')
        # create labels, 3 possible directions
        array = np.zeros((OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE), 'float')
        array[0, 0] = 1
        array[1, 1] = 1
        array[2, 2] = 1

        # stream video frames one by one
        try:
            for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):

                gray_image = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)

                # select lower half of the image
                lower_half = gray_image[100:220, :]

                # reshape the image in matrix
                frame_array = lower_half.reshape(1, IMAGE_PIXELS).astype(np.float32)

                # reset raspberry pi camera
                self.rawCapture.truncate(0)

                if self.joy.X():
                    # Left
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, array[0]))
                    self.car.set_motors(0.3, 0, 0.4, 0)
                elif self.joy.Y():
                    # Forward
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, array[1]))
                    self.car.set_motors(0.3, 0, 0.3, 0)
                elif self.joy.B():
                    # Right
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, array[2]))
                    self.car.set_motors(0.4, 0, 0.3, 0)
                elif self.joy.Back():
                    self.car.stop()
                    break
                else:
                    self.car.stop()

            # save training images and labels
            self.save_training_data_to_file(train_images=image_array[1:, :], train_labels=label_array[1:, :])
        except:
            logging.error("Error collecting images, check camera and try again")

        finally:
            self.car.stop()
            self.joy.close()
            logging.info('Training Complete')


if __name__ == '__main__':
    CollectTrainingImages()
