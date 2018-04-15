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

        # create labels, 4 possible directions
        self.array = np.zeros((OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE), 'float')
        for i in range(OUTPUT_LAYER_SIZE):
            self.array[i, i] = 1
        self.temp_label = np.zeros((1, OUTPUT_LAYER_SIZE), 'float')
        self.send_inst = True
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

    def save_training_data(self, train_images, train_labels):
        # save training data as a numpy file and name it by current time
        file_name = time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(TRAINING_DIR):
            os.makedirs(TRAINING_DIR)
        try:
            np.savez(TRAINING_DIR + '/' + file_name + '.npz', train=train_images, train_labels=train_labels)
        except IOError:
            logging.error("Couldn't save files!")

    def stream_frames(self):

        saved_frame = 0

        # collect images for training
        logging.info('Start controlling car ...')

        # get current amount of ticks
        image_array = np.zeros((1, IMAGE_PIXELS))
        label_array = np.zeros((1, OUTPUT_LAYER_SIZE), 'float')

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
                    # print("Forward Left")
                    saved_frame += 1
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, self.array[0]))
                    self.car.set_motors(0.3, 0, 0.4, 0)
                elif self.joy.Y():
                    saved_frame += 1
                    # print("Forward")
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, self.array[1]))
                    self.car.set_motors(0.3, 0, 0.3, 0)
                elif self.joy.B():
                    # print("Forward Right")
                    saved_frame += 1
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, self.array[2]))
                    self.car.set_motors(0.4, 0, 0.3, 0)
                elif self.joy.Back():
                    logging.info('Back button pressed, exiting')
                    self.car.stop()
                    break
                else:
                    self.car.stop()

            # save training images and labels
            training_images = image_array[1:, :]
            training_labels = label_array[1:, :]
            self.save_training_data(train_images=training_images, train_labels=training_labels)

            logging.info(training_images.shape)
            logging.info(training_labels.shape)
            logging.info('Number of Frames Saved: ' + str(saved_frame))

        finally:
            self.car.stop()
            self.joy.close()
            logging.info('Training Complete')


if __name__ == '__main__':
    CollectTrainingImages()
