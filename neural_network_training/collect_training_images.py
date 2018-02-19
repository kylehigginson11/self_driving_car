import sys
sys.path.append('../')
import numpy as np
import cv2
import time
import os
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
from car_control.car import Car
from xbox_control import xbox
import logging
from datetime import datetime

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class CollectTrainingImages:

    car = Car(9, 6)
    joy = xbox.Joystick()
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    rawCapture = PiRGBArray(camera, size=(320, 240))

    def __init__(self):

        # create labels, 4 possible directions
        self.array = np.zeros((3, 3), 'float')
        for i in range(3):
            self.array[i, i] = 1
        self.temp_label = np.zeros((1, 3), 'float')
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
        self.rawCapture = PiRGBArray(self.camera, size=(320, 240))

    def save_training_data(self, train_images, train_labels):
        # save training data as a numpy file and name it by current time
        file_name = str(int(time.time()))
        directory = "training_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            np.savez(directory + '/' + file_name + '.npz', train=train_images, train_labels=train_labels)
        except IOError as e:
            print(e)

    def stream_frames(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        logging.info('Start controlling car ...')

        # get current amount of ticks
        time_start = datetime.now()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 3), 'float')

        # stream video frames one by one
        try:
            frame_number = 1
            for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):

                gray_image = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)

                # select lower half of the image
                lower_half = gray_image[100:220, :]

                # save streamed images
                cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame_number), lower_half)

                # reshape the image in matrix
                frame_array = lower_half.reshape(1, 38400).astype(np.float32)

                # increment frame number and total frames
                frame_number += 1
                total_frame += 1
                # reset raspberry pi camera
                self.rawCapture.truncate(0)

                if self.joy.X():
                    # print("Forward Left")
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, self.array[0]))
                    saved_frame += 1
                    self.car.set_motors(0.315, 0, 0.4, 0)
                elif self.joy.Y():
                    # print("Forward")
                    saved_frame += 1
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, self.array[1]))
                    self.car.set_motors(0.3, 0, 0.3, 0)
                elif self.joy.B():
                    # print("Forward Right")
                    image_array = np.vstack((image_array, frame_array))
                    label_array = np.vstack((label_array, self.array[2]))
                    saved_frame += 1
                    self.car.set_motors(0.4, 0, 0.315, 0)
                elif self.joy.dpadDown():
                    logging.info('Keypad down pressed, exiting')
                    self.car.stop()
                    break
                else:
                    self.car.stop()

            # save training images and labels
            training_images = image_array[1:, :]
            training_labels = label_array[1:, :]
            self.save_training_data(train_images=training_images, train_labels=training_labels)

            time_end = datetime.now()
            # calculate collection duration
            total_time = (time_end - time_start).seconds
            logging.info('Collection Time: ' + str(total_time) + 'seconds')

            logging.info(training_images.shape)
            logging.info(training_labels.shape)
            logging.info('Frames: ' + str(total_frame))
            logging.info('Saved Frames: ' + str(saved_frame))

        finally:
            self.car.stop()
            self.joy.close()
            logging.info('Training Complete')


if __name__ == '__main__':
    CollectTrainingImages()
