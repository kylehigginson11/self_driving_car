import sys

sys.path.append('../')
import cv2
import numpy as np
import math, time
from car_control.car import Car
import picamera
from picamera.array import PiRGBArray
import logging
from datetime import datetime

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class NeuralNetwork:

    def __init__(self):
        layer_sizes = np.int32([38400, 32, 4])
        self.ann = cv2.ml.ANN_MLP_create()
        self.ann.setLayerSizes(layer_sizes)

    def create(self):
        # load neural network from file
        logging.info("Loading MLP ...")
        self.ann = cv2.ml.ANN_MLP_load('neural_networks/neural_network.xml')
        logging.info("MLP loaded ...")

    def predict(self, samples):
        # make prediction on passed data
        ret, resp = self.ann.predict(samples)
        return resp.argmax(-1)


class CarControl:
    speed = 0.4
    turning_speed = 0.315

    def __init__(self):
        self.car = Car(9, 6)

    def steer(self, prediction, sign_decision):
        distance = self.car.get_distance()

        if distance > 15:
            # if a sign is not detcted (sign_decision will be 0)
            if sign_decision == 0:
                if prediction == 1:
                    # speed left wheel, left dir, speed right wheel, right dir
                    self.car.set_motors(self.turning_speed, 0, self.speed, 0)
                    # print("Left")
                elif prediction == 2:
                    self.car.set_motors(self.speed, 0, self.speed, 0)
                    # print("Forward")
                elif prediction == 3:
                    self.car.set_motors(self.speed, 0, self.turning_speed, 0)
                    # print("Right")
                else:
                    self.car.stop()
            # elif sign_decision == 1:
            #     # this is a left arrow sign
            #     self.car.stop()
            elif sign_decision == 2:
                self.speed = 0.3
                self.turning_speed = 0.236
                self.car.set_motors(self.speed, 0, self.speed, 0)
            elif sign_decision == 3:
                self.speed = 0.5
                self.turning_speed = 0.394
                self.car.set_motors(self.speed, 0, self.speed, 0)
            elif sign_decision == 4:
                self.speed = 0.7
                self.turning_speed = 0.551
                self.car.set_motors(self.speed, 0, self.speed, 0)
            elif sign_decision == 5:
                self.car.stop()
        else:
            self.car.stop()

    def stop_car(self):
        self.car.stop()

    def change_speed(self, speed):
        self.speed = speed


class SignDetector:
    # left_sign_path = "../classifier_training/working_classifiers/left_sign_classifier.xml"
    thirty_speed_sign_path = "../classifier_training/working_classifiers/thirty_speed_limit_classifier.xml"
    # forty_speed_sign_path = "../classifier_training/working_classifiers/forty_speed_limit_classifier.xml"
    national_speed_sign_path = "../classifier_training/working_classifiers/national_speed_limit_classifier.xml"
    red_light_path = "../classifier_training/working_classifiers/red_light_classifier.xml"

    def __init__(self):
        # loading sign classifiers
        logging.info("Loading sign classifiers")
        # self.left_sign_cascade = cv2.CascadeClassifier(self.left_sign_path)
        self.thirty_speed_sign_path = cv2.CascadeClassifier(self.thirty_speed_sign_path)
        # self.forty_speed_sign_cascade = cv2.CascadeClassifier(self.forty_speed_sign_path)
        self.national_speed_sign_cascade = cv2.CascadeClassifier(self.national_speed_sign_path)
        self.red_light_cascade = cv2.CascadeClassifier(self.red_light_path)

    def detcted_sign(self, image):
        # left_sign_rect = self.left_sign_cascade.detectMultiScale(image, 1.3, 5)
        thirty_speed_sign_rect = self.thirty_speed_sign_path.detectMultiScale(image, 1.3, 5)
        # forty_speed_sign_rect = self.forty_speed_sign_cascade.detectMultiScale(image, 1.3, 5)
        national_speed_sign_rect = self.national_speed_sign_cascade.detectMultiScale(image, 1.3, 5)
        red_light_rect = self.red_light_cascade.detectMultiScale(image, 1.3, 5)

        # if len(left_sign_rect) != 0:
        #    print("Left sign detected")
        #    return 1
        if len(thirty_speed_sign_rect) != 0:
            print("30 speed limit sign detected")
            return 2
        # elif len(forty_speed_sign_rect) != 0:
        #    print("40 speed limit sign detected")
        #    return 3
        elif len(national_speed_sign_rect) != 0:
            print("National speed limit sign detected")
            return 4
        elif len(red_light_rect) != 0:
            print("Red light detected!")
            return 5
        else:
            return 0


class StreamFrames:
    start_time = datetime.now()
    # load neural network
    model = NeuralNetwork()
    model.create()

    # initialise car and sign detector
    car_controller = CarControl()
    sign_detector = SignDetector()

    def __init__(self):

        # initialize the camera and grab a reference to the raw camera capture
        logging.info("Initialising Camera ...")
        camera = picamera.PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 32
        camera.rotation = 180
        raw_capture = PiRGBArray(camera, size=(320, 240))
        time.sleep(1)
        # stream video frames one by one
        logging.info("Camera Initialised ...")

        try:
            for frame in camera.capture_continuous(raw_capture, 'bgr', use_video_port=True):
                image = frame.array
                image.setflags(write=1)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                sign_segment = gray[0:160, 200:320]
                sign_decision = self.sign_detector.detcted_sign(sign_segment)

                # lower half of the image
                half_gray = gray[100:220, :]

                # reshape the image in matrix
                image_array = half_gray.reshape(1, 38400).astype(np.float32)

                # reset camera for next frame
                raw_capture.truncate(0)

                # neural network makes prediction
                prediction = self.model.predict(image_array)
                # print (prediction)

                self.car_controller.steer(prediction, sign_decision)
        finally:
            cv2.destroyAllWindows()
            self.car_controller.stop_car()
            end_time = datetime.now()
            logging.info("Total Journey duration: " + str(end_time - self.start_time) + 'seconds')
            logging.info("Connection closed on thread 1")


if __name__ == '__main__':
    StreamFrames()
