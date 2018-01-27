import sys
sys.path.append('../')
import cv2
import numpy as np
import math, time
from car_control.car import Car
import picamera
from picamera.array import PiRGBArray
import logging


logger = logging.getLogger('driverless_car')
handler = logging.FileHandler('/var/log/driverless_car/driverless_car.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)


class NeuralNetwork:

    def __init__(self):
        layer_sizes = np.int32([38400, 32, 4])
        self.ann = cv2.ml.ANN_MLP_create()
        self.ann.setLayerSizes(layer_sizes)

    def create(self):
        # load neural network from file
        logger.info("Loading MLP ...")
        self.ann = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')
        logger.info("MLP loaded ...")

    def predict(self, samples):
        # make prediction on passed data
        ret, resp = self.ann.predict(samples)
        return resp.argmax(-1)


class CarControl:

    def __init__(self):
        self.car = Car(9, 6)

    def steer(self, prediction):
        if prediction == 1:
            # speed left wheel, left dir, speed right wheel, right dir
            self.car.set_motors(0.3, 0, 0.4, 0)
            # print("Left")
        elif prediction == 2:
            self.car.set_motors(0.3, 0, 0.3, 0)
            # print("Forward")
        elif prediction == 3:
            self.car.set_motors(0.4, 0, 0.3, 0)
            # print("Right")
        else:
            self.stop()

    def stop(self):
        self.car.stop()


class StreamFrames:
    # load neural network
    model = NeuralNetwork()
    model.create()

    car = CarControl()

    def __init__(self):

        # initialize the camera and grab a reference to the raw camera capture
        logger.info("Initialising Camera ...")
        camera = picamera.PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 32
        raw_capture = PiRGBArray(camera, size=(320, 240))
        time.sleep(1)
        # stream video frames one by one
        logger.info("Camera Initialised ...")

        try:
            for frame in camera.capture_continuous(raw_capture, 'bgr', use_video_port=True):
                image = frame.array
                image.setflags(write=1)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # lower half of the image
                half_gray = gray[80:200, :]

                # reshape image
                image_array = half_gray.reshape(1, 38400).astype(np.float32)

                # reset camera for next frame
                raw_capture.truncate(0)

                # neural network makes prediction
                prediction = self.model.predict(image_array)
                # print (prediction)

                self.car.steer(prediction)
        finally:
            cv2.destroyAllWindows()
            self.car.stop()
            logger.info("Connection closed on thread 1")


if __name__ == '__main__':
    StreamFrames()
