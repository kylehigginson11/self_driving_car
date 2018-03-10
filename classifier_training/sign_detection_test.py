"""Script to load casacde classifier to that will print weather object is detected or not"""

# Python imports
import cv2
import sys
sys.path.append('../')
# Raspberry Pi imports
from picamera import PiCamera
from picamera.array import PiRGBArray
# local imports
from car_control.car import Car

print("Loading classifier")
sign_cascade = cv2.CascadeClassifier('working_classifiers/traffic_light_stop_classifier.xml')

car = Car(9, 6)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
camera.rotation = 180
rawCapture = PiRGBArray(camera, size=(320, 240))

try:
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        image.setflags(write=1)

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        left_sign_rect = sign_cascade.detectMultiScale(image, 1.3, 5)

        print(left_sign_rect)
        rawCapture.truncate(0)
        c = cv2.waitKey(1)
        if c == 27:
            break

finally:
    cv2.destroyAllWindows()
    car.stop()
    car.cleanup()
