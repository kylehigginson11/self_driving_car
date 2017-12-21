import cv2
import numpy as np
from time import sleep
import sys
from picamera import PiCamera
from picamera.array import PiRGBArray
from car import Car

print ("Loading classifier")
sign_cascade = cv2.CascadeClassifier('./left_sign_classifier.xml')

car = Car(9, 6)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 64
rawCapture = PiRGBArray(camera, size=(320, 240))

try:
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        image.setflags(write=1)

        obj_distance = car.get_distance()

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        left_sign_rect = sign_cascade.detectMultiScale(image, 1.3, 5)

        print (left_sign_rect)
        if len(left_sign_rect) == 0:
            if obj_distance is not None and obj_distance < 15:
                print ("Object in front")
                car.stop()
            else:
                car.set_motors(0.4, 0, 0.4, 0)
        else:
            print ("Sign detection")
            car.stop()

        rawCapture.truncate(0)
        c = cv2.waitKey(1)
        if c == 27:
            break

finally:
    cv2.destroyAllWindows()
    car.stop()
    car.cleanup()
