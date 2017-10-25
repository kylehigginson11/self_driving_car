# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from car import Car

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))


# initialise car
car = Car(9, 6)
car.forward()

def nothing(x):
    pass


# allow the camera to warmup
time.sleep(0.1)

# opencv window
#cv2.namedWindow('image')

# capture frames from the camera continuously
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image.setflags(write=1)

    # apply a blur to the frame for faster processing
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # strip the image down
    #image = image[140:240, :]

    # convert to an HSV stream
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    LOW = np.array([0, 0, 0])
    HIGH = np.array([20, 255, 255])

    # create mask
    mask = cv2.inRange(hsv, LOW, HIGH)

    thresholdimage, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = 0
    # print (contours)
    if contours:
        for contour in contours:
            this_area = cv2.contourArea(contour)
            if this_area > biggest_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                if x < 100:
                    # assume this is left lane
                    biggest_area = this_area
                    #print ("left lane: " + str(y))
                    if x > 0:
                        # turn right
                        car.set_motors(0.2, 1, 0.7. 1)
                    else if x == 0:
                        # go forward
                        car.set_motors(0.7, 1, 0.7, 1)
                else:
                    # assume right lane
                    #print ("right lane: " + str(y))
                    if x < 150:
                        # turn left
                        car.set_motors(0.7, 1, 0.2, 1)
                    else if x > 150:
                        # go forward
                        car.set_motors(0.7, 1, 0.7, 1)

    #res = cv2.bitwise_and(image, image, mask=mask)

   # update image on window
    #cv2.imshow('image', res)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
    car.stop()
        break
