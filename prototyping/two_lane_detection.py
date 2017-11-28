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
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(320, 240))


# initialise car
car = Car(9, 6)
#car.reverse()

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
    distance = car.get_distance()

    # apply a blur to the frame for faster processing
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # strip the image down
    image = image[140:240, :]

    # convert to an HSV stream
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    LOW = np.array([0, 0, 0])
    HIGH = np.array([15, 255, 255])

    # create mask
    mask = cv2.inRange(hsv, LOW, HIGH)

    thresholdimage, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = 0
    # print (contours)
    if len(contours) > 1:
        print (len(contours))
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            print (x, y, w, h)
            if x < 150 and w > 5 and h > 5:
                # assume this is left lane
                print ("left lane: " + str(x))
                if x > 80 and x < 120:
                    print("turn right")
                    car.set_motors(0.4, 0, 0.25, 0)
                #elif x > 120:
                #    car.set_motors(0.5, 0, 0.1, 0)
                elif x < 40:
                    # go forward
                    car.set_motors(0.4, 0, 0.4, 0)
            else:
                # assume right lane
                print ("right lane: " + str(x))
                if x > 190 and x < 210:
                    # turn left
                    print ("Turn left")
                    car.set_motors(0.25, 0, 0.4, 0)
                #elif x > 180 and x < 200:
                #    car.set_motors(0.1, 0, 0.5, 0)
                elif x > 220 and x < 300:
                    # go forward
                    car.set_motors(0.4, 0, 0.4, 0)
        time.sleep(0.2)
    if distance < 20:
        print ("Object in Front!")
        car.stop()
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
