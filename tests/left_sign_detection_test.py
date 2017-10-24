import cv2
import numpy as np
from time import sleep
import sys
from picamera import PiCamera
from picamera.array import PiRGBArray
from car import Car

print ("Load classifier")
left_sign_cascade = cv2.CascadeClassifier('./left_sign_classifier.xml')
 
cap = None
scf = 0.5

car = Car(9, 6)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 64
rawCapture = PiRGBArray(camera, size=(320, 240))
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    image.setflags(write=1)
 
    # print ("Resizing")
    image = cv2.resize(image, None, fx=scf, fy=scf, interpolation=cv2.INTER_AREA)
 
    # print ("Face detection")
    left_sign_rect = left_sign_cascade.detectMultiScale(image, 1.3, 5)
 
    #print "Sign Detected"
    print (left_sign_rect)
    if len(left_sign_rect) == 0:
        car.set_motors(0.4, 1, 0.5, 1)
    else:
        car.left(1.2)
    
    rawCapture.truncate(0) 
    c = cv2.waitKey(1)
    if c == 27:
        break
 
if cap != None:
    cap.release()
cv2.destroyAllWindows()
car.stop()
car.cleanup()
