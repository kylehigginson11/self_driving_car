import cv2
import numpy as np
from time import sleep
import sys
from picamera import PiCamera
from car import Car

print "Load classifier"
sign_cascade = cv2.CascadeClassifier('./left_sign_classifier.xml')
 
cap = None
scf = 0.5
image = "image.jpg"
 
camera = PiCamera()
car = Car(9, 6)
 
while True:
    print "Capture image"
    camera.capture(image)
    
    print "Get a frame"
    frame = cv2.imread(image)
 
    print "Resizing"
    frame = cv2.resize(frame, None, fx=scf, fy=scf, interpolation=cv2.INTER_AREA)
 
    print "Face detection"
    sign_rect = sign_cascade.detectMultiScale(frame, 1.3, 5)
 
    #print "Sign Detected"
    print sign_rect
    if len(sign_rect) == 0:
        car.set_motors(0.5, 1, 0.5, 1)
    else:
        car.stop()
 
    #print "Drawing rectangle"
    
    #for (x,y,w,h) in sign_rect:
    #    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
 
    #print "Show image"
    #cv2.imshow('Face Detector', frame)
 
    c = cv2.waitKey(1)
    if c == 27:
        break
 
if cap != None:
    cap.release()
cv2.destroyAllWindows()
car.stop()
car.cleanup()
