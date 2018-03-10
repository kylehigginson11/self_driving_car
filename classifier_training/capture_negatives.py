# Raspberry Pi Camera import
from picamera import PiCamera
# Python imports
from time import sleep

"""Script to constantly capture images from camera to build up a directory of random images"""

# initialise camera
camera = PiCamera()

for i in range(600, 620):
    file_name = "new_negatives/negative_" + str(i) + ".jpg"
    print("Capturing..")
    camera.capture(file_name)
    sleep(2)
