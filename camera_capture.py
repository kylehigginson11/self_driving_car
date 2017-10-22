from picamera import PiCamera
from time import sleep

camera = PiCamera()

for i in range(600, 620):
    file_name = "new_negatives/negative_" + str(i) + ".jpg"
    print ("Capturing..")
    camera.capture(file_name)
    sleep(2)
