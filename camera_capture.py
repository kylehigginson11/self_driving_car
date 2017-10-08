from picamera import PiCamera
from time import sleep

camera = PiCamera()

for i in range(100):
    file_name = "negative_images/negative_" + str(i) + ".jpg"
    for n in range(0, 3):
        sleep(1)
        print (n)
    print ("Capturing..")
    camera.capture(file_name)
