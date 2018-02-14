import picamera
import time
import cv2

camera = picamera.PiCamera()
time.sleep(1)
camera.resolution = (320, 240)
camera.rotation = 180
camera.capture('sample_image.jpg')
time.sleep(0.5)
img = cv2.imread('sample_image.jpg', 0)

lower_half = img[0:160, 200:320]

cv2.imwrite('sample_image.jpg', lower_half)
