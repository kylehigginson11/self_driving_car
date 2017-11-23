from car import Car
import time

car = Car(9, 6)

car.set_motors(0, 0, 0.6, 0)
time.sleep(0.3)
car.stop()

