import sys
sys.path.append('../')
from car_control.car import Car


car = Car(9, 6)
distance = car.get_distance()
car.cleanup()
sys.stdout.write(str(distance))
sys.stdout.flush()
