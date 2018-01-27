import sys
sys.path.append('../')
from car_control.car import Car


car = Car(9, 6)
distance = car.get_distance()
sys.stdout.write(distance)
sys.stdout.flush()
