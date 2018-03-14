# Python imports
import sys
sys.path.append('../')
# local imports
from car_control.car import Car


car = Car(9, 6)
distance = car.get_distance()
# Take second reading, first is not always accurate
distance = car.get_distance()
car.cleanup()
sys.stdout.write(str(distance))
sys.stdout.flush()
