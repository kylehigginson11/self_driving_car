# Python imports
import logging
import sys
sys.path.append('../')
from time import sleep
# xbox module import
import xbox
# local imports
from car_control.car import Car

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class ControlCar:
    speed = 0.3
    turning_speed = 0.236

    def __init__(self):
        self.joy = xbox.Joystick()
        self.car = Car(9, 6)

        logging.info("Car started in manual mode")
        self.control()

    # increase speed function
    def shift_up(self):
        if self.speed < 0.9:
            self.speed += 0.2
            self.turning_speed = self.speed * 0.7875

    # decrease speed function
    def shift_down(self):
        if self.speed > 0.3:
            self.speed -= 0.2
            self.turning_speed = self.speed * 0.7875

    def control(self):
        # Loop until back button is pressed
        while not self.joy.Back():
            # A, B, X, Y buttons
            if self.joy.A():
                self.car.set_motors(self.speed, 1, self.speed, 1)
            elif self.joy.B():
                self.car.set_motors(self.speed, 0, self.turning_speed, 0)
            elif self.joy.X():
                self.car.set_motors(self.turning_speed, 0, self.speed, 0)
            elif self.joy.Y():
                self.car.set_motors(self.speed, 0, self.speed, 0)
            # DPAD, up, down, left or right
            elif self.joy.dpadUp():
                self.shift_up()
                sleep(0.3)
            elif self.joy.dpadDown():
                self.shift_down()
                sleep(0.3)
            else:
                self.car.stop()
        # Close out when done
        self.joy.close()


if __name__ == "__main__":
    ControlCar()
