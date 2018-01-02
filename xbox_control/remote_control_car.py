import xbox
import sys
sys.path.append('../')
from car_control.car import Car

# Format floating point number to string format -x.xxx
def convert_float(n):
    return '{:6.3f}'.format(n)


joy = xbox.Joystick()
car = Car(9, 6)

print ("Start controlling car remotely, press back to quit ...")
# Loop until back button is pressed
while not joy.Back():
    # Connection status
    if joy.connected():
        print ("Connected   "),
    else:
        print ("Disconnected"),
    # Left analog stick
    print ("Left Stick X: {}, Left Stick Y: {}".format(convert_float(joy.leftX()), convert_float(joy.leftY()))),
    # Right trigger
    print ("Right Trigger: ", convert_float(joy.rightTrigger())),
    # A, B, X, Y buttons
    if joy.A():
        car.reverse()
    elif joy.B():
        car.set_motors(0.4, 0, 0.332, 0)
    elif joy.X():
        car.set_motors(0.332, 0, 0.4, 0)
    elif joy.Y():
        car.set_motors(0.4, 0, 0.4, 0)
    # DPAD, up, down, left or right
    elif joy.dpadUp():
        car.forward()
    elif joy.dpadDown():
        car.reverse()
    elif joy.dpadLeft():
        car.left()
    elif joy.dpadRight():
        car.right()
    else:
        car.stop()

    # Move cursor back to start of line
    print chr(13),
# Close out when done
joy.close()
