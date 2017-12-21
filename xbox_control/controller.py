import xbox
from car_control.car import Car


# Format floating point number to string format -x.xxx
def convert_float(n):
    return '{:6.3f}'.format(n)
    
joy = xbox.Joystick()
car = Car(9, 6)

print ("Controller calibration, press back to quit ...")
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
    print ("Buttons "),
    if joy.A():
        print ("A"),
    else:
        print (" "),
    if joy.B():
        print ("B"),
    else:
        print (" "),
    if joy.X():
        print ("X"),
    else:
        print (" "),
    if joy.Y():
        print ("Y"),
        car.stop()
    else:
        print (" "),
    # DPAD, up, down, left or right
    print ("Dpad "),
    if joy.dpadUp():
        car.forward()
        print ("U"),
    elif joy.dpadDown():
        car.reverse()
        print ("D"),
    elif joy.dpadLeft():
        car.left()
        print ("L"),
    elif joy.dpadRight():
        print ("R"),
        car.right()
    else:
        print (" "),
        car.stop()
        
    # Move cursor back to start of line
    print chr(13),
# Close out when done
joy.close()
