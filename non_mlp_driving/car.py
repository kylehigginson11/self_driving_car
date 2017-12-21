import RPi.GPIO as GPIO
import time
import pigpio


# part of the code in this file is Based on: https://github.com/simonmonk/raspirobotboard3/blob/master/python/rrb3.py
class Car:
    MOTOR_DELAY = 0.2

    RIGHT_PWM_PIN = 14
    RIGHT_1_PIN = 10
    RIGHT_2_PIN = 25
    LEFT_PWM_PIN = 24
    LEFT_1_PIN = 17
    LEFT_2_PIN = 4
    left_pwm = 0
    right_pwm = 0
    pwm_scale = 0

    # Ultrasonic sensor pins
    TRIGGER_PIN = 18
    ECHO_PIN = 23

    # Servo motor pin and initialisation
    SERVO_PIN = 15
    servo = pigpio.pi()

    old_left_dir = -1
    old_right_dir = -1

    DEFAULT_TAG = "albert_photo"

    def __init__(self, battery_voltage=9.0, motor_voltage=6.0, revision=2):

        self.pwm_scale = float(motor_voltage) / float(battery_voltage)

        if self.pwm_scale > 1:
            print("WARNING: Motor voltage is higher than battery votage. Motor may run slow.")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # GPIO setup for left wheels
        GPIO.setup(self.LEFT_PWM_PIN, GPIO.OUT)
        self.left_pwm = GPIO.PWM(self.LEFT_PWM_PIN, 500)
        self.left_pwm.start(0)
        GPIO.setup(self.LEFT_1_PIN, GPIO.OUT)
        GPIO.setup(self.LEFT_2_PIN, GPIO.OUT)

        # GPIO setup for right wheels
        GPIO.setup(self.RIGHT_PWM_PIN, GPIO.OUT)
        self.right_pwm = GPIO.PWM(self.RIGHT_PWM_PIN, 500)
        self.right_pwm.start(0)
        GPIO.setup(self.RIGHT_1_PIN, GPIO.OUT)
        GPIO.setup(self.RIGHT_2_PIN, GPIO.OUT)

        # Ultrasonic sensor GPIO Setup
        GPIO.setup(self.TRIGGER_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)

    def set_motors(self, left_pwm, left_dir, right_pwm, right_dir):
        if self.old_left_dir != left_dir or self.old_right_dir != right_dir:
            self.set_driver_pins(0, 0, 0, 0)  # stop motors between sudden changes of direction
            time.sleep(self.MOTOR_DELAY)
        self.set_driver_pins(left_pwm, left_dir, right_pwm, right_dir)
        self.old_left_dir = left_dir
        self.old_right_dir = right_dir

    def set_driver_pins(self, left_pwm, left_dir, right_pwm, right_dir):
        self.left_pwm.ChangeDutyCycle(left_pwm * 100 * self.pwm_scale)
        GPIO.output(self.LEFT_1_PIN, left_dir)
        GPIO.output(self.LEFT_2_PIN, not left_dir)
        self.right_pwm.ChangeDutyCycle(right_pwm * 100 * self.pwm_scale)
        GPIO.output(self.RIGHT_1_PIN, right_dir)
        GPIO.output(self.RIGHT_2_PIN, not right_dir)

    def forward(self, seconds=0, speed=1.0):
        self.set_motors(speed, 0, speed, 0)
        if seconds > 0:
            time.sleep(seconds)
            self.stop()

    def stop(self):
        self.set_motors(0, 0, 0, 0)

    def reverse(self, seconds=0, speed=1.0):
        self.set_motors(speed, 1, speed, 1)
        if seconds > 0:
            time.sleep(seconds)
            self.stop()

    def right(self, seconds=0, speed=0.5):
        self.set_motors(speed, 0, speed, 1)
        if seconds > 0:
            time.sleep(seconds)
            self.stop()

    def left(self, seconds=0, speed=0.5):
        self.set_motors(speed, 1, speed, 0)
        if seconds > 0:
            time.sleep(seconds)
            self.stop()

    # functions to get values from sensor
    def _send_trigger_pulse(self):
        GPIO.output(self.TRIGGER_PIN, True)
        time.sleep(0.0001)
        GPIO.output(self.TRIGGER_PIN, False)

    def _wait_for_echo(self, value, timeout):
        count = timeout
        while GPIO.input(self.ECHO_PIN) != value and count > 0:
            count -= 1

    def get_distance(self):
        self._send_trigger_pulse()
        self._wait_for_echo(True, 10000)
        start = time.time()
        self._wait_for_echo(False, 10000)
        finish = time.time()
        pulse_len = finish - start
        distance_cm = pulse_len / 0.000058
        return distance_cm

    # functions to change angle of servo motor
    def send_pluse(self, angle):
        self.servo.set_servo_pulsewidth(self.SERVO_PIN, angle)

    def look_center(self):
        self.servo.set_servo_pulsewidth(self.SERVO_PIN, 1800)

    def look_left(self):
        self.servo.set_servo_pulsewidth(self.SERVO_PIN, 2500)

    def look_right(self):
        self.servo.set_servo_pulsewidth(self.SERVO_PIN, 700)

    def cleanup(self):
        GPIO.cleanup()
