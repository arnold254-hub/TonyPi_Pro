#!/usr/bin/python3
# coding=utf8
"""
Head (pan-tilt) servo test for TonyPi Pro.

The head is driven by two PWM servos on the controller board:
    servo 1 - tilt (nod up/down)
    servo 2 - pan  (shake left/right)
Pulse range is 500-2500, centre is 1500. Servo 2 has a per-robot centre
offset stored in servo_config.yaml, which is applied here as the app does.

Run on the robot with:
    sudo systemctl stop tonypi      # free the serial port used by the board
    python3 /home/pi/TonyPi/Functions/test_script.py
    sudo systemctl start tonypi
"""
import sys
import time

sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.yaml_handle as yaml_handle

TILT = 1
PAN = 2
CENTER = 1500


def load_pan_offset():
    try:
        data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        return int(data.get("servo2", CENTER)) - CENTER
    except Exception as e:
        print("Could not read servo_config.yaml (%s), using zero offset" % e)
        return 0


def main():
    print("Testing head servos...")
    board = rrc.Board()
    ctl = Controller(board)
    pan_offset = load_pan_offset()

    def move(servo_id, pulse, ms):
        if servo_id == PAN:
            pulse += pan_offset
        pulse = max(500, min(2500, pulse))
        ctl.set_pwm_servo_pulse(servo_id, pulse, ms)
        time.sleep(ms / 1000 + 0.3)

    try:
        print("Centering head")
        move(TILT, CENTER, 500)
        move(PAN, CENTER, 500)

        print("Pan left")
        move(PAN, 1800, 500)
        print("Pan right")
        move(PAN, 1200, 500)
        print("Pan centre")
        move(PAN, CENTER, 500)

        print("Tilt up")
        move(TILT, 1800, 500)
        print("Tilt down")
        move(TILT, 1200, 500)
        print("Tilt centre")
        move(TILT, CENTER, 500)
    except KeyboardInterrupt:
        print("\nInterrupted, re-centering head")
        move(TILT, CENTER, 300)
        move(PAN, CENTER, 300)
        return

    print("Servo test completed!")


if __name__ == "__main__":
    main()
