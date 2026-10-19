import sys
import time

sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

import hiwonder.ros_robot_controller_sdk as Board

def main():
    print("Testing Head Servos (PWM Servo ID 1)...")
    board = Board.Board()

    # Function signature: pwm_servo_set_position(duration, [[servo_id, position]])
    # Duration is in seconds (e.g., 0.5) or milliseconds depending on SDK version
    
    # Pan left side
    board.pwm_servo_set_position(0.5, [[1, 1800]])
    time.sleep(1)

    # Pan right side
    board.pwm_servo_set_position(0.5, [[1, 1200]])
    time.sleep(1)

    # Center position
    board.pwm_servo_set_position(0.5, [[1, 1500]])
    time.sleep(1)

    print("Servo test successful!")

if __name__ == "__main__":
    main()