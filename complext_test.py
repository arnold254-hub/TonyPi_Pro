import sys
import time

sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

import hiwonder.ActionGroupControl as AGC
import hiwonder.ros_robot_controller_sdk as Board

def main():
    print("Starting Multi-Action Test Sequence...")
    board = Board.Board()

    # 1. Reset posture and center head
    print("Step 1: Stand up & center head...")
    AGC.runActionGroup('stand')
    # Servo 1 = Head Pan, Servo 2 = Head Tilt
    board.pwm_servo_set_position(0.5, [[1, 1500], [2, 1500]])
    time.sleep(2)

    # 2. Wave arm while sweeping head right
    print("Step 2: Waving while scanning right...")
    board.pwm_servo_set_position(1.0, [[1, 1200], [2, 1300]])  # Head looks right & down
    AGC.runActionGroup('wave')                                 # Perform wave action
    time.sleep(1)

    # 3. Sweep head left
    print("Step 3: Scanning left...")
    board.pwm_servo_set_position(1.0, [[1, 1800], [2, 1500]])  # Head looks left
    time.sleep(1.5)

    # 4. Perform a side-step routine while re-centering head
    print("Step 4: Side stepping left & right...")
    board.pwm_servo_set_position(0.5, [[1, 1500], [2, 1500]])  # Center head
    
    AGC.runActionGroup('left_step', times=2)
    time.sleep(0.5)
    AGC.runActionGroup('right_step', times=2)
    time.sleep(0.5)

    # 5. Bow and return to stand
    print("Step 5: Bowing & resetting...")
    AGC.runActionGroup('bow')
    time.sleep(1)
    AGC.runActionGroup('stand')

    print("Sequence completed successfully!")

if __name__ == "__main__":
    main()