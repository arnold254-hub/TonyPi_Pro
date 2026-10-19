#!/usr/bin/python3
# coding=utf8
"""
Full-body movement test for TonyPi Pro.

Runs a sequence of preset action groups from /home/pi/TonyPi/ActionGroups.
Every action name below is checked against the .d6a files on disk before
the sequence starts, so a typo fails fast instead of mid-routine.

Run on the robot with:
    sudo systemctl stop tonypi      # free the serial port used by the board
    python3 /home/pi/TonyPi/test.py
    sudo systemctl start tonypi     # restore the app service afterwards
"""
import os
import sys
import time

sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

import hiwonder.ActionGroupControl as AGC

ACTION_DIR = "/home/pi/TonyPi/ActionGroups/"

# (action group name, repeat count, pause after in seconds)
SEQUENCE = [
    ("stand",          1, 2.0),   # reset posture
    ("left_uppercut",  5, 1.0),
    ("right_uppercut", 4, 1.0),
    ("stepping",       5, 1.0),   # march in place
    ("back_fast",      5, 1.0),   # walk backward
    ("go_forward",     3, 1.0),   # walk forward
    ("stand",          1, 1.0),
    ("sit_ups",        6, 1.0),
    ("squat",          3, 1.0),   # stands in for push_ups (no .d6a on this robot)
    ("chest",          1, 1.0),   # stands in for weightlifting (no .d6a on this robot)
    ("stand",          1, 2.0),
]

# Walking gaits need a start/end step so the robot finishes balanced.
WALK_ACTIONS = {"go_forward", "go_forward_fast", "go_forward_slow", "back", "back_fast"}


def check_actions():
    missing = [name for name, _, _ in SEQUENCE
               if not os.path.isfile(os.path.join(ACTION_DIR, name + ".d6a"))]
    if missing:
        print("Missing action groups in %s: %s" % (ACTION_DIR, ", ".join(missing)))
        sys.exit(1)


def run(name, times, pause):
    print("-> %s x%d" % (name, times))
    AGC.runActionGroup(name, times=times, with_stand=(name in WALK_ACTIONS), path=ACTION_DIR)
    time.sleep(pause)


def main():
    check_actions()
    print("Testing robot movement...")
    try:
        for name, times, pause in SEQUENCE:
            run(name, times, pause)
    except KeyboardInterrupt:
        print("\nInterrupted, stopping and standing...")
        AGC.stopActionGroup()
        time.sleep(0.5)
        AGC.runActionGroup("stand", path=ACTION_DIR)
        return
    print("Movement test complete!")


if __name__ == "__main__":
    main()
