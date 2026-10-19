#!/usr/bin/python3
# coding=utf8
import sys
import os
import time
import json
import pygame
import asyncio
import threading
import websockets
import subprocess
import logging
import hiwonder.ActionGroupControl as AGC
import hiwonder.ros_robot_controller_sdk as rrc

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create robot control board instance
board = rrc.Board()

# Define two types of gamepad button mappings
# SHANWAN Android Gamepad button mapping
BUTTONS_SHANWAN = {
    "PSB_CROSS": 0, "PSB_CIRCLE": 1, "PSB_SQUARE": 3, "PSB_TRIANGLE": 4,
    "PSB_L1": 6, "PSB_R1": 7, "PSB_L2": 8, "PSB_R2": 9,
    "PSB_SELECT": 10, "PSB_START": 11, "PSB_L3": 13, "PSB_R3": 14
}

# Default gamepad button mapping
BUTTONS_DEFAULT = {
    "PSB_TRIANGLE": 0, "PSB_CIRCLE": 1, "PSB_CROSS": 2, "PSB_SQUARE": 3,
    "PSB_L1": 4, "PSB_R1": 5, "PSB_L2": 6, "PSB_R2": 7,
    "PSB_SELECT": 8, "PSB_START": 9, "L_AXIS_BUTTON": 10, "R_AXIS_BUTTON": 11
}

# Action name mapping
action_map = ["CROSS", "CIRCLE", "", "SQUARE", "TRIANGLE", "L1", "R1", "L2", "R2", 
              "SELECT", "START", "", "L3", "R3"]

# Global variables
joystick_map = BUTTONS_DEFAULT  # Default to standard mapping
last_buttons = {}
athletics_perform_finish = True
th = None
last_status = ''
connected = False
action_name = None
status = 'init'
aplay_thread = None

def joystick_init():
    """Initialize joystick (integrate two mapping methods)"""
    global joystick_map, last_buttons, connected
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() > 0:
        try:
            js = pygame.joystick.Joystick(0)
            js.init()
            js_name = js.get_name()
            logging.info(f"Joystick detected: {js_name}")
            
            # Select mapping based on joystick type
            if "SHANWAN" in js_name.upper():
                joystick_map = BUTTONS_SHANWAN
                logging.info("Using SHANWAN gamepad mapping")
            else:
                joystick_map = BUTTONS_DEFAULT
                logging.info("Using default gamepad mapping")
            
            # Initialize button state recording
            last_buttons = {key: 0 for key in joystick_map}
            
            # Print joystick info
            logging.info(f"Axis count: {js.get_numaxes()}")
            logging.info(f"Button count: {js.get_numbuttons()}")
            logging.info(f"Trackball count: {js.get_numballs()}")
            logging.info(f"DPad count: {js.get_numhats()}")
            
            connected = True
            return js
        except Exception as e:
            logging.error(f"Joystick initialization failed: {e}")
            connected = False
            return None
    else:
        logging.warning("No joystick device detected")
        connected = False
        return None

async def call_rpc(method, params=None):
    """Call remote RPC method"""
    websocket = None
    try:
        websocket = await websockets.connect('ws://localhost:7788/up')
        call = dict(jsonrpc='2.0', method=method)
        if params is not None:
            call['params'] = params
        await websocket.send(json.dumps(call))
        await websocket.close()
    except Exception as e:
        logging.error(f"RPC call failed: {e}")
        if websocket is not None:
            await websocket.close()

async def run_action_set(action_set_name, repeat):
    """Run action set"""
    await call_rpc('run_action_set', [action_set_name, repeat])

async def stop(action_set_name=None):
    """Stop current action"""
    await call_rpc('stop')
    if action_set_name is not None:
        await run_action_set(action_set_name, 1)

# Main loop
while True:
    # Check joystick connection status
    if os.path.exists("/dev/input/js0"):
        if not connected:
            js = joystick_init()
            if connected:
                logging.info("Joystick connected successfully")
    else:
        if connected:
            connected = False
            pygame.joystick.quit()
            logging.info("Joystick disconnected")
    
    # Handle connected joystick logic
    if connected:
        pygame.event.pump()     
        actName = None
        times = 1
        
        try:
            # Get current button states
            buttons = {}
            for key in joystick_map:
                buttons[key] = js.get_button(joystick_map[key])
            
            # 打印按键变化
            pressed = [k for k in buttons if buttons[k] and not last_buttons.get(k)]
            if pressed:
                logging.info(f"按键按下: {pressed}")

            # SELECT key combination functions
            if js.get_button(joystick_map["PSB_SELECT"]):
                time.sleep(0.01)
                if last_buttons != buttons:
                    # Set volume
                    os.system('amixer -q -D pulse set Master 20%')
                    action_name = None
                    
                    # SELECT + START: Start athletics performance
                    if js.get_button(joystick_map["PSB_START"]):
                        athletics_perform_finish = False
                        subprocess.Popen("python3 /home/pi/TonyPi/Extend/athletics_course/athletics_perform.py", shell=True)
                        time.sleep(1)
                    
                    # SELECT + L1: Action 21
                    elif js.get_button(joystick_map["PSB_L1"]):
                        action_name = '21'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + L2: Action 22
                    elif js.get_button(joystick_map["PSB_L2"]):
                        action_name = '22'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + R1: Action 23
                    elif js.get_button(joystick_map["PSB_R1"]):
                        action_name = '23'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + R2: Action 24
                    elif js.get_button(joystick_map["PSB_R2"]):
                        action_name = '24'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + CIRCLE: Action 18
                    elif js.get_button(joystick_map["PSB_CIRCLE"]):
                        action_name = '18'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + SQUARE: Action 19
                    elif js.get_button(joystick_map["PSB_SQUARE"]):
                        action_name = '19'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + TRIANGLE: Action 20
                    elif js.get_button(joystick_map["PSB_TRIANGLE"]):
                        action_name = '20'
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                    
                    # SELECT + CROSS: Stop athletics performance
                    elif js.get_button(joystick_map["PSB_CROSS"]):
                        board.set_buzzer(1900, 0.1, 0.1, 2)
                        athletics_perform_finish = True
                        os.system("/home/pi/TonyPi/Extend/test.sh")
            
            # Handle regular buttons when athletics performance not active
            elif athletics_perform_finish:
                # Right shoulder button functions
                if js.get_button(joystick_map["PSB_R1"]):
                    actName = 'right_kick'
                if js.get_button(joystick_map["PSB_R2"]):
                    actName = 'turn_right'
                
                # Left shoulder button functions
                if js.get_button(joystick_map["PSB_L1"]):
                    actName = 'left_kick'
                if js.get_button(joystick_map["PSB_L2"]):
                    actName = 'turn_left'
                
                # Main button functions
                if js.get_button(joystick_map["PSB_SQUARE"]):  # Square
                    actName = 'twist'
                if js.get_button(joystick_map["PSB_CIRCLE"]):  # Circle
                    actName = 'right_shot_fast'
                if js.get_button(joystick_map["PSB_TRIANGLE"]):  # Triangle
                    actName = 'wave'
                if js.get_button(joystick_map["PSB_CROSS"]):  # Cross
                    actName = 'bow'
                
                # DPad functions
                hat = js.get_hat(0)
                if hat[0] == 0 and hat[1] == 1:  # Up
                    last_status = 'go'
                    actName = 'go_forward'
                    times = 0
                elif hat[0] == 0 and hat[1] == -1:  # Down
                    last_status = 'back_fast'
                    actName = 'back_fast'
                    times = 0
                elif hat[0] == -1 and hat[1] == 0:  # Left
                    actName = 'left_move'
                elif hat[0] == 1 and hat[1] == 0:  # Right
                    actName = 'right_move'
                else:
                    # Joystick control
                    lx = js.get_axis(0)
                    ly = js.get_axis(1)
                    
                    if lx < -0.5:
                        actName = 'left_move'
                    elif lx > 0.5:
                        actName = 'right_move'
                    if ly < -0.5:
                        last_status = 'go'
                        actName = 'go_forward'
                        times = 0
                    elif ly > 0.5:
                        last_status = 'back_fast'
                        actName = 'back_fast'
                        times = 0
                    else:
                        if (last_status == 'go' or last_status == 'back_fast') and actName is None:
                            AGC.stopActionGroup()
                            last_status = ''
                
                # START button function
                if js.get_button(joystick_map["PSB_START"]):
                    if status == 'play':
                        AGC.stopAction()
                        if aplay_thread is not None:
                            aplay_thread.terminate()
                            aplay_thread.wait()
                            aplay_thread = None
                            time.sleep(0.5)
                        status = 'init'
                    actName = 'stand_slow'
                    board.set_buzzer(1900, 0.1, 0.9, 1)
            
            # Execute action group
            if action_name is not None:
                threading.Thread(target=AGC.runActionGroup, args=(action_name, 1, True), daemon=True).start()
                aplay_thread = subprocess.Popen(
                    ["aplay", f"/home/pi/TonyPi/audio/{action_name}.wav"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                status = 'play'
                action_name = None
            
            # Run action
            if actName is not None:
                if th is not None:
                    if not th.is_alive():
                        asyncio.run(run_action_set(actName, 1))
                        th = threading.Thread(target=AGC.runActionGroup, args=(actName, times), daemon=True)
                        th.start()
                else:
                    asyncio.run(run_action_set(actName, 1))
                    th = threading.Thread(target=AGC.runActionGroup, args=(actName, times), daemon=True)
                    th.start()
            
            # Save current button states
            last_buttons = buttons
            
        except Exception as e:
            logging.error(f"Joystick processing error: {e}")
            connected = False
    
    time.sleep(0.01)
