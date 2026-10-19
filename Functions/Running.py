#!/usr/bin/python3
# coding=utf8
import sys
import time
import threading
import os

LOG_FILE = '/home/pi/TonyPi/tonypi_service.log'


def log(message):
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as fp:
            fp.write(f"[Running] {message}\n")
    except Exception:
        pass

import Functions.KickBall as KickBall
import Functions.Transport as Transport
import Functions.ColorTrack as ColorTrack
import Functions.FaceDetect as FaceDetect
import Functions.lab_adjust as lab_adjust
import Functions.ColorDetect as ColorDetect
import Functions.VisualPatrol as VisualPatrol
import Functions.RemoteControl as RemoteControl
import Functions.ApriltagDetect as ApriltagDetect
import Functions.gesture_control as GestureControl 
import Extend.athletics_course.hurdles as Hurdles
import Extend.athletics_course.stairway as Stairway
import Extend.vision_grab_course.color_classify as ColorClassify
import Functions.AgentChat as AgentChat
import Functions.AgentAction as AgentAction
import Functions.AgentKickBall as AgentKickBall
import Functions.AgentTrack as AgentTrack
import Functions.AgentScene as AgentScene
import Functions.AgentWeather as AgentWeather

RunningFunc = 0
LastHeartbeat = 0
cam = None
open_once = False

FUNCTIONS = {
    1: RemoteControl, # 机体控制(remote control)
    2: KickBall,      # 自动踢球(auto shooting)
    3: GestureControl,# 颜色识别(color recognition)
    4: VisualPatrol,  # 智能巡线 (intelligent line follow)
    5: ColorTrack,    # 云台跟踪(pan-tilt tracking)
    6: FaceDetect,    # 人脸识别(face recognition)
    7: ApriltagDetect,# 标签识别(tag recognition)
    8: Transport,     # 智能搬运(intelligent transportation)
    9: lab_adjust,     # lab阈值调节(lab threshold adjustment)
    10: Hurdles ,     # 跨栏避障(hurdles)
    11: Stairway ,    # 上下台阶(go up and down stairs)
    12: ColorClassify, # 色块分类(color block classification)
    13: AgentChat,     # 大模型-对话
    14: AgentAction,   # 大模型-动作
    15: AgentKickBall, # 大模型-踢球
    16: AgentTrack,    # 大模型-追踪
    17: AgentScene,    # 大模型-场景描述
    18: AgentWeather   # 大模型-天气
}

def doHeartbeat(tmp=()):
    global LastHeartbeat
    LastHeartbeat = time.time() + 7
    return (True, ())

def CurrentEXE():
    global RunningFunc
    return FUNCTIONS[RunningFunc]

def loadFunc(newf):
    global RunningFunc
    new_func = newf[0]

    doHeartbeat()

    if new_func not in FUNCTIONS:
        return (False,  sys._getframe().f_code.co_name + ": Invalid argument")
    else:
        try:
            log(f"loadFunc called with {new_func}, current {RunningFunc}")
            if RunningFunc in FUNCTIONS and RunningFunc != new_func:
                log(f"calling exit on {RunningFunc}")
                FUNCTIONS[RunningFunc].exit()
            RunningFunc = newf[0]
            if not open_once:
                if cam is not None:
                    cam.camera_close()
                    cam.camera_open()
            log(f"calling init on {RunningFunc}")
            FUNCTIONS[RunningFunc].init()
            log(f"init done for {RunningFunc}")
        except Exception as e:
            log(f"loadFunc exception: {e}")
            print(e)
    return (True, (RunningFunc,))

def unloadFunc(tmp = ()):
    global RunningFunc

    # Check if any function is loaded
    if RunningFunc == 0:
        log("[unloadFunc] No function loaded (RunningFunc=0), nothing to unload")
        return (True, (0,))

    # Check if function exists in FUNCTIONS
    if RunningFunc not in FUNCTIONS:
        log(f"[unloadFunc] Invalid RunningFunc={RunningFunc}, resetting to 0")
        RunningFunc = 0
        return (True, (0,))

    # Try to exit the function
    try:
        log(f"unloadFunc exit {RunningFunc}")
        FUNCTIONS[RunningFunc].exit()
        RunningFunc = 0
    except Exception as e:
        log(f"[unloadFunc] Error exiting function {RunningFunc}: {e}")
        print(f"[unloadFunc] Error exiting function {RunningFunc}: {e}")
        RunningFunc = 0  # Reset to 0 even if exit fails

    # Close camera if not in open_once mode
    if not open_once:
        if cam is not None:
            try:
                cam.camera_close()
            except Exception as e:
                log(f"[unloadFunc] Error closing camera: {e}")

    return (True, (0,))

def getLoadedFunc(newf):
    global RunningFunc
    return (True, (RunningFunc,))

def startFunc(tmp):
    global RunningFunc
    log(f"startFunc called with RunningFunc={RunningFunc}")

    # Check if any function is loaded
    if RunningFunc == 0:
        log("[startFunc] No function loaded (RunningFunc=0), ignoring start request")
        return (True, (0,))

    # Check if function exists in FUNCTIONS
    if RunningFunc not in FUNCTIONS:
        log(f"[startFunc] Invalid RunningFunc={RunningFunc}, resetting to 0")
        RunningFunc = 0
        return (True, (0,))

    # Try to start the function
    try:
        FUNCTIONS[RunningFunc].start()
        log(f"startFunc finished for {RunningFunc}")
        return (True, (RunningFunc,))
    except Exception as e:
        log(f"[startFunc] Error starting function {RunningFunc}: {e}")
        print(f"[startFunc] Error starting function {RunningFunc}: {e}")
        return (True, (RunningFunc,))

def stopFunc(tmp):
    global RunningFunc
    log(f"stopFunc called with RunningFunc={RunningFunc}")

    # Check if any function is loaded
    if RunningFunc == 0:
        log("[stopFunc] No function loaded (RunningFunc=0), ignoring stop request")
        return (True, (0,))

    # Check if function exists in FUNCTIONS
    if RunningFunc not in FUNCTIONS:
        log(f"[stopFunc] Invalid RunningFunc={RunningFunc}, resetting to 0")
        RunningFunc = 0
        return (True, (0,))

    # Try to stop the function
    try:
        FUNCTIONS[RunningFunc].stop()
        log(f"stopFunc finished for {RunningFunc}")
        return (True, (RunningFunc,))
    except Exception as e:
        log(f"[stopFunc] Error stopping function {RunningFunc}: {e}")
        print(f"[stopFunc] Error stopping function {RunningFunc}: {e}")
        return (True, (RunningFunc,))

def finishFunc(tmp = ()):
    """Finish function - similar to stop but may have different semantics in some functions"""
    global RunningFunc
    log(f"finishFunc called with RunningFunc={RunningFunc}")

    # Check if any function is loaded
    if RunningFunc == 0:
        log("[finishFunc] No function loaded (RunningFunc=0), ignoring finish request")
        return (True, (0,))

    # Check if function exists in FUNCTIONS
    if RunningFunc not in FUNCTIONS:
        log(f"[finishFunc] Invalid RunningFunc={RunningFunc}, resetting to 0")
        RunningFunc = 0
        return (True, (0,))

    # Try to finish/stop the function (use stop as finish is not always implemented)
    try:
        # Try finish method if exists, otherwise use stop
        func = FUNCTIONS[RunningFunc]
        if hasattr(func, 'finish'):
            func.finish()
            log(f"finishFunc finished for {RunningFunc}")
        else:
            func.stop()
            log(f"finishFunc (via stop) finished for {RunningFunc}")
        return (True, (RunningFunc,))
    except Exception as e:
        log(f"[finishFunc] Error finishing function {RunningFunc}: {e}")
        print(f"[finishFunc] Error finishing function {RunningFunc}: {e}")
        return (True, (RunningFunc,))

def heartbeatTask():
    global LastHeartbeat
    global RunningFunc
    while True:
        try:
            if LastHeartbeat < time.time():
                if RunningFunc != 0:
                    unloadFunc()
            time.sleep(0.1)
        except Exception as e:
            print(e)

        except KeyboardInterrupt:
            break

threading.Thread(target=heartbeatTask, daemon=True).start()
