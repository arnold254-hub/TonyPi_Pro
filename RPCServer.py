#!/usr/bin/python3
# coding=utf8
import os
import sys
import time
import math
import logging
import threading
import subprocess
import re

from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

from jsonrpc import JSONRPCResponseManager, dispatcher
import hiwonder.yaml_handle as yaml_handle
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.ActionGroupControl as AGC
from ActionGroupDict import action_group_dict
from speech import speech

import Functions.Running as Running
import Functions.KickBall as KickBall
import Functions.Transport as Transport
import Functions.lab_adjust as lab_adjust
import Functions.ColorTrack as ColorTrack
import Functions.VisualPatrol as VisualPatrol

# Import MjpgServer for clearing overlay text
try:
    import MjpgServer
    MJPG_AVAILABLE = True
except ImportError:
    MJPG_AVAILABLE = False

# 远程调用api，框架jsonrpc(remote api call using the jsonrpc framework)
# 主要用于手机端和电脑端的客户端调用(primarily used for client-side calls between mobile and desktop applications)

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

__RPC_E01 = "E01 - Invalid number of parameter!"
__RPC_E02 = "E02 - Invalid parameter!"
__RPC_E03 = "E03 - Operation failed!"
__RPC_E04 = "E04 - Operation timeout!"
__RPC_E05 = "E05 - Not callable"

board = rrc.Board()
ctl = Controller(board)

ctl.enable_recv()        


ctl.set_buzzer(1900, 0.1, 0.9, 1) # 以1900Hz的频率，持续响0.1秒，关闭0.9秒，重复1次(at a frequency of 1900Hz, beep for 0.1 seconds, then silence for 0.9 seconds, repeating once)
time.sleep(0.1)

ColorTrack.initMove()
AGC.runActionGroup('stand_slow')

HWSONAR = None
QUEUE = None
sing_process = None
sing_lock = threading.Lock()

servo_data = None
# 加载配置文件数据(load configuration file data)
def load_config():
    global servo_data
    
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

load_config()


# app pwm舵机控制(pwm servo control)
# 参数1：时间（ms）(parameter1:time(ms))
# 参数2：舵机个数(parameter2:servo number)
# 参数3：舵机id（1或2）(parameter3:servo id(1 or 2))
# 参数4：舵机位置（500-2500）(parameter4:servo position(500-2500)
# ...舵机id(servo id)
# ...舵机位置(servo position)
# 例如：[1000, 2, 1, 1500, 5, 1500](such as [1000, 2, 1, 1500, 5, 1500])
@dispatcher.add_method
def SetPWMServo(*args, **kwargs):
    ret = (True, (), 'SetPWMServo')
    arglen = len(args)
    if 0 != (arglen % 2):
        return (False, __RPC_E01, 'SetPWMServo')
    try:
        servos = args[2:arglen:2]
        pulses = args[3:arglen:2]
        use_times = args[0]
        for s in servos:
            if s < 1 or s > 2:
                return (False, __RPC_E02, 'SetPWMServo')
        dat = zip(servos, pulses)
        for (s, p) in dat:
            if s == 2:
                p = p + servo_data['servo2'] - 1500
            ctl.set_pwm_servo_pulse(s, p ,use_times)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'SetPWMServo')
    return ret

# pwm舵机控制(pwm servo control)
# 参数1：时间（ms）(parameter1:time(ms))
# 参数2：舵机个数(parameter2:servo number)
# 参数3：舵机id（1或2）(parameter3:servo id(1 or 2))
# 参数4：舵机位置（500-2500）(parameter4:servo position(500-2500)
# ...舵机id(servo id)
# ...舵机位置(servo position)
# 例如：[1000, 2, 1, 1500, 5, 1500](such as [1000, 2, 1, 1500, 5, 1500])
@dispatcher.add_method
def set_pwm_servo(*args, **kwargs):
    ret = (True, (), 'set_pwm_servo')
    arglen = len(args)
    if 0 != (arglen % 2):
        return (False, __RPC_E01, 'set_pwm_servo')
    try:
        servos = args[2:arglen:2]
        pulses = args[3:arglen:2]
        use_times = args[0]
        for s in servos:
            if s < 1 or s > 2:
                return (False, __RPC_E02, 'set_pwm_servo')
        dat = zip(servos, pulses)
        for (s, p) in dat:
            ctl.set_pwm_servo_pulse(s, p ,use_times)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'set_pwm_servo')
    return ret


# 串口舵机控制(interface servo control)
# 参数1：时间（ms）(parameter1:time(ms))
# 参数2：舵机个数(parameter2:servo number)
# 参数3：舵机id(parameter3:servo id)
# 参数4：舵机位置（0-500）(parameter4:servo position(0-500)
# ...舵机id(servo id)
# ...舵机位置(servo position)
# ...
# 例如：[1000, 1, 1, 500](such as [1000, 1, 1, 500])
@dispatcher.add_method
def SetBusServoPulse(*args, **kwargs):
    ret = (True, (), 'SetBusServoPulse')   
    arglen = len(args)
    if (args[1] * 2 + 2) != arglen or arglen < 4:
        return (False, __RPC_E01, 'SetBusServoPulse')
    try:
        servos = args[2:arglen:2]
        pulses = args[3:arglen:2]
        use_times = args[0]
        for s in servos:
           if s < 1 or s > 16:
                return (False, __RPC_E02, 'SetBusServoPulse')
        dat = zip(servos, pulses)
        for (s, p) in dat:
            ctl.set_bus_servo_pulse(s, p ,use_times)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'SetBusServoPulse')
    return ret

# 串口舵机偏差设置(interface servo deviation setting)
# 参数：舵机偏差（-125-125）(parameter: servo deviation(-125-125))
@dispatcher.add_method
def SetBusServoDeviation(*args):
    ret = (True, (), 'SetBusServoDeviation')
    arglen = len(args)
    if arglen != 2:
        return (False, __RPC_E01, 'SetBusServoDeviation')
    try:
        servo = args[0]
        deviation = args[1]
        ctl.set_bus_servo_deviation(servo, deviation)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'SetBusServoDeviation')
    return ret

# 串口舵机偏差读取(interface servo deviation reading)
# 参数：readDeviation(parameter: readDeviation)
# 返回：1-16舵机偏差(return: 1-16 servo deviation)
@dispatcher.add_method
def GetBusServosDeviation(args):
    ret = (True, (), 'GetBusServosDeviation')
    data = []
    if args != "readDeviation":
        return (False, __RPC_E01, 'GetBusServosDeviation')
    try:
        for i in range(1, 16):
            dev = ctl.get_bus_servo_deviation(i)
            if dev is None:
                dev = 999
            data.append(dev)
        ret = (True, data, 'GetBusServosDeviation')
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'GetBusServosDeviation')
    return ret 

# 串口舵机偏差保存(interface servo deviation saving)
# 参数：downloadDeviation(parameter: downloadDeviation)
@dispatcher.add_method
def SaveBusServosDeviation(args):
    ret = (True, (), 'SaveBusServosDeviation')
    if args != "downloadDeviation":
        return (False, __RPC_E01, 'SaveBusServosDeviation')
    try:
        for i in range(1, 16):
            ctl.save_bus_servo_deviation(i)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'SaveBusServosDeviation')
    return ret 

# 串口舵机掉电(interface servo power loss)
# 参数：servoPowerDown(parameter:servoPowerDown)
@dispatcher.add_method
def UnloadBusServo(args):
    ret = (True, (), 'UnloadBusServo')
    if args != 'servoPowerDown':
        return (False, __RPC_E01, 'UnloadBusServo')
    try:
        for i in range(1, 16):
            ctl.unload_bus_servo(i)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'UnloadBusServo')
    return ret

# 获取串口舵机位置(obtain interface servo position)
# 参数：angularReadback(parameter: angularReadback)
# 返回：1-16舵机位置(return: 1-16 servo position)
@dispatcher.add_method
def GetBusServosPulse(args):
    ret = (True, (), 'GetBusServosPulse')
    data = []
    if args != 'angularReadback':
        return (False, __RPC_E01, 'GetBusServosPulse')
    try:
        for i in range(1, 16):
            pulse = ctl.get_bus_servo_pulse(i)
            if pulse is None:
                ret = (False, __RPC_E04, 'GetBusServosPulse')
                return ret
            else:
                data.append(pulse)
        ret = (True, data, 'GetBusServosPulse')
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'GetBusServosPulse')
    return ret 

# 停止当前动作(stop current action)
# 参数：stopAction(parameter:stopAction)
@dispatcher.add_method
def StopBusServo(args):
    ret = (True, (), 'StopBusServo')
    if args != 'stopAction':
        return (False, __RPC_E01, 'StopBusServo')
    try:     
        AGC.stopAction()
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'StopBusServo')
    return ret

# 停止当前动作组运行(stop current action group running)
# 参数：stopActionGroup(parameter: stopActionGroup)
@dispatcher.add_method
def StopActionGroup(args):
    ret = (True, (), 'StopActionGroup')
    if args != 'stopActionGroup':
        return (False, __RPC_E01, 'StopActionGroup')
    try:     
        AGC.stopActionGroup()
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'StopActionGroup')
    return ret

# 动作组运行(action group running)
# 参数1：动作编号（字符形式）(parameter1: action number (in character format))
# 参数2：动作次数（0表示循环）(parameter 2: Number of actions (0 indicates looping))
# 例如：['1', 2](such as: ['1', 2])
th = None
have_move = True
@dispatcher.add_method
def RunAction(*args_):
    global th
    global have_move 
    ret = (True, (), 'RunAction')
    actName = '0'
    times = 1
    print(args_) 
    if len(args_) != 2:
        return (False, __RPC_E01, 'RunAction')
    try:
        if args_[0] == '0':
            if have_move:
                AGC.stopActionGroup()
                have_move = False
        else:
            if th is not None:
                if not th.is_alive():
                    if args_[0] in action_group_dict:
                        actName = action_group_dict[args_[0]]
                    else:
                        actName = args_[0]
                    times = int(args_[1])
                    th = threading.Thread(target=AGC.runActionGroup, args=(actName, times), daemon=True)
                    th.start()
                    have_move = True
            else:
                if args_[0] in action_group_dict:
                    actName = action_group_dict[args_[0]]
                else:
                    actName = args_[0]
                times = int(args_[1])
                th = threading.Thread(target=AGC.runActionGroup, args=(actName, times), daemon=True)
                th.start()
                have_move = True
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03, 'RunAction')
    return ret


@dispatcher.add_method
def SingAndDance(param):
    global sing_process
    try:
        idx = int(param)
    except Exception:
        return (False, __RPC_E02, 'SingAndDance')

    if idx == 0:
        with sing_lock:
            proc = sing_process
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception as e:
                    print(f"[SingAndDance] stop error: {e}")
            sing_process = None
        try:
            AGC.runActionGroup('stand')
        except Exception as e:
            print(f"[SingAndDance] stand failed: {e}")
        return (True, (), 'SingAndDance')

    if idx not in [1, 2, 3, 4]:
        return (False, __RPC_E02, 'SingAndDance')

    script_path = os.path.join(os.path.dirname(__file__), 'Functions/voice_interaction/sing_and_dance.py')
    if not os.path.exists(script_path):
        return (False, __RPC_E03, 'SingAndDance')

    with sing_lock:
        proc = sing_process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                pass

        try:
            sing_process = subprocess.Popen(['python3', script_path, str(idx)])
        except Exception as e:
            print(f"[SingAndDance] start error: {e}")
            return (False, __RPC_E03, 'SingAndDance')

    return (True, (), 'SingAndDance')

@dispatcher.add_method
def NetWorkTest():
    ret = (True, False, 'NetWorkTest')
    try:
        result = subprocess.run(['curl', '-I', 'www.baidu.com'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=5)
        if result.returncode == 0:
            ret = (True, True, 'NetWorkTest')
        else:
            ret = (True, False, 'NetWorkTest')
    except subprocess.TimeoutExpired:
        ret = (True, False, 'NetWorkTest')
    except Exception as e:
        print(f"[NetWorkTest] error: {e}")
        ret = (True, False, 'NetWorkTest')
    return ret

@dispatcher.add_method
def SetAPIKey(api_key):
    ret = (True, (), 'SetAPIKey')
    if not api_key or not isinstance(api_key, str):
        return (False, __RPC_E02, 'SetAPIKey')

    config_path = os.path.expanduser('~/embodied_intelligence/common/config.py')

    if not os.path.exists(config_path):
        print(f"[SetAPIKey] config file not found: {config_path}")
        return (False, __RPC_E03, 'SetAPIKey')

    try:
        # Use ^ to match start of line to ensure exact match
        pattern = r"(?m)^(aliyun_api_key\s*=\s*)['\"].*?['\"]"
        replacement = f"\\1'{api_key}'"

        # Retry mechanism to handle concurrent access and file I/O issues
        max_retries = 3
        retry_delay = 0.05  # 50ms

        for attempt in range(max_retries):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # First check if the pattern exists in the file
            match = re.search(pattern, content)
            if not match:
                # Pattern not found - this is a real error
                if attempt < max_retries - 1:
                    print(f"[SetAPIKey] Match failed, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[SetAPIKey] aliyun_api_key not found in config after {max_retries} attempts")
                    return (False, __RPC_E03, 'SetAPIKey')

            # Pattern found, now do the replacement
            new_content = re.sub(pattern, replacement, content)

            if new_content == content:
                # Content unchanged means the value is already the same
                print(f"[SetAPIKey] API key is already set to the provided value (no change needed)")
                ret = (True, (), 'SetAPIKey')
                break

            # Content changed, write the file
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            if attempt > 0:
                print(f"[SetAPIKey] API key updated successfully (after {attempt + 1} attempts)")
            else:
                print(f"[SetAPIKey] API key updated successfully")
            ret = (True, (), 'SetAPIKey')
            break

    except Exception as e:
        print(f"[SetAPIKey] error: {e}")
        ret = (False, __RPC_E03, 'SetAPIKey')

    return ret

@dispatcher.add_method
def SetLLMKey(api_key):
    ret = (True, (), 'SetLLMKey')
    if not api_key or not isinstance(api_key, str):
        return (False, __RPC_E02, 'SetLLMKey')

    config_path = os.path.expanduser('~/embodied_intelligence/common/config.py')

    if not os.path.exists(config_path):
        print(f"[SetLLMKey] config file not found: {config_path}")
        return (False, __RPC_E03, 'SetLLMKey')

    try:
        # Get ASR_LANGUAGE to determine which key to update
        import os as os_env
        asr_language = os_env.environ.get('ASR_LANGUAGE', 'Chinese')

        if asr_language == 'Chinese':
            # Chinese mode: update aliyun_api_key
            pattern = r"(?m)^(aliyun_api_key\s*=\s*)['\"].*?['\"]"
            replacement = f"\\1'{api_key}'"
            print(f"[SetLLMKey] Setting Aliyun API key (Chinese mode)")
        else:
            # English mode: update llm_api_key (OpenAI)
            # Use ^ to match start of line to avoid matching vllm_api_key
            pattern = r"(?m)^(llm_api_key\s*=\s*)['\"].*?['\"]"
            replacement = f"\\1'{api_key}'"
            print(f"[SetLLMKey] Setting OpenAI API key (English mode)")

        # Retry mechanism to handle concurrent access and file I/O issues
        max_retries = 3
        retry_delay = 0.05  # 50ms

        for attempt in range(max_retries):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # First check if the pattern exists in the file
            match = re.search(pattern, content)
            if not match:
                # Pattern not found - this is a real error
                if attempt < max_retries - 1:
                    print(f"[SetLLMKey] Match failed, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[SetLLMKey] Target key not found in config after {max_retries} attempts")
                    return (False, __RPC_E03, 'SetLLMKey')

            # Pattern found, now do the replacement
            new_content = re.sub(pattern, replacement, content)

            if new_content == content:
                # Content unchanged means the value is already the same
                print(f"[SetLLMKey] LLM API key is already set to the provided value (no change needed)")
                ret = (True, (), 'SetLLMKey')
                break

            # Content changed, write the file
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            if attempt > 0:
                print(f"[SetLLMKey] LLM API key updated successfully (after {attempt + 1} attempts)")
            else:
                print(f"[SetLLMKey] LLM API key updated successfully")
            ret = (True, (), 'SetLLMKey')
            break

    except Exception as e:
        print(f"[SetLLMKey] error: {e}")
        ret = (False, __RPC_E03, 'SetLLMKey')

    return ret

@dispatcher.add_method
def SetVLLMKey(api_key):
    ret = (True, (), 'SetVLLMKey')
    if not api_key or not isinstance(api_key, str):
        return (False, __RPC_E02, 'SetVLLMKey')

    config_path = os.path.expanduser('~/embodied_intelligence/common/config.py')

    if not os.path.exists(config_path):
        print(f"[SetVLLMKey] config file not found: {config_path}")
        return (False, __RPC_E03, 'SetVLLMKey')

    try:
        # Get ASR_LANGUAGE to determine which key to update
        import os as os_env
        asr_language = os_env.environ.get('ASR_LANGUAGE', 'Chinese')

        if asr_language == 'Chinese':
            # Chinese mode: VLLM uses same key as LLM (aliyun), no need to configure separately
            print(f"[SetVLLMKey] Chinese mode - VLLM uses Aliyun key, no separate configuration needed")
            return (True, (), 'SetVLLMKey')
        else:
            # English mode: update vllm_api_key (OpenRouter)
            # Use ^ to match start of line to ensure exact match
            pattern = r"(?m)^(vllm_api_key\s*=\s*)['\"].*?['\"]"
            replacement = f"\\1'{api_key}'"
            print(f"[SetVLLMKey] Setting OpenRouter API key (English mode)")

            # Retry mechanism to handle concurrent access and file I/O issues
            max_retries = 3
            retry_delay = 0.05  # 50ms

            for attempt in range(max_retries):
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # First check if the pattern exists in the file
                match = re.search(pattern, content)
                if not match:
                    # Pattern not found - this is a real error
                    if attempt < max_retries - 1:
                        print(f"[SetVLLMKey] Match failed, retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[SetVLLMKey] vllm_api_key not found in config after {max_retries} attempts")
                        return (False, __RPC_E03, 'SetVLLMKey')

                # Pattern found, now do the replacement
                new_content = re.sub(pattern, replacement, content)

                if new_content == content:
                    # Content unchanged means the value is already the same
                    print(f"[SetVLLMKey] VLLM API key is already set to the provided value (no change needed)")
                    ret = (True, (), 'SetVLLMKey')
                    break

                # Content changed, write the file
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                if attempt > 0:
                    print(f"[SetVLLMKey] VLLM API key updated successfully (after {attempt + 1} attempts)")
                else:
                    print(f"[SetVLLMKey] VLLM API key updated successfully")
                ret = (True, (), 'SetVLLMKey')
                break

    except Exception as e:
        print(f"[SetVLLMKey] error: {e}")
        ret = (False, __RPC_E03, 'SetVLLMKey')

    return ret


@dispatcher.add_method
def GetAPIKey():
    """
    Get API keys based on current language setting

    Returns:
        Chinese mode: (True, {'llm_key': aliyun_api_key, 'vllm_key': ''}, 'GetAPIKey')
        English mode: (True, {'llm_key': openai_key, 'vllm_key': openrouter_key}, 'GetAPIKey')
    """
    config_path = os.path.expanduser('~/embodied_intelligence/common/config.py')

    if not os.path.exists(config_path):
        print(f"[GetAPIKey] config file not found: {config_path}")
        return (False, __RPC_E03, 'GetAPIKey')

    try:
        # Get ASR_LANGUAGE to determine which keys to return
        import os as os_env
        asr_language = os_env.environ.get('ASR_LANGUAGE', 'Chinese')

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        result = {}

        if asr_language == 'Chinese':
            # Chinese mode: return aliyun_api_key for llm, empty for vllm
            pattern_llm = r"(?m)^aliyun_api_key\s*=\s*['\"](.+?)['\"]"
            match_llm = re.search(pattern_llm, content)

            if match_llm:
                result['llm_key'] = match_llm.group(1)
            else:
                result['llm_key'] = ''

            result['vllm_key'] = ''
            result['language'] = 'Chinese'

            print(f"[GetAPIKey] Returning Aliyun API key (Chinese mode)")
        else:
            # English mode: return llm_api_key (OpenAI) and vllm_api_key (OpenRouter)
            pattern_llm = r"(?m)^llm_api_key\s*=\s*['\"](.+?)['\"]"
            pattern_vllm = r"(?m)^vllm_api_key\s*=\s*['\"](.+?)['\"]"

            match_llm = re.search(pattern_llm, content)
            match_vllm = re.search(pattern_vllm, content)

            if match_llm:
                result['llm_key'] = match_llm.group(1)
            else:
                result['llm_key'] = ''

            if match_vllm:
                result['vllm_key'] = match_vllm.group(1)
            else:
                result['vllm_key'] = ''
            result['language'] = 'English'

            print(f"[GetAPIKey] Returning OpenAI and OpenRouter API keys (English mode)")

        return (True, result, 'GetAPIKey')

    except Exception as e:
        print(f"[GetAPIKey] error: {e}")
        return (False, __RPC_E03, 'GetAPIKey')


# 跌倒起立检测(falling and standing up detection)
def standup():
    count1 = 0
    count2 = 0
    count3 = 0

    for i in range(20):
        try:
            data = ctl.get_imu() #获取传感器值(get sensor value)
            print("imu",data)
            angle_y = int(math.degrees(math.atan2(data[0], data[2]))) #转化为角度值(convert to the angle value)
            
            if abs(angle_y) > 160: 
                count1 += 1
            else:
                count1 = 0
            if abs(angle_y) < 10:
                count2 += 1
            else:
                count2 = 0
            time.sleep(0.1)
            count3 += 1
            if count3 > 5 and count1 < 3 and count2 < 3:
                break
            if count1 >= 10: #往后倒了一定时间后起来(lying down for a certain period and then getting up)
                count1 = 0
                AGC.runActionGroup('stand_up_back')
                break
            elif count2 >= 10: #后前倒了一定时间后起来(falling forward for a certain period of time and then standing up)
                count2 = 0
                AGC.runActionGroup('stand_up_front')
                break
        except BaseException as e:
            print(e)

# 跌倒起立调用(fall and stand up invocation)
th2 = None
@dispatcher.add_method
def StandUp():
    global th2

    ret = (True, (), 'StandUp')

    if th2 is not None:
        if not th2.is_alive():
            th2 = threading.Thread(target=standup)
            th2.start()
        else:
            pass
    else:
        th2 = threading.Thread(target=standup)
        th2.start()

    return ret

def runbymainth(req, pas):
    if callable(req):
        # print('pas', req)
        event = threading.Event()
        ret = [event, pas, None]
        QUEUE.put((req, ret))
        count = 0
        timeout = 200
        try:
            if req.__name__ in ('loadFunc', 'unloadFunc', 'startFunc', 'stopFunc', 'finishFunc'):
                timeout = 1000
        except AttributeError:
            pass
        #ret[2] =  req(pas)
        # print('ret', ret)
        while ret[2] is None:
            time.sleep(0.01)
            count += 1
            if count > timeout:
                break
        if ret[2] is not None:
            if ret[2][0]:
                return ret[2]
            else:
                return (False, __RPC_E03 + " " + ret[2][1])
        else:
            return (False, __RPC_E04)
    else:
        return (False, __RPC_E05)

@dispatcher.add_method
def LoadFunc(new_func = 0):
    ret = runbymainth(Running.loadFunc, (new_func, ))

    # Reset head servos when loading AgentsRouter (function 13, large model functionality)
    if new_func in (13, 14, 15, 16, 17, 18) and ret[0]:
        try:
            # Reset servo 1 (pitch) and servo 2 (yaw) to center position (1500)
            ctl.set_pwm_servo_pulse(1, 1500, 500)  # Pitch to horizontal
            ctl.set_pwm_servo_pulse(2, 1500, 500)  # Yaw to forward
            print("[LoadFunc] Head servos reset to center position for AgentsRouter")
        except Exception as e:
            print(f"[LoadFunc] Failed to reset head servos: {e}")

    return ret

@dispatcher.add_method
def UnloadFunc():
    ret = runbymainth(Running.unloadFunc, ())
    # Clear overlay text when unloading function
    if MJPG_AVAILABLE:
        try:
            MjpgServer.clear_overlay_text()
        except Exception:
            pass
    return ret

@dispatcher.add_method
def StartFunc():
    return runbymainth(Running.startFunc, ())

@dispatcher.add_method
def StopFunc():
    ret = runbymainth(Running.stopFunc, ())
    # Clear overlay text when stopping function
    if MJPG_AVAILABLE:
        try:
            MjpgServer.clear_overlay_text()
        except Exception:
            pass
    return ret

@dispatcher.add_method
def FinishFunc():
    ret = runbymainth(Running.finishFunc, ())
    # Clear overlay text when finishing function
    if MJPG_AVAILABLE:
        try:
            MjpgServer.clear_overlay_text()
        except Exception:
            pass
    return ret

@dispatcher.add_method
def Heartbeat():
    # Heartbeat 只需要刷新一个时间戳，直接调用即可，避免排队等待主线程导致 E04
    return Running.doHeartbeat()

@dispatcher.add_method
def GetRunningFunc():
    #return runbymainth("GetRunningFunc", ())
    return (True, (0,))

# 设置颜色阈值
# 参数：颜色lab
# 例如：[{'red': ((0, 0, 0), (255, 255, 255))}]
@dispatcher.add_method
def SetLABValue(*lab_value):
    #print(lab_value)
    return runbymainth(lab_adjust.setLABValue, lab_value)

# 保存颜色阈值
@dispatcher.add_method
def GetLABValue():
    return (True, lab_adjust.getLABValue()[1], 'GetLABValue')

# 保存颜色阈值
@dispatcher.add_method
def SaveLABValue(color=''):
    return runbymainth(lab_adjust.saveLABValue, (color, ))

@dispatcher.add_method
def HaveLABAdjust():
    return (True, True, 'HaveLABAdjust')

@dispatcher.add_method
def SetPoint(*point):
    if point[0] == 2:
        return runbymainth(KickBall.set_point, point[1])
    elif point[0] == 4:
        return runbymainth(VisualPatrol.set_point, point[1])
    elif point[0] == 5:
        return runbymainth(ColorTrack.set_point, point[1])

@dispatcher.add_method
def SetThreshold(*value):
    if value[0] == 2:
        return runbymainth(KickBall.set_threshold, value[1])
    elif value[0] == 4:
        return runbymainth(VisualPatrol.set_threshold, value[1])
    elif value[0] == 5:
        return runbymainth(ColorTrack.set_threshold, value[1])

@dispatcher.add_method
def GetRGBValue(value):
    if value == 2:
        return (True, KickBall.get_rgb_value(), 'GetRGBValue')
    elif value == 4:
        return (True, VisualPatrol.get_rgb_value(), 'GetRGBValue')
    elif value == 5:
        return (True, ColorTrack.get_rgb_value(), 'GetRGBValue')

@Request.application
def application(request):
    dispatcher["echo"] = lambda s: s
    dispatcher["add"] = lambda a, b: a + b
    response = JSONRPCResponseManager.handle(request.data, dispatcher)

    return Response(response.json, mimetype='application/json')

def startRPCServer():
    # Enable threading so long-running requests (e.g. LoadFunc for large models)
    # do not block heartbeat RPCs and cause client-side timeouts.
    run_simple('', 9030, application, threaded=True)

if __name__ == '__main__':
    startRPCServer()
