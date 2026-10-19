#!/usr/bin/python3
# coding=utf8
import sys
import os
import signal
import threading
import subprocess
from speech import speech
import hiwonder.ActionGroupControl as AGC

ACTION_AUDIO_MAP = {
    1: ('dance1', '16.wav'),
    2: ('dance2', '17.wav'),
    3: ('dance3', '18.wav'),
    4: ('dance4', '19.wav'),
}

AUDIO_DIR = '/home/pi/TonyPi/audio'
ACTION_GROUP_DIR = '/home/pi/TonyPi/ActionGroups'

audio_process = None
stop_flag = threading.Event()

def signal_handler(sig, frame):
    print("[SingAndDance] Received signal, stopping...")
    stop_flag.set()
    if audio_process and audio_process.poll() is None:
        try:
            audio_process.terminate()
            audio_process.wait(timeout=1)
        except Exception:
            pass
    try:
        AGC.stopActionGroup()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def play_audio(audio_path):
    global audio_process
    try:
        speech.set_volume(80)
        audio_process = subprocess.Popen(['aplay', '-q', audio_path])
        audio_process.wait()
    except Exception as e:
        print(f"[SingAndDance] audio error: {e}")
    finally:
        audio_process = None

def main():
    if len(sys.argv) < 2:
        print("Usage: sing_and_dance.py <index>")
        sys.exit(1)

    try:
        idx = int(sys.argv[1])
    except ValueError:
        print("Invalid parameter")
        sys.exit(1)

    if idx not in ACTION_AUDIO_MAP:
        print(f"Invalid index: {idx}")
        sys.exit(1)

    action_name, audio_file = ACTION_AUDIO_MAP[idx]
    action_path = os.path.join(ACTION_GROUP_DIR, action_name + '.d6a')
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(action_path):
        print(f"Action file not found: {action_path}")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    audio_thread = threading.Thread(target=play_audio, args=(audio_path,), daemon=True)
    audio_thread.start()

    try:
        print(f"[SingAndDance] executing action {action_name}")
        AGC.runActionGroup(action_name)
        print(f"[SingAndDance] action {action_name} finished")
    except Exception as e:
        print(f"[SingAndDance] action error: {e}")

    audio_thread.join()

    if not stop_flag.is_set():
        try:
            AGC.runActionGroup('stand')
        except Exception as e:
            print(f"[SingAndDance] stand failed: {e}")

if __name__ == '__main__':
    main()
