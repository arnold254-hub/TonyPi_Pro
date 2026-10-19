#!/usr/bin/python3
# coding=utf8
"""
ShakeHand: greet people the camera sees.

Behaviour
    1. The head sweeps left and right looking for a face (a background
       thread moves it on a fixed timer, so the sweep is smooth no matter
       how fast face detection runs).
    2. When a face is found, the head turns to follow it (pan + tilt).
    3. Once the face is settled near the centre of the image, the robot
       says "Bonjour" and runs the 'wave' action group at the same time.
    4. It will not greet the same person again until they leave the frame
       for a couple of seconds and a cooldown has passed.

Head servos (PWM): 1 = tilt, 2 = pan. The tilt rest position is 1800, the
same as the stock face-detection mode, so the small robot looks up at a
standing person. The pan centre offset comes from servo_config.yaml.

Audio: plays /home/pi/TonyPi/audio/bonjour.wav (shipped with the project).

Run standalone on the robot as the pi user (no sudo):
    cd /home/pi/TonyPi/Functions && python3 ShakeHand.py
The module also exposes init/start/stop/exit/run so it can be added to
Functions/Running.py and driven from the app.
"""
import os
import sys
import time
import shutil
import threading
import subprocess

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except ImportError:
    HAVE_MEDIAPIPE = False

sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")
sys.path.append("/home/pi/TonyPi")

import hiwonder.PID as PID
import hiwonder.Camera as Camera
import hiwonder.yaml_handle as yaml_handle
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.ActionGroupControl as AGC

try:
    from speech import speech          # Hiwonder speaker helper installed on the robot
    HAVE_SPEECH = True
except ImportError:
    HAVE_SPEECH = False

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# ----------------------------------------------------------------- settings
AUDIO_DIR = "/home/pi/TonyPi/audio"
BONJOUR_WAV = os.path.join(AUDIO_DIR, "bonjour.wav")
GREETING_TEXT = "Bonjour"
GREET_ACTION = "wave"

TILT, PAN = 1, 2
TILT_CENTER = 1800              # stock FaceDetect rest position: look up at people
TILT_MIN, TILT_MAX = 1200, 2200
PAN_MIN, PAN_MAX = 1000, 2000
SCAN_STEP = 10                  # pulse change per scan tick
SCAN_TICK = 0.05                # seconds between scan ticks
DETECT_WIDTH = 320              # detection runs on a frame this wide (speed)
CENTER_TOLERANCE = 20           # px; smaller error than this is "centred"
GREET_ZONE = 1 / 4.0            # face within this fraction of width from centre
STABLE_FRAMES = 4               # consecutive centred frames before greeting
LOST_FRAMES = 10                # frames without a face before we call it lost
GREET_COOLDOWN = 8.0            # seconds between greetings
LOST_RESET_TIME = 2.0           # seconds a face must be gone before re-arming

# ----------------------------------------------------------------- hardware
board = rrc.Board()
ctl = Controller(board)

servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
PAN_CENTER = int(servo_data.get("servo2", 1500))

x_pid = PID.PID(P=0.145, I=0.00, D=0.0007)
y_pid = PID.PID(P=0.145, I=0.00, D=0.0007)

# ----------------------------------------------------------------- face detector
# Prefer MediaPipe (accurate), otherwise use OpenCV's Haar cascade, which needs
# nothing beyond OpenCV itself. The cascade file is bundled in Functions/model.
CASCADE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "haarcascade_frontalface_default.xml"),
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]
if hasattr(cv2, "data"):
    CASCADE_PATHS.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


class MediapipeFaces:
    name = "MediaPipe"

    def __init__(self):
        self.det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)

    def find(self, img):
        h, w = img.shape[:2]
        res = self.det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = []
        if res.detections:
            for d in res.detections:
                r = d.location_data.relative_bounding_box
                boxes.append((int(r.xmin * w), int(r.ymin * h), int(r.width * w), int(r.height * h)))
        return boxes


class HaarFaces:
    name = "OpenCV Haar cascade"

    def __init__(self):
        path = next((p for p in CASCADE_PATHS if os.path.isfile(p)), None)
        if path is None:
            raise RuntimeError("haarcascade_frontalface_default.xml not found in: %s" % CASCADE_PATHS)
        self.det = cv2.CascadeClassifier(path)
        if self.det.empty():
            raise RuntimeError("failed to load cascade %s" % path)

    def find(self, img):
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        faces = self.det.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
        return [tuple(int(v) for v in f) for f in faces]


detector = MediapipeFaces() if HAVE_MEDIAPIPE else HaarFaces()
print("ShakeHand: face detector = %s" % detector.name)


def find_faces(img):
    """Run the detector on a downscaled copy and map boxes back to full size."""
    h, w = img.shape[:2]
    if w > DETECT_WIDTH:
        scale = w / float(DETECT_WIDTH)
        small = cv2.resize(img, (DETECT_WIDTH, int(h / scale)), interpolation=cv2.INTER_AREA)
        return [tuple(int(v * scale) for v in b) for b in detector.find(small)]
    return detector.find(img)


def largest_face(boxes):
    best = None
    for bbox in boxes:
        if best is None or bbox[2] * bbox[3] > best[2] * best[3]:
            best = bbox
    return best

# ----------------------------------------------------------------- state
__isRunning = False
pan_pulse = PAN_CENTER
tilt_pulse = TILT_CENTER
scan_dir = SCAN_STEP
stable_count = 0
miss_count = 0
face_seen = False
face_lost_at = 0.0
last_greet_at = 0.0
armed = True
greeting = False
head_lock = threading.Lock()


def set_head(pan, tilt, ms):
    global pan_pulse, tilt_pulse
    with head_lock:
        pan_pulse = int(max(PAN_MIN, min(PAN_MAX, pan)))
        tilt_pulse = int(max(TILT_MIN, min(TILT_MAX, tilt)))
        ctl.set_pwm_servo_pulse(PAN, pan_pulse, ms)
        ctl.set_pwm_servo_pulse(TILT, tilt_pulse, ms)


def initMove():
    set_head(PAN_CENTER, TILT_CENTER, 500)
    time.sleep(0.5)


def reset():
    global stable_count, miss_count, face_seen, armed, greeting, scan_dir
    stable_count = 0
    miss_count = 0
    face_seen = False
    armed = True
    greeting = False
    scan_dir = SCAN_STEP
    x_pid.clear()
    y_pid.clear()
    initMove()

# ----------------------------------------------------------------- head sweep thread
def scan_loop():
    """Sweep the head left and right whenever no face is being tracked."""
    global scan_dir
    while True:
        if __isRunning and not face_seen and not greeting:
            if pan_pulse >= PAN_MAX:
                scan_dir = -abs(SCAN_STEP)
            elif pan_pulse <= PAN_MIN:
                scan_dir = abs(SCAN_STEP)
            set_head(pan_pulse + scan_dir, TILT_CENTER, int(SCAN_TICK * 1000))
        time.sleep(SCAN_TICK)


threading.Thread(target=scan_loop, daemon=True).start()

# ----------------------------------------------------------------- speech
def _ensure_bonjour_wav():
    """The wav ships with the project; synthesize it only if it went missing."""
    if os.path.isfile(BONJOUR_WAV):
        return True
    os.makedirs(AUDIO_DIR, exist_ok=True)
    tools = [
        (["espeak-ng", "-v", "fr", "-s", "140", "-w", BONJOUR_WAV, GREETING_TEXT], "espeak-ng"),
        (["espeak", "-v", "fr", "-s", "140", "-w", BONJOUR_WAV, GREETING_TEXT], "espeak"),
        (["pico2wave", "-l", "fr-FR", "-w", BONJOUR_WAV, GREETING_TEXT], "pico2wave"),
    ]
    for cmd, name in tools:
        if shutil.which(name) is None:
            continue
        try:
            subprocess.run(cmd, check=True, timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.isfile(BONJOUR_WAV):
                print("ShakeHand: generated %s with %s" % (BONJOUR_WAV, name))
                return True
        except Exception as e:
            print("ShakeHand: %s failed: %s" % (name, e))
    return False


def say_bonjour():
    """Play the greeting through the robot speaker."""
    if not _ensure_bonjour_wav():
        print("ShakeHand: %s is missing. Copy audio/bonjour.wav from the project "
              "to the robot, or install espeak (sudo apt install espeak)." % BONJOUR_WAV)
        return
    try:
        if HAVE_SPEECH:
            speech.play_audio(BONJOUR_WAV)
        else:
            subprocess.run(["aplay", "-q", BONJOUR_WAV], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("ShakeHand: audio playback failed: %s" % e)


def greet():
    """Say bonjour and wave at the same time, then re-enable tracking."""
    global greeting, last_greet_at
    print("ShakeHand: Bonjour!")
    speaker = threading.Thread(target=say_bonjour, daemon=True)
    speaker.start()
    try:
        AGC.runActionGroup(GREET_ACTION)
    finally:
        speaker.join(timeout=5)
        last_greet_at = time.time()
        greeting = False

# ----------------------------------------------------------------- app hooks
def init():
    print("ShakeHand Init")
    if HAVE_SPEECH:
        try:
            speech.set_volume(80)
        except Exception:
            pass
    if not os.path.isfile(BONJOUR_WAV):
        print("ShakeHand: warning, %s not found; the robot will wave silently" % BONJOUR_WAV)
    reset()


def start():
    global __isRunning
    __isRunning = True
    print("ShakeHand Start")


def stop():
    global __isRunning
    __isRunning = False
    reset()
    print("ShakeHand Stop")


def exit():
    global __isRunning
    __isRunning = False
    AGC.runActionGroup("stand_slow")
    print("ShakeHand Exit")

# ----------------------------------------------------------------- tracking
def track(center_x, center_y, img_w, img_h):
    """Move the head so the face drifts toward the image centre."""
    if abs(center_x - img_w / 2.0) < CENTER_TOLERANCE:
        center_x = img_w / 2.0
    if abs(center_y - img_h / 2.0) < CENTER_TOLERANCE:
        center_y = img_h / 2.0

    x_pid.SetPoint = img_w / 2
    x_pid.update(center_x)
    dx = int(x_pid.output)

    y_pid.SetPoint = img_h / 2
    y_pid.update(center_y)
    dy = int(y_pid.output)

    if dx == 0 and dy == 0:
        return
    use_time = max(abs(dx), abs(dy)) * 0.00025
    set_head(pan_pulse + dx, tilt_pulse + dy, use_time * 1000)
    time.sleep(use_time)


def run(img):
    global stable_count, miss_count, face_seen, face_lost_at, armed, greeting

    if not __isRunning:
        return img

    img_h, img_w = img.shape[:2]
    bbox = largest_face(find_faces(img))

    if bbox is None:
        miss_count += 1
        stable_count = 0
        if face_seen and miss_count > LOST_FRAMES:
            face_seen = False                       # scan thread takes over
            face_lost_at = time.time()
            x_pid.clear()
            y_pid.clear()
        if not face_seen and not armed and time.time() - face_lost_at > LOST_RESET_TIME:
            armed = True
        return img

    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
    miss_count = 0
    if not face_seen:
        face_seen = True                            # scan thread pauses
        x_pid.clear()
        y_pid.clear()

    if greeting:
        return img                                  # hold the head still while waving

    track(cx, cy, img_w, img_h)

    if abs(cx - img_w / 2.0) < img_w * GREET_ZONE:
        stable_count += 1
    else:
        stable_count = 0

    ready = (armed and stable_count >= STABLE_FRAMES
             and time.time() - last_greet_at > GREET_COOLDOWN)
    if ready:
        armed = False
        stable_count = 0
        greeting = True
        threading.Thread(target=greet, daemon=True).start()
        cv2.putText(img, "Bonjour!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return img

# ----------------------------------------------------------------- standalone
if __name__ == '__main__':
    init()
    start()

    open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')['open_once']
    if open_once:
        # the tonypi service owns the camera; read its MJPEG stream instead
        my_camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
    else:
        my_camera = Camera.Camera()
        my_camera.camera_open()

    show = bool(os.environ.get("DISPLAY"))
    AGC.runActionGroup("stand")
    print("ShakeHand running, Ctrl+C to quit")
    frames, t0 = 0, time.time()
    try:
        while True:
            ret, img = my_camera.read()
            if ret and img is not None:
                frame = run(img.copy())
                frames += 1
                if time.time() - t0 > 5:
                    print("ShakeHand: %.1f fps, face=%s, armed=%s" % (frames / (time.time() - t0), face_seen, armed))
                    frames, t0 = 0, time.time()
                if show:
                    cv2.imshow('ShakeHand', frame)
                    if cv2.waitKey(1) == 27:
                        break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        exit()
        if open_once:
            my_camera.release()
        else:
            my_camera.camera_close()
        if show:
            cv2.destroyAllWindows()
