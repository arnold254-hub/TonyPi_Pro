#!/usr/bin/python3
# coding=utf8
"""
Arms, grippers, head and stance of the TonyPi for chess.

Nothing here assumes the length of the robot's arms or where the board
stands. Every position is TAUGHT with chess_teach.py: the arm is put over a
square by hand (or jogged there), the servo positions are read back and
stored in chess_arm.yaml. Squares that were not taught themselves are
interpolated from the ones around them; squares outside the taught area are
"out of reach" and the program asks the player to move that piece instead.

Taught per arm
    points   a handful of squares, each with two poses:
               grasp  gripper around the piece, ready to close
               hover  straight above it, high enough to carry a piece over
                      every other piece
    path     way-points from the rest pose to a spot above the board, so
             the hand swings clear of the pieces on its way in and out
    bin      pose above the box that collects captured pieces
    open / grip / relax   gripper positions: wide open, holding a piece,
             and closed without squeezing (used while the arm rests)

Poses are tied to places on the table, not to square names: the taught
points are stored in the frame of the board outline the camera saw while
teaching. When the board is turned round for the other colour, or has been
nudged, the square centres are looked up where the camera sees them now.

Motion rules
    * one arm at a time, the other one rests
    * rest -> path -> hover -> grasp -> hover -> ... -> path -> rest
    * travelling speed and lowering speed are limited (pulses per second)
    * every target is clamped to the servo limits in the config
    * after a move the servos are read back; an arm that did not arrive
      stops the move (ArmError) and the arm backs out the way it came
    * the battery is checked before every move
"""
import os
import time
import copy
import queue
import struct
import logging
import subprocess

import numpy as np
import cv2
import chess

import chess_vision as vision

log = logging.getLogger("chess.arm")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARM_YAML = os.path.join(ROOT, "chess_arm.yaml")
ACTION_DIR = os.path.join(ROOT, "ActionGroups") + os.sep

_sleep = time.sleep             # replaced by the self-test so simulated games do not wait

ARM_NAMES = ("right", "left")
JOINT_NAMES = ("shoulder", "upper arm", "elbow")

DEFAULT_CONFIG = {
    "version": 1,
    "stance_action": "stand_slow",      # action group that puts the robot in its playing stance
    "head": {"tilt": 1000, "pan": 1500},
    "limp_value": 1,                    # bus_servo_enable_torque() argument that lets a servo go limp
    "speed": 350,                       # pulses per second while travelling
    "slow_speed": 150,                  # pulses per second while lowering onto / lifting off a square
    "settle": 0.25,                     # seconds to wait after every move
    "arrive_tolerance": 35,             # pulses; 0 switches the arrival check off
    "battery_warn_mv": 10500,
    "battery_min_mv": 9800,
    "allow_relay": True,                # hand a piece from one arm to the other over a free square
    "max_drift": 0.75,                  # squares the board may be away from where it was taught
    "reach_margin": 0.30,               # squares outside the taught area that still count as reachable
    "grasp_lift": {"P": 0.0, "N": 0.0, "B": 0.0, "R": 0.0, "Q": 0.0, "K": 0.0},
    "board": None,                      # {"corners": [[x, y] * 4], "image_size": [w, h]} while teaching
    "arms": {
        "right": {
            "joints": [16, 15, 14], "gripper": 17, "open": 760, "grip": 500, "relax": 500,
            "grip_by_piece": {},
            "grip_empty": None, "grip_held": None,
            "rest": [275, 200, 425], "limits": [[0, 1000], [0, 1000], [0, 1000]],
            "gripper_limits": [400, 800],
            "path": [], "bin": None, "points": [],
        },
        "left": {
            "joints": [8, 7, 6], "gripper": 18, "open": 240, "grip": 500, "relax": 500,
            "grip_by_piece": {},
            "grip_empty": None, "grip_held": None,
            "rest": [725, 800, 575], "limits": [[0, 1000], [0, 1000], [0, 1000]],
            "gripper_limits": [200, 600],
            "path": [], "bin": None, "points": [],
        },
    },
}


class ArmError(RuntimeError):
    """The arm could not do what was asked; it has stopped."""


class GraspError(ArmError):
    """The gripper closed on nothing."""


# ---------------------------------------------------------------- configuration
def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path=ARM_YAML):
    """Read chess_arm.yaml merged over the defaults. Returns None if there is no file."""
    import yaml
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    cfg = _merge(DEFAULT_CONFIG, data)
    problems = validate_config(cfg)
    if problems:
        raise ArmError("%s is not usable: %s" % (path, "; ".join(problems)))
    return cfg


def save_config(cfg, path=ARM_YAML):
    import yaml
    problems = validate_config(cfg)
    if problems:
        raise ArmError("refusing to save a broken configuration: " + "; ".join(problems))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        yaml.safe_dump(_plain(cfg), fp, default_flow_style=None, sort_keys=False)
    os.replace(tmp, path)


def _plain(v):
    """numpy -> plain Python so the yaml stays readable."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _is_pose(p, n=3):
    try:
        return len(p) == n and all(0 <= float(x) <= 1000 for x in p)
    except TypeError:
        return False


def validate_config(cfg):
    """Return a list of problems (empty when the configuration can be used)."""
    bad = []
    for key in ("speed", "slow_speed"):
        if not 20 <= float(cfg.get(key, 0)) <= 1500:
            bad.append("%s must be between 20 and 1500 pulses per second" % key)
    head = cfg.get("head") or {}
    for key in ("tilt", "pan"):
        if not 500 <= float(head.get(key, 0)) <= 2500:
            bad.append("head %s must be between 500 and 2500" % key)
    used = set()
    for name in ARM_NAMES:
        arm = (cfg.get("arms") or {}).get(name)
        if arm is None:
            bad.append("arm '%s' is missing" % name)
            continue
        ids = list(arm.get("joints") or []) + [arm.get("gripper")]
        if len(ids) != 4 or any(not isinstance(i, int) or not 1 <= i <= 253 for i in ids):
            bad.append("%s arm: needs 3 joint servo ids and a gripper id" % name)
            continue
        if used & set(ids) or len(set(ids)) != 4:
            bad.append("%s arm: servo ids are used twice" % name)
        used |= set(ids)
        if not _is_pose(arm.get("rest")):
            bad.append("%s arm: rest pose must be 3 values in 0..1000" % name)
        lim = arm.get("limits") or []
        if len(lim) != 3 or any(len(l) != 2 or not 0 <= l[0] < l[1] <= 1000 for l in lim):
            bad.append("%s arm: limits must be 3 pairs [low, high] within 0..1000" % name)
        glim = arm.get("gripper_limits") or [0, 1000]
        if len(glim) != 2 or not 0 <= glim[0] < glim[1] <= 1000:
            bad.append("%s arm: gripper_limits must be [low, high] within 0..1000" % name)
            glim = [0, 1000]
        for key in ("open", "grip", "relax"):
            if not glim[0] <= float(arm.get(key, -1)) <= glim[1]:
                bad.append("%s arm: gripper '%s' must be within gripper_limits %s" % (name, key, list(glim)))
        for sym, val in (arm.get("grip_by_piece") or {}).items():
            if sym not in "PNBRQK" or not glim[0] <= float(val) <= glim[1]:
                bad.append("%s arm: grip_by_piece[%s] is not usable" % (name, sym))
        for i, wp in enumerate(arm.get("path") or []):
            if not _is_pose(wp):
                bad.append("%s arm: way-point %d is not a pose" % (name, i + 1))
        if arm.get("bin") is not None and not _is_pose(arm["bin"]):
            bad.append("%s arm: bin is not a pose" % name)
        for i, pt in enumerate(arm.get("points") or []):
            if (not isinstance(pt, dict) or not _is_pose(pt.get("grasp")) or not _is_pose(pt.get("hover"))
                    or len(pt.get("uv") or []) != 2):
                bad.append("%s arm: taught point %d is incomplete" % (name, i + 1))
    if any((cfg["arms"].get(n) or {}).get("points") for n in ARM_NAMES if n in (cfg.get("arms") or {})):
        b = cfg.get("board") or {}
        if len(b.get("corners") or []) != 4:
            bad.append("squares were taught but the board outline at teaching time is missing")
    return bad


def is_taught(cfg):
    """True when at least one arm knows at least one square."""
    return bool(cfg) and any(cfg["arms"][n]["points"] for n in ARM_NAMES)


# ---------------------------------------------------------------- interpolation between taught poses
class PoseField:
    """
    Smooth map from a place on the board (u, v) to servo positions, passing
    exactly through the taught points (thin-plate spline). With one taught
    point only that point is known; with points on a line, the line.
    """

    def __init__(self, uv, values, margin=0.3):
        self.uv = np.array(uv, np.float64).reshape(-1, 2)
        self.values = np.array(values, np.float64).reshape(len(self.uv), -1)
        self.margin = float(margin)
        n = len(self.uv)
        if n == 0:
            raise ValueError("no taught points")
        self.mode = "tps"
        self.hull = None
        if n == 1:
            self.mode = "single"
            return
        centred = self.uv - self.uv.mean(axis=0)
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        if n == 2 or s[1] < 1e-6 * max(s[0], 1e-9) or s[1] < 0.05:
            self.mode = "line"
            self.axis = vt[0]
            self.origin = self.uv.mean(axis=0)
            t = centred @ self.axis
            order = np.argsort(t)
            self.t = t[order]
            self.line_values = self.values[order]
            return
        self.hull = cv2.convexHull(self.uv.astype(np.float32)).reshape(-1, 2)
        K = self._kernel(np.linalg.norm(self.uv[:, None] - self.uv[None], axis=2))
        P = np.column_stack([np.ones(n), self.uv])
        A = np.zeros((n + 3, n + 3))
        A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
        rhs = np.zeros((n + 3, self.values.shape[1]))
        rhs[:n] = self.values
        try:
            self.coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.lstsq(A, rhs, rcond=None)[0]

    @staticmethod
    def _kernel(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 1e-12, r * r * np.log(r), 0.0)

    def distance_outside(self, u, v):
        """How far (u, v) lies outside the taught area, in squares (0 when inside)."""
        p = np.array([u, v], np.float64)
        if self.mode == "single":
            return float(np.linalg.norm(p - self.uv[0]))
        if self.mode == "line":
            t = float((p - self.origin) @ self.axis)
            tc = min(max(t, self.t[0]), self.t[-1])
            return float(np.linalg.norm(p - (self.origin + tc * self.axis)))
        d = cv2.pointPolygonTest(self.hull.astype(np.float32), (float(u), float(v)), True)
        return float(max(0.0, -d))

    def reachable(self, u, v):
        return self.distance_outside(u, v) <= self.margin

    def __call__(self, u, v):
        p = np.array([u, v], np.float64)
        if self.mode == "single":
            return self.values[0].copy()
        if self.mode == "line":
            t = float((p - self.origin) @ self.axis)
            return np.array([np.interp(t, self.t, self.line_values[:, k])
                             for k in range(self.values.shape[1])])
        n = len(self.uv)
        k = self._kernel(np.linalg.norm(self.uv - p, axis=1))
        return k @ self.coef[:n] + np.array([1.0, p[0], p[1]]) @ self.coef[n:]


def leave_one_out(points):
    """
    Interpolation check: for every taught point that lies BETWEEN other
    taught points, how far (in pulses) the pose predicted from the others
    is from the taught one. Points on the rim of the taught area are left
    out: without them the area is smaller and they would be guessed from
    outside, which never happens in a game.
    Returns a list of (name, worst joint error for grasp, for hover).
    """
    out = []
    if len(points) < 4:
        return out
    for i, pt in enumerate(points):
        rest = points[:i] + points[i + 1:]
        uv = [p["uv"] for p in rest]
        row = [pt.get("name", "?")]
        for kind in ("grasp", "hover"):
            field = PoseField(uv, [p[kind] for p in rest], margin=99)
            if field.distance_outside(*pt["uv"]) > 0.05:
                break
            row.append(float(np.abs(field(*pt["uv"]) - np.array(pt[kind], float)).max()))
        if len(row) == 3:
            out.append(tuple(row))
    return out


# ---------------------------------------------------------------- servo bus
class HiwonderBus:
    """The robot's controller board. All reads time out instead of blocking for ever."""

    PORT_WAIT = 12.0            # seconds a command waits for a closed serial port to come back

    def __init__(self, limp_value=1):
        import hiwonder.ActionGroupControl as AGC
        import hiwonder.ros_robot_controller_sdk as rrc
        self.rrc = rrc
        self.AGC = AGC
        self.board = AGC.board                 # share the port the action groups use
        self.port_waits = 0
        self._raw_write = getattr(self.board, "_chess_raw_write", None) or self.board.buf_write
        self.board._chess_raw_write = self._raw_write
        self.board.buf_write = self._patient_write      # every command of the SDK goes through it
        self.board.enable_reception(True)
        self.limp_value = int(limp_value)
        self.read_failures = 0
        _sleep(0.3)

    def _patient_write(self, func, data):
        """
        The SDK's reader closes the serial port for a second, then opens it again, whenever
        it has heard nothing for five seconds or a read fails (the tonypi service reading the
        same port does that). A command sent in that second is refused with PortNotOpenError.
        Wait for the port instead.
        """
        deadline = time.time() + self.PORT_WAIT
        waited = False
        while True:
            try:
                return self._raw_write(func, data)
            except (OSError, TypeError, ValueError) as e:       # pyserial's errors are OSErrors
                if time.time() >= deadline:
                    raise ArmError("the controller board does not take commands (%s). Is the tonypi service "
                                   "using the serial port? Stop it with:  sudo systemctl stop tonypi"
                                   % (str(e) or type(e).__name__))
                if not waited:
                    waited = True
                    self.port_waits += 1
                    log.info("serial port closed (%s); waiting for it (time number %d)", e, self.port_waits)
                time.sleep(0.05)

    # -- bus servos
    def move(self, positions, ms):
        """positions: {servo id: pulse}; all servos start together and take `ms` milliseconds."""
        pairs = [[int(i), int(round(p))] for i, p in positions.items()]
        self.board.bus_servo_set_position(max(0.02, ms / 1000.0), pairs)

    def _query(self, servo_id, cmd, fmt, timeout=0.25):
        b = self.board
        func = self.rrc.PacketFunction.PACKET_FUNC_BUS_SERVO
        with b.servo_read_lock:
            try:                                # drop an old answer nobody collected
                b.bus_servo_queue.get_nowait()
            except queue.Empty:
                pass
            b.buf_write(func, [cmd, servo_id])
            try:
                data = b.bus_servo_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        try:
            sid, rcmd, status, *info = struct.unpack(fmt, data)
        except struct.error:
            return None
        if sid != servo_id or rcmd != cmd or status != 0:
            return None
        return info

    def read(self, servo_id, tries=3):
        """Current position of a servo, or None when it does not answer."""
        values = []
        for _ in range(tries * 2):
            info = self._query(servo_id, 0x05, "<BBbh")
            if info is not None and -50 <= info[0] <= 1050:
                values.append(int(info[0]))
                if len(values) >= tries:
                    break
        if not values:
            self.read_failures += 1
            return None
        return int(np.median(values))

    def torque(self, servo_id, on):
        value = (1 - self.limp_value) if on else self.limp_value
        self.board.bus_servo_enable_torque(int(servo_id), value)

    def servo_voltage_mv(self, servo_id):
        info = self._query(servo_id, 0x07, "<BBbH")
        return None if info is None else int(info[0])

    def servo_temperature(self, servo_id):
        info = self._query(servo_id, 0x09, "<BBbB")
        return None if info is None else int(info[0])

    # -- everything else
    def head(self, tilt, pan, ms=600):
        self.board.pwm_servo_set_position(ms / 1000.0, [[1, int(tilt)], [2, int(pan)]])

    def battery_mv(self):
        for _ in range(30):                     # the board reports about once a second
            mv = self.board.get_battery()
            if mv is not None and 5000 < mv < 20000:
                return int(mv)
            _sleep(0.05)
        return None

    def beep(self, freq=1900, on=0.1, repeat=1):
        try:
            self.board.set_buzzer(int(freq), on, 0.1, repeat)
        except Exception:
            pass

    def run_action(self, name):
        if not os.path.isfile(os.path.join(ACTION_DIR, name + ".d6a")):
            raise ArmError("action group %s.d6a not found in %s" % (name, ACTION_DIR))
        self.AGC.runActionGroup(name, path=ACTION_DIR)

    def close(self):
        pass


class NullBus:
    """Stands in for the robot on a PC: accepts every command, remembers positions."""

    def __init__(self):
        self.pos = {}
        self.limp = set()
        self.read_failures = 0

    def move(self, positions, ms):
        for i, p in positions.items():
            self.pos[int(i)] = int(round(p))
            self.limp.discard(int(i))

    def read(self, servo_id, tries=3):
        return self.pos.get(int(servo_id))

    def torque(self, servo_id, on):
        (self.limp.discard if on else self.limp.add)(int(servo_id))

    def servo_voltage_mv(self, servo_id):
        return None

    def servo_temperature(self, servo_id):
        return None

    def head(self, tilt, pan, ms=600):
        pass

    def battery_mv(self):
        return None

    def beep(self, freq=1900, on=0.1, repeat=1):
        pass

    def run_action(self, name):
        pass

    def close(self):
        pass


def stop_service(name="tonypi"):
    """Stop the robot's app service. True when it is not running afterwards."""
    try:
        subprocess.run(["sudo", "-n", "systemctl", "stop", name], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log.warning("could not stop the %s service: %s", name, e)
        return False
    time.sleep(1.5)                     # let go of the serial port and the camera
    return not service_running(name)


def service_running(name="tonypi"):
    """True when the robot's app service is running (it shares the serial port and the camera)."""
    try:
        r = subprocess.run(["systemctl", "is-active", "--quiet", name], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------- one arm
class Arm:
    def __init__(self, name, cfg, bus):
        self.name = name
        self.cfg = cfg
        self.arm = cfg["arms"][name]
        self.bus = bus
        self.joints = [int(i) for i in self.arm["joints"]]
        self.gripper = int(self.arm["gripper"])
        self.limits = [(float(a), float(b)) for a, b in self.arm["limits"]]
        self.rest = np.array(self.arm["rest"], np.float64)
        self.path = [np.array(p, np.float64) for p in self.arm["path"]]
        self.bin = None if self.arm["bin"] is None else np.array(self.arm["bin"], np.float64)
        self.pose = None                # last commanded joint pose
        self.where = "unknown"          # rest | board | bin | unknown
        self.holding = None             # piece type in the gripper
        self.gripper_open = None
        self.is_limp = False
        self._last_hover = None
        self._warned_read = False
        self.rebuild()

    def rebuild(self):
        """(Re)build the interpolation from the taught points."""
        pts = self.arm["points"]
        self.fields = None
        if pts:
            uv = [p["uv"] for p in pts]
            margin = float(self.cfg["reach_margin"])
            self.fields = {kind: PoseField(uv, [p[kind] for p in pts], margin) for kind in ("grasp", "hover")}

    # -- what the arm knows
    def reachable(self, uv):
        return self.fields is not None and self.fields["grasp"].reachable(*uv)

    def target(self, uv, kind, lift=0.0):
        """Joint pose for a place on the board. kind: 'grasp' or 'hover'."""
        if self.fields is None:
            raise ArmError("the %s arm has not been taught any square" % self.name)
        pose = self.fields[kind](*uv)
        if kind == "grasp" and lift:
            pose = pose + float(lift) * (self.fields["hover"](*uv) - pose)
        return self.clamp(pose)

    def clamp(self, pose):
        return np.array([min(max(float(p), lo), hi) for p, (lo, hi) in zip(pose, self.limits)])

    def lift_for(self, piece_type):
        if piece_type is None:
            return 0.0
        return float((self.cfg.get("grasp_lift") or {}).get(chess.piece_symbol(piece_type).upper(), 0.0))

    # -- motion
    def sync(self):
        """Find out where the arm really is (after an action group, or at start-up)."""
        read = [self.bus.read(i) for i in self.joints]
        if all(r is not None for r in read):
            self.pose = np.array(read, np.float64)
            self.where = "rest" if np.abs(self.pose - self.rest).max() <= 40 else "unknown"
        else:
            self.pose = None
            self.where = "unknown"
        return self.pose

    def move(self, pose, speed=None, check=True):
        pose = self.clamp(pose)
        speed = float(speed or self.cfg["speed"])
        if self.pose is None:
            seconds = 1.5                       # unknown start: take it slowly
        else:
            seconds = max(0.25, float(np.abs(pose - self.pose).max()) / speed)
        self.bus.move({i: p for i, p in zip(self.joints, pose)}, int(seconds * 1000))
        _sleep(seconds + float(self.cfg["settle"]))
        self.pose = pose
        self.is_limp = False
        if check:
            self._check_arrival(pose)

    def _check_arrival(self, pose):
        tol = float(self.cfg["arrive_tolerance"])
        if tol <= 0:
            return
        for attempt in range(3):
            read = [self.bus.read(i) for i in self.joints]
            if any(r is None for r in read):
                if not self._warned_read:
                    log.warning("%s arm: servos do not answer; cannot check that moves arrive "
                                "(is the tonypi service using the port?)", self.name)
                    self._warned_read = True
                return
            off = [(i, r, p) for i, r, p in zip(self.joints, read, pose) if abs(r - p) > tol]
            if not off:
                return
            _sleep(0.3)
        i, r, p = off[0]
        raise ArmError("%s arm: servo %d stopped at %d instead of %d (blocked?)" % (self.name, i, r, round(p)))

    def set_gripper(self, pulse, wait=0.45):
        lo, hi = self.arm.get("gripper_limits") or (0, 1000)
        pulse = min(max(float(pulse), lo), hi)
        self.bus.move({self.gripper: pulse}, 300)
        _sleep(wait)
        return pulse

    def open(self):
        self.set_gripper(self.arm["open"])
        self.gripper_open = True

    def relax(self):
        """Close without squeezing; the position the gripper keeps while the arm rests."""
        self.set_gripper(self.arm["relax"], wait=0.3)
        self.gripper_open = False

    def close(self, piece_type=None):
        """Close on a piece. Raises GraspError when the gripper can tell it closed on nothing."""
        grip = self.arm["grip"]
        if piece_type is not None:
            grip = (self.arm.get("grip_by_piece") or {}).get(chess.piece_symbol(piece_type).upper(), grip)
        self.set_gripper(grip, wait=0.6)
        self.gripper_open = False
        empty, held = self.arm.get("grip_empty"), self.arm.get("grip_held")
        if empty is None or held is None or abs(empty - held) < 8:
            return                              # the gripper cannot feel the piece; the camera will check
        read = self.bus.read(self.gripper)
        if read is not None and abs(read - empty) < abs(read - held):
            raise GraspError("%s gripper closed on nothing (reads %d)" % (self.name, read))

    def to_rest(self):
        """Back to the rest pose along the taught path, wherever the arm is."""
        if self.where == "rest":
            return
        for wp in reversed(self.path):
            self.move(wp)
        self.move(self.rest)
        self.where = "rest"
        self._last_hover = None
        if self.holding is None and self.gripper_open:
            self.relax()

    def to_board(self):
        """From the rest pose along the taught path to above the board."""
        if self.where in ("board", "bin"):
            return
        if self.where != "rest":
            self.to_rest()
        for wp in self.path:
            self.move(wp)
        self.where = "board"

    def pick(self, uv, piece_type=None):
        lift = self.lift_for(piece_type)
        hover, grasp = self.target(uv, "hover"), self.target(uv, "grasp", lift)
        self.to_board()
        self.open()
        self.move(hover)
        self._last_hover = hover
        self.move(grasp, self.cfg["slow_speed"])
        try:
            self.close(piece_type)
        except GraspError:
            self.open()
            self.move(hover, self.cfg["slow_speed"])
            raise
        self.holding = piece_type if piece_type is not None else chess.PAWN
        self.move(hover, self.cfg["slow_speed"])

    def place(self, uv, piece_type=None):
        lift = self.lift_for(piece_type)
        hover, grasp = self.target(uv, "hover"), self.target(uv, "grasp", lift)
        self.to_board()
        self.move(hover)
        self._last_hover = hover
        self.move(grasp, self.cfg["slow_speed"])
        self.open()
        self.holding = None
        self.move(hover, self.cfg["slow_speed"])

    def drop_in_bin(self):
        if self.bin is None:
            raise ArmError("the %s arm has no bin" % self.name)
        if self.path:
            self.move(self.path[-1])
        self.move(self.bin)
        self.where = "bin"
        self.open()
        self.holding = None
        if self.path:
            self.move(self.path[-1])
        self.where = "board"
        self._last_hover = None

    def recover(self):
        """After a failure: back out the way the arm came in. The gripper is left as it is."""
        try:
            if self._last_hover is not None:
                self.move(self._last_hover, self.cfg["slow_speed"], check=False)
            for wp in reversed(self.path):
                self.move(wp, check=False)
            self.move(self.rest, check=False)
            self.where = "rest"
            self._last_hover = None
        except Exception as e:                  # never let the clean-up hide the first error
            log.error("%s arm: could not back out: %s", self.name, e)

    def limp(self):
        for i in self.joints:
            self.bus.torque(i, False)
        self.pose = None
        self.where = "unknown"
        self.is_limp = True

    def hold(self):
        """Switch the servos back on, holding the arm where it is now."""
        read = [self.bus.read(i) for i in self.joints]
        for i in self.joints:
            self.bus.torque(i, True)
        if all(r is not None for r in read):
            self.bus.move({i: r for i, r in zip(self.joints, read)}, 200)
            self.pose = self.clamp(read)
        self.is_limp = False
        return read


# ---------------------------------------------------------------- planning
class Step:
    """One piece carried from one place to another."""

    def __init__(self, kind, arm, src, dst=None, piece_type=None):
        self.kind = kind                # "move" (square to square) or "bin" (square to the box)
        self.arm = arm
        self.src = src
        self.dst = dst
        self.piece_type = piece_type

    def __repr__(self):
        piece = chess.piece_name(self.piece_type) if self.piece_type else "piece"
        if self.kind == "bin":
            return "%s arm: %s %s -> box" % (self.arm, piece, chess.square_name(self.src))
        return "%s arm: %s %s -> %s" % (self.arm, piece, chess.square_name(self.src),
                                        chess.square_name(self.dst))


class Plan:
    def __init__(self, move, steps, ask=None):
        self.move = move
        self.steps = steps
        self.ask = ask                  # something the player has to do afterwards (promotion)
        self.done = 0                   # steps completed

    def __repr__(self):
        return "; ".join(repr(s) for s in self.steps)


class Hands:
    """Both arms, the head and the stance, plus the planner that turns a chess move into steps."""

    def __init__(self, cfg, bus):
        self.cfg = cfg
        self.bus = bus
        self.arms = {n: Arm(n, cfg, bus) for n in ARM_NAMES}
        self.to_teach = None            # oriented board units -> taught (u, v); None = board not located
        self.drift = None
        self.why_off = "the board has not been located yet"
        self._battery_warned = False

    # -- body
    def stance(self):
        """Stand in the playing stance, arms at rest, looking at the board."""
        self.bus.run_action(self.cfg["stance_action"])
        _sleep(0.3)
        for arm in self.arms.values():
            arm.sync()
            arm.move(arm.rest, check=False)
            arm.where = "rest"
            arm.relax()
        self.look()

    def look(self):
        head = self.cfg["head"]
        self.bus.head(head["tilt"], head["pan"], 600)
        _sleep(0.8)

    def rest(self):
        for arm in self.arms.values():
            if arm.where != "rest":
                arm.to_rest()

    def at_rest(self):
        return all(a.where == "rest" for a in self.arms.values())

    def holding(self):
        """(arm name, piece type) for a piece left in a gripper, or None."""
        for name, arm in self.arms.items():
            if arm.holding is not None:
                return name, arm.holding
        return None

    def battery_ok(self):
        """False when the battery is too low to move the arms safely."""
        mv = self.bus.battery_mv()
        if mv is None:
            return True
        if mv < float(self.cfg["battery_min_mv"]):
            log.error("battery at %.1f V: too low to move the arms", mv / 1000.0)
            return False
        if mv < float(self.cfg["battery_warn_mv"]) and not self._battery_warned:
            log.warning("battery at %.1f V: charge it soon", mv / 1000.0)
            self._battery_warned = True
        return True

    # -- where the board is
    def set_board(self, oriented_corners):
        """
        Tell the arms where the camera sees the board now (corners in the
        order a8, h8, h1, a1). Returns how far the board is from where it
        stood while teaching, in squares, or None if nothing was taught.
        """
        taught = self.cfg.get("board")
        if not taught:
            self.to_teach, self.drift = None, None
            self.why_off = "the arms have not been taught (run chess_teach.py)"
            return None
        now = vision.board_matrix(oriented_corners)                     # oriented units -> image
        teach = np.linalg.inv(vision.board_matrix(taught["corners"]))   # image -> taught units
        self.to_teach = teach @ now
        # drift: how far the board outline is from the taught one, whatever way round it is
        here = vision.transform_points(teach, np.array(oriented_corners).reshape(4, 2))
        ref = vision._BOARD_DST.astype(np.float64)
        self.drift = min(float(np.linalg.norm(here - np.roll(ref, -k, axis=0), axis=1).max())
                         for k in range(4))
        if not is_taught(self.cfg):
            self.why_off = "the arms have not been taught any square (run chess_teach.py)"
        elif self.drift > float(self.cfg["max_drift"]):
            self.why_off = ("the board is %.1f squares away from where the arms were taught"
                            % self.drift)
        else:
            self.why_off = ""
        return self.drift

    def ready(self):
        return self.to_teach is not None and not self.why_off

    def square_uv(self, square):
        if self.to_teach is None:
            raise ArmError(self.why_off)
        return tuple(vision.transform_points(self.to_teach, [vision.square_centre(square)])[0])

    def who_reaches(self, square):
        if self.to_teach is None:
            return []
        uv = self.square_uv(square)
        return [n for n in ARM_NAMES if self.arms[n].reachable(uv)]

    def reach_map(self):
        """Text picture of the board: R, L, B(oth) or . for every square, rank 8 at the top."""
        rows = []
        for rank in range(7, -1, -1):
            row = []
            for f in range(8):
                who = self.who_reaches(chess.square(f, rank))
                row.append("B" if len(who) == 2 else (who[0][0].upper() if who else "."))
            rows.append("%d  %s" % (rank + 1, " ".join(row)))
        rows.append("   a b c d e f g h")
        return "\n".join(rows)

    # -- planning
    def _carry(self, src, dst, piece_type, occupied, prefer=None):
        """Steps that take a piece from src to dst, or (None, reason)."""
        a, b = self.who_reaches(src), self.who_reaches(dst)
        if not a:
            return None, "%s is out of reach" % chess.square_name(src)
        if not b:
            return None, "%s is out of reach" % chess.square_name(dst)
        both = [n for n in a if n in b]
        if both:
            arm = prefer if prefer in both else both[0]
            return [Step("move", arm, src, dst, piece_type)], ""
        if not self.cfg.get("allow_relay"):
            return None, "no arm reaches both %s and %s" % (chess.square_name(src), chess.square_name(dst))
        p, q = np.array(self.square_uv(src)), np.array(self.square_uv(dst))
        best = None
        for via in chess.SQUARES:
            if via in occupied or via in (src, dst):
                continue
            if len(self.who_reaches(via)) < 2:
                continue
            # keep clear of pieces standing next to the hand-over square
            crowd = sum(1 for s in occupied if s not in (src,) and chess.square_distance(s, via) == 1)
            m = np.array(self.square_uv(via))
            cost = np.linalg.norm(m - p) + np.linalg.norm(m - q) + 2.0 * crowd
            if best is None or cost < best[0]:
                best = (cost, via)
        if best is None:
            return None, "no free square both arms can reach to hand the piece over"
        via = best[1]
        return [Step("move", a[0], src, via, piece_type), Step("move", b[0], via, dst, piece_type)], ""

    def plan(self, board, move):
        """
        Work out how to play `move` on the real board. Returns (Plan, "") or
        (None, reason) when the player has to make the move for the robot.
        """
        if not self.ready():
            return None, self.why_off
        piece_type = board.piece_type_at(move.from_square)
        occupied = set(board.piece_map())
        steps = []
        prefer = None
        if board.is_capture(move):
            victim = move.to_square
            if board.is_en_passant(move):
                victim = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            able = [n for n in self.who_reaches(victim) if self.arms[n].bin is not None]
            if not able:
                why = "out of reach" if not self.who_reaches(victim) else "that arm has no box for captured pieces"
                return None, "the piece on %s: %s" % (chess.square_name(victim), why)
            both = [n for n in able if n in self.who_reaches(move.from_square)
                    and n in self.who_reaches(move.to_square)]
            prefer = both[0] if both else able[0]
            steps.append(Step("bin", prefer, victim, None, board.piece_type_at(victim)))
            occupied.discard(victim)
        carry, why = self._carry(move.from_square, move.to_square, piece_type, occupied, prefer)
        if carry is None:
            return None, why
        steps += carry
        if board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            occupied.discard(move.from_square)
            occupied.add(move.to_square)
            carry, why = self._carry(rook_from, rook_to, chess.ROOK, occupied, steps[-1].arm)
            if carry is None:
                return None, why
            steps += carry
        ask = None
        if move.promotion:
            ask = ("promotion", move.to_square, move.promotion)
        return Plan(move, steps, ask), ""

    # -- doing it
    def execute(self, plan, on_step=None):
        """
        Carry out a plan. Raises ArmError if something goes wrong; by then
        the arms have backed out and plan.done says how far it got.
        """
        if not self.battery_ok():
            raise ArmError("the battery is too low to move the arms")
        current = None
        try:
            for step in plan.steps:
                if on_step:
                    on_step(step)
                log.info("step: %r", step)
                arm = self.arms[step.arm]
                if current is not None and current is not arm:
                    current.to_rest()
                current = arm
                arm.pick(self.square_uv(step.src), step.piece_type)
                if step.kind == "bin":
                    arm.drop_in_bin()
                else:
                    arm.place(self.square_uv(step.dst), step.piece_type)
                plan.done += 1
            self.rest()
        except (KeyboardInterrupt, Exception) as e:
            log.error("arm move failed at step %d of %d: %s", plan.done + 1, len(plan.steps), e)
            for arm in self.arms.values():
                if arm.where != "rest":
                    arm.recover()
            if isinstance(e, (ArmError, KeyboardInterrupt)):
                raise
            raise ArmError(str(e))

    def release(self):
        """Open every gripper (used to hand a piece back to the player)."""
        for arm in self.arms.values():
            arm.open()
            arm.holding = None
            arm.relax()

    def shutdown(self):
        """Arms to rest; never raises."""
        for arm in self.arms.values():
            try:
                if arm.where != "rest":
                    arm.recover()
            except Exception as e:
                log.error("shutdown: %s", e)
