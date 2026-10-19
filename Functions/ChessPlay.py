#!/usr/bin/python3
# coding=utf8
"""
ChessPlay: TonyPi plays chess against a person and moves its own pieces.

How a game goes
    1. The robot takes its playing stance, looks down at the board and
       checks that the board is where it expects it.
    2. Setup (first run only): show the camera the EMPTY board, then the
       full STARTING position. The robot learns where the board is, which
       way round it sits and what an occupied / white / black square looks
       like with your pieces and lighting. Nothing about the pieces' shape
       is assumed, so any set works.
    3. You make a move on the board and press Enter (with --auto the robot
       notices by itself when you have moved).
    4. The robot photographs the board, works out which legal move you
       made, thinks, announces its reply, picks its piece up and puts it
       down on the new square. Then it looks again to check that the board
       shows what it should.

The arms have to be taught first, once, with chess_teach.py. Pieces the
arms cannot reach (and every piece, if the arms were never taught or you
pass --no-arms) are moved by you: the robot says its move, you make it and
press Enter, and the robot checks the board.

At the prompt you can also type a move (e2e4 or Nf3) if the camera got it
wrong, 'u' to undo the last pair of moves, or 'q' to quit.

Run on the robot as the pi user (no sudo), with the app service stopped:
    sudo systemctl stop tonypi
    cd /home/pi/TonyPi/Functions && python3 ChessPlay.py
Options:
    --white          robot plays white and moves first
    --time 2.0       thinking time per move in seconds
    --skill 10       Stockfish skill 0-20 (also picks built-in depth)
    --engine auto    auto = Stockfish if installed else built-in search;
                     stockfish = require Stockfish; builtin = never use it
    --stockfish P    path to a Stockfish binary (default: search PATH and
                     /usr/games, or the STOCKFISH_PATH environment variable)
    --recalibrate    redo the board / empty-board setup
    --no-arms        the robot only announces its moves; you move its pieces
    --check-arms     before the game, hold a hand over a square of each arm
                     so you can see that the arms are still on target
    --auto           notice the player's move without waiting for Enter
    --resume         carry on with the game that was interrupted
    --no-robot       test on a laptop: no servos, use --camera N or a URL
    --no-camera      no board reading at all: type every move at the prompt
                     (handy for testing the engine on a PC)
    --tilt 1000      head tilt for this game (lower looks further down); the
                     arms stay down if this differs from what they were taught with
    --web            show what the camera sees in a browser: http://<robot>:8081/
Every game is saved as PGN in /home/pi/TonyPi/chess_games/ (after every
move, so nothing is lost if the battery runs out).
Requires python-chess, which comes with the project as the folder
Functions/chess (recommended as well: sudo apt install stockfish).
Run chess_setup.sh once to check that everything is there, and
chess_selftest.py to check the vision / engine / speech / arm code without a robot.
"""
import os
import sys
import time
import signal
import select
import logging
import argparse
import datetime
import logging.handlers

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "HiwonderSDK"))
sys.path.append(ROOT)

try:
    import chess
    import chess.pgn
except ImportError:
    print("python-chess is missing. It comes with the project as the folder Functions/chess;")
    print("copy that folder to the robot (or, with internet: python3 -m pip install chess).")
    sys.exit(1)

import chess_vision as vision
import chess_engine
import chess_speech as speak
import chess_arm

log = logging.getLogger("chess")

CONFIG_YAML = os.path.join(ROOT, "chess_config.yaml")
REFERENCE_PNG = os.path.join(ROOT, "chess_reference.png")
DEBUG_DIR = os.path.join(ROOT, "chess_debug")
GAMES_DIR = os.path.join(ROOT, "chess_games")
LENS_NPZ = os.path.join(HERE, "CameraCalibration", "calibration_param.npz")
STREAM_URL = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
UNFINISHED = "unfinished.pgn"

FRAMES_TO_AVERAGE = 5
WATCH_PERIOD = 0.4              # --auto: seconds between looks at the board
WATCH_STEADY = 3                # --auto: looks in a row that must agree before a move is read
WATCH_TRIES = 3                 # --auto: steady but unreadable boards before asking for help


# ----------------------------------------------------------------- logging
def setup_logging():
    """Everything goes to chess_debug/chess.log; warnings and errors also to the console."""
    root = logging.getLogger("chess")
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(os.path.join(DEBUG_DIR, "chess.log"),
                                                  maxBytes=1000000, backupCount=2, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(fh)
    except OSError as e:
        print("Cannot write the log file: %s" % e)


# ----------------------------------------------------------------- robot hardware (optional)
class Robot:
    """The body: stance, head, arms. With enabled=False nothing moves (PC testing)."""

    def __init__(self, enabled=True, arms=True, bus=None, tilt=None):
        self.enabled = enabled
        self.cfg = None
        self.why_no_arms = ""
        if bus is None:
            bus = chess_arm.HiwonderBus() if enabled else chess_arm.NullBus()
        self.bus = bus
        if not enabled:
            self.why_no_arms = "the robot is switched off (--no-robot)"
        else:
            try:
                self.cfg = chess_arm.load_config()
            except Exception as e:
                log.error("%s", e)
                self.why_no_arms = "chess_arm.yaml cannot be used: %s" % e
            else:
                if not chess_arm.is_taught(self.cfg):
                    self.why_no_arms = "the arms have not been taught yet (run chess_teach.py)"
                elif not arms:
                    self.why_no_arms = "--no-arms was given"
        cfg = self.cfg
        if cfg is None:
            cfg = chess_arm.default_config()
            if enabled:
                cfg["head"]["pan"] = self._pan_centre(cfg["head"]["pan"])
        if tilt is not None:
            tilt = int(min(max(tilt, 500), 2500))
            if tilt != cfg["head"]["tilt"] and not self.why_no_arms:
                # the arms find the squares through the camera; they were taught for one view only
                self.why_no_arms = "--tilt points the head differently from when the arms were taught"
            cfg["head"]["tilt"] = tilt
        if hasattr(bus, "limp_value"):
            bus.limp_value = int(cfg["limp_value"])
        self.hands = chess_arm.Hands(cfg, bus)
        self.use_arms = not self.why_no_arms

    @staticmethod
    def _pan_centre(default):
        try:
            import hiwonder.yaml_handle as yaml_handle
            data = yaml_handle.get_yaml_data(os.path.join(ROOT, "servo_config.yaml"))
            return int(data.get("servo2", default))
        except Exception:
            return default

    def stance(self):
        if self.enabled:
            self.hands.stance()

    def look_at_board(self):
        if self.enabled:
            self.hands.look()

    def gesture(self, name):
        """Run an action group, then return to the playing stance."""
        if not self.enabled:
            return
        try:
            self.bus.run_action(name)
            self.hands.stance()
        except Exception as e:
            log.warning("gesture %s failed: %s", name, e)

    def warn(self):
        if self.enabled:
            self.bus.beep(1900, 0.15)

    def shutdown(self):
        if self.enabled:
            self.hands.shutdown()


# ----------------------------------------------------------------- camera
class Cam:
    """
    The robot's camera (or a webcam / stream with `source`). Frames from the
    robot's camera are corrected for lens distortion; self.K is then the
    camera matrix of the corrected picture.
    """

    def __init__(self, source=None, use_robot=True):
        self.cap = None
        self.hw = None
        self.K = None
        self._maps = None
        self._size = None
        self.source = source
        self.use_robot = use_robot
        self._open()
        if source is None and use_robot:
            self._load_lens()

    def _open(self):
        if self.source is not None:
            src = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src)
        elif self.use_robot:
            # The app service owns the camera while it runs and republishes it as a stream.
            if chess_arm.service_running():
                self.cap = cv2.VideoCapture(STREAM_URL)
                if self._first_frame() is not None:
                    return
                self.cap.release()
                self.cap = None
            try:
                import hiwonder.Camera as Camera
                self.hw = Camera.Camera()
                self.hw.camera_open()
            except Exception as e:
                log.warning("hiwonder.Camera is not usable (%s); opening the camera directly", e)
                self.hw = None
                self.cap = cv2.VideoCapture(-1)
        else:
            self.cap = cv2.VideoCapture(0)

    def _first_frame(self, seconds=3.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            img = self._raw()
            if img is not None:
                return img
            time.sleep(0.05)
        return None

    def _load_lens(self):
        try:
            data = np.load(LENS_NPZ)
            self._lens = (np.array(data["mtx_array"], np.float64), np.array(data["dist_array"], np.float64))
        except Exception as e:
            log.warning("no lens calibration (%s); using the picture as it comes", e)
            self._lens = None

    def _undistort(self, img):
        lens = getattr(self, "_lens", None)
        if lens is None:
            return img
        size = (img.shape[1], img.shape[0])
        if self._size != size:
            mtx, dist = lens
            if size != (640, 480):                      # the calibration was made at 640 x 480
                mtx = mtx.copy()
                mtx[0] *= size[0] / 640.0
                mtx[1] *= size[1] / 480.0
            new, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, size, 0, size)
            self._maps = cv2.initUndistortRectifyMap(mtx, dist, None, new, size, cv2.CV_16SC2)
            self.K = np.array(new, np.float64)
            self._size = size
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def _raw(self):
        if self.hw is not None:
            ret, img = self.hw.read()
            return img.copy() if ret and img is not None else None
        if self.cap is None:
            return None
        ret, img = self.cap.read()
        return img if ret else None

    def frame(self):
        img = self._raw()
        return None if img is None else self._undistort(img)

    def grab(self, n=FRAMES_TO_AVERAGE):
        """Median of several fresh frames: removes sensor noise and a hand passing by."""
        if self.cap is not None:                        # throw away what waited in the buffer
            for _ in range(4):
                self.cap.grab()
        frames = []
        deadline = time.time() + 6
        while len(frames) < n and time.time() < deadline:
            img = self._raw()
            if img is not None:
                frames.append(img)
            time.sleep(0.05)
        if not frames:
            raise RuntimeError("no image from the camera")
        img = frames[0] if len(frames) == 1 else np.median(np.stack(frames), axis=0).astype(np.uint8)
        return self._undistort(img)

    def close(self):
        if self.hw is not None:
            self.hw.camera_close()
        elif self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------- display / debug
SHOW = bool(os.environ.get("DISPLAY"))


def show(name, img):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, name + ".jpg"), img)
    except (OSError, cv2.error) as e:
        log.debug("could not save %s: %s", name, e)
    if SHOW:
        cv2.imshow(name, img)
        cv2.waitKey(1)


def serve_debug(port=8081):
    """Show the pictures in chess_debug to a browser on the network (http://<robot>:<port>/)."""
    import threading
    import http.server
    os.makedirs(DEBUG_DIR, exist_ok=True)
    page = ("<html><head><meta http-equiv='refresh' content='2'><title>TonyPi chess</title></head>"
            "<body style='background:#222;color:#ddd;font-family:sans-serif'>"
            + "".join("<div style='float:left;margin:6px'>%s<br><img src='%s.jpg' style='max-width:640px'></div>"
                      % (n, n) for n in ("camera", "last_read", "start_position", "empty_board"))
            + "</body></html>")
    with open(os.path.join(DEBUG_DIR, "index.html"), "w", encoding="utf-8") as fp:
        fp.write(page)
    folder = DEBUG_DIR

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=folder, **k)

        def log_message(self, *a):
            pass

    try:
        server = http.server.ThreadingHTTPServer(("", int(port)), Handler)
    except OSError as e:
        print("Cannot show the pictures on port %s: %s" % (port, e))
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Pictures of what I see: http://<address of the robot>:%d/" % int(port))
    return server


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def typed(timeout):
    """A line typed within `timeout` seconds, or None. Never blocks longer than that."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        time.sleep(timeout)
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    return "q" if line == "" else line.strip()


def yes(text):
    return text.lower() in ("", "y", "yes")


def click_corners(img):
    """Let the user click the 4 board corners in a window. Returns corners or None."""
    if not SHOW:
        return None
    pts = []
    view = img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow("click corners")
    cv2.setMouseCallback("click corners", on_mouse)
    print("Click the 4 corners of the board (any order). Press Esc to cancel.")
    while len(pts) < 4:
        cv2.imshow("click corners", view)
        if cv2.waitKey(30) == 27:
            cv2.destroyWindow("click corners")
            return None
    cv2.destroyWindow("click corners")
    return vision.order_corners(pts)


def type_corners():
    print("Open chess_debug/camera.jpg, note the pixel coordinates of the 4 board corners,")
    print("and type them as: x,y x,y x,y x,y   (or press Enter to retry auto-detection)")
    txt = ask("> ")
    if not txt:
        return None
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in txt.split()]
        assert len(pts) == 4 and all(len(p) == 2 for p in pts)
        return vision.order_corners(pts)
    except Exception:
        print("Could not parse that.")
        return None


# ----------------------------------------------------------------- calibration
def calibrate_board(cam):
    """Find the board corners and capture the empty-board reference."""
    while True:
        ask("\nSTEP 1  Put the EMPTY board in front of the robot, then press Enter > ")
        img = cam.grab()
        show("camera", img)
        corners = vision.find_board_corners(img)
        if corners is not None:
            print("Board found automatically.")
        else:
            print("Automatic detection failed.")
            corners = click_corners(img) or type_corners()
            if corners is None:
                continue
        h, w = img.shape[:2]
        inside = ((corners[:, 0] > 2) & (corners[:, 0] < w - 3) & (corners[:, 1] > 2) & (corners[:, 1] < h - 3))
        if not inside.all():
            print("Part of the board is outside the picture (see chess_debug/camera.jpg).")
            print("Move the board or tilt the head until the whole board is in view.")
            if not yes(ask("Try again? [Y/n] > ")):
                pass
            else:
                continue
        cal = vision.Calibration(corners, image_size=(w, h), K=getattr(cam, "K", None))
        vision.set_reference(cal, img)
        show("camera", vision.draw_corners(img, corners))
        show("empty_board", vision.draw_grid(cal.reference))
        print("Check chess_debug/empty_board.jpg: the 8x8 grid should line up with the squares.")
        if yes(ask("Is the grid aligned? [Y/n] > ")):
            cal.save(CONFIG_YAML, REFERENCE_PNG)
            print("Saved board calibration to %s" % CONFIG_YAML)
            return cal


def describe_offset(reg):
    """Where the board is compared with the calibration, in words, from the robot's side."""
    dx, dy = reg.offset
    parts = []
    if abs(dx) >= 0.05:
        parts.append("%.1f squares to my %s" % (abs(dx), "right" if dx > 0 else "left"))
    if abs(dy) >= 0.05:
        parts.append("%.1f squares %s" % (abs(dy), "nearer to me" if dy > 0 else "further from me"))
    if abs(reg.angle) >= 0.5:
        parts.append("turned %.0f degrees %s" % (abs(reg.angle), "clockwise" if reg.angle > 0 else "anticlockwise"))
    return ", ".join(parts) or "where I expect it"


def raw_offset(cal, reg):
    """reg.offset is in the oriented board; turn it so that +x is the robot's right, +y towards the robot."""
    dx, dy = reg.offset
    for _ in range(cal.rotation % 4):
        dx, dy = -dy, dx
    return dx, dy


def locate_board(cam, cal, expected_occ=None, quiet=False):
    """
    Check that the board is where the calibration says, following it if it
    was nudged. Returns the Registration of the last look, or None when the
    player chose to recalibrate.
    """
    while True:
        img = cam.grab()
        reg = vision.register(cal, img, expected_occ)
        if reg.ok and reg.shift <= vision.REG_APPLY_MAX:
            if reg.shift >= vision.REG_APPLY_MIN:
                reg.offset = raw_offset(cal, reg)
                vision.apply_registration(cal, reg)
                if not quiet:
                    print("The board is %s; following it." % describe_offset(reg))
                log.info("board followed: %r", reg)
            return reg
        show("camera", vision.draw_corners(img, cal.oriented_corners()))
        if reg.ok:
            reg.offset = raw_offset(cal, reg)
            print("The board is %s." % describe_offset(reg))
        else:
            print("I cannot find the board where I expect it (%s)." % reg.reason)
        print("chess_debug/camera.jpg shows where I expect the board (green outline).")
        speak.say("board_moved", "The board has moved. Please put it back where it was.")
        text = ask("Put the board back and press Enter, 'r' to recalibrate, 'i' to ignore > ").lower()
        if text in ("r", "recalibrate"):
            return None
        if text in ("i", "ignore"):
            return reg
        if text in ("q", "quit"):
            raise KeyboardInterrupt


def learn_pieces(cam, cal):
    """Learn occupancy / colour thresholds and orientation from the starting position."""
    while True:
        ask("\nSTEP 2  Set up ALL pieces in the starting position, then press Enter > ")
        img = cam.grab()
        reg = vision.register(cal, img)                 # the board is often nudged while setting up
        if reg.ok and vision.REG_APPLY_MIN <= reg.shift <= vision.REG_APPLY_MAX:
            vision.apply_registration(cal, reg)
            log.info("board followed while learning the pieces: %r", reg)
        rep = vision.learn_from_start(cal, img)
        obs = vision.observe(cal, img)
        problems = vision.position_problems(cal, chess.Board(), obs)
        show("start_position", vision.draw_problems(
            vision.draw_grid(obs.warped, obs.occ, obs.white, chess.Board()), problems))
        n_occ = int(obs.occ[vision.START_OCC].sum())
        n_white = int((obs.occ & obs.white)[vision.START_WHITE].sum())
        print("Detected %d of 32 pieces, %d of 16 white pieces." % (n_occ, n_white))
        print("Occupancy separation %.1f, colour separation %.1f (both should be clearly > 0)."
              % (rep["occupancy_separation"], rep["colour_separation"]))
        log.info("learned pieces: %s", rep)
        good = (not problems and rep["occupancy_separation"] > 0 and rep["colour_separation"] > 0)
        if good:
            print("Board orientation: white is at the bottom of chess_debug/start_position.jpg.")
            cal.save(CONFIG_YAML, REFERENCE_PNG)
            return obs
        for _, text in problems[:8]:
            print("   " + text)
        print("Detection is not clean. Improve the lighting / avoid shadows and glare, then retry.")
        if not yes(ask("Retry? [Y/n] > ")):
            print("Continuing anyway; you can always type moves at the prompt.")
            cal.save(CONFIG_YAML, REFERENCE_PNG)
            return obs


# ----------------------------------------------------------------- saving games
def parse_typed_move(board, text):
    for parser in (board.parse_uci, board.parse_san):
        try:
            return parser(text)
        except Exception:
            pass
    return None


def _write_pgn(board, robot_color, engine_name, result, started, path):
    game = chess.pgn.Game.from_board(board)
    robot_name = "TonyPi (%s)" % engine_name
    game.headers["Event"] = "TonyPi chess"
    game.headers["Site"] = "TonyPi"
    game.headers["Date"] = started.strftime("%Y.%m.%d")
    game.headers["White"] = robot_name if robot_color == chess.WHITE else "Human"
    game.headers["Black"] = robot_name if robot_color == chess.BLACK else "Human"
    game.headers["Result"] = result
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        print(game, file=fp, end="\n\n")
    os.replace(tmp, path)


def save_pgn(board, robot_color, engine_name, result, started):
    """Write the game to chess_games/<timestamp>.pgn. Returns the path or None."""
    if not board.move_stack:
        return None
    try:
        path = os.path.join(GAMES_DIR, started.strftime("%Y%m%d_%H%M%S") + ".pgn")
        _write_pgn(board, robot_color, engine_name, result, started, path)
        return path
    except Exception as e:
        print("Could not save the PGN: %s" % e)
        return None


def autosave(board, robot_color, engine_name, started):
    """Keep the game so far in chess_games/unfinished.pgn."""
    try:
        _write_pgn(board, robot_color, engine_name, "*", started, os.path.join(GAMES_DIR, UNFINISHED))
    except Exception as e:
        log.warning("could not save the game so far: %s", e)


def forget_unfinished():
    try:
        os.remove(os.path.join(GAMES_DIR, UNFINISHED))
    except OSError:
        pass


def load_unfinished():
    """Returns (board, robot colour) of the interrupted game, or None."""
    path = os.path.join(GAMES_DIR, UNFINISHED)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            game = chess.pgn.read_game(fp)
    except OSError:
        return None
    if game is None:
        return None
    board = game.board()
    for m in game.mainline_moves():
        board.push(m)
    robot_color = chess.WHITE if game.headers.get("White", "").startswith("TonyPi") else chess.BLACK
    return board, robot_color


# ----------------------------------------------------------------- the game
class Game:
    def __init__(self, args, robot, cam, cal, engine):
        self.args = args
        self.robot = robot
        self.cam = cam
        self.cal = cal
        self.engine = engine
        self.board = chess.Board()
        self.robot_color = chess.WHITE if args.white else chess.BLACK
        self.started = datetime.datetime.now()
        self.result = "*"
        self.baseline = None            # the last look at the board that matched the position
        self.arm_failures = 0
        self.accepted = {}              # square -> piece the player said is fine although it looks wrong
        self._last_obs = None
        self.lost_board = False         # the last look could not find the board where it should be
        self.never_mind_board = False   # the player said to carry on although the board was not found
        self.watching = bool(args.auto) # --auto, until a move cannot be read; then the Enter key again

    # -- looking
    def look(self, expected_board):
        """Photograph the board, following it if it was nudged. Returns (image, observation)."""
        img = self.cam.grab()
        occ = vision.expected_state(expected_board)[0]
        reg, moved = vision.track(self.cal, img, occ)
        self.lost_board = (not reg.ok or reg.shift > vision.REG_APPLY_MAX) and not self.never_mind_board
        if moved:
            log.info("board followed: %r", reg)
            self.arms_see_board()
        elif self.lost_board:
            log.warning("board not found where expected: %r", reg)
        return img, vision.observe(self.cal, img)

    def find_board_again(self, board):
        """The board is not where it was: get the player to put it back. False = stop the game."""
        reg = locate_board(self.cam, self.cal, vision.expected_state(board)[0])
        if reg is None:
            print("The board has to be photographed empty for that, which cannot be done in the middle")
            print("of a game. Stop with 'q' and start again with --recalibrate --resume.")
            return False
        if not reg.ok or reg.shift > vision.REG_APPLY_MAX:
            self.never_mind_board = True                  # 'ignore': do not ask again in this game
            if self.robot.use_arms:
                print("I cannot see where the board is, so my arms stay down.")
                self.robot.use_arms = False
        self.lost_board = False
        self.arms_see_board()
        if self.robot.use_arms and not self.robot.hands.ready():
            print("My arms stay down for now: %s." % self.robot.hands.why_off)
        return True

    def arms_see_board(self):
        if self.robot.use_arms:
            self.robot.hands.set_board(self.cal.oriented_corners())

    def check_position(self, board, before=None, what="the board"):
        """
        Look until the board shows `board` (or the player says it is fine).
        `before` is the position the last good look was taken in. Returns
        False if the player wants to stop.
        """
        if self.cam is None:
            return True
        while True:
            try:
                img, obs = self.look(board)
            except RuntimeError as e:
                print("Camera problem: %s" % e)
                return True
            problems = self.new_problems(board, vision.position_problems(self.cal, board, obs))
            if problems and self.lost_board:
                if not self.find_board_again(board):
                    return False
                continue
            if before is not None and self.baseline is not None and not problems:
                # the position is right; has anything else been knocked out of place?
                change = vision.change_map(obs, self.baseline)
                problems = self.new_problems(board, [
                    p for p in vision.change_problems(self.cal, before, board, change)
                    if p[1].endswith(" changed")])
            show("last_read", vision.draw_problems(
                vision.draw_grid(obs.warped, obs.occ, obs.white, board), problems))
            if not problems:
                self.baseline = obs
                return True
            print("I looked at %s and something is not right:" % what)
            for _, text in problems[:8]:
                print("   " + text)
            print("(chess_debug/last_read.jpg shows what I see; the board should be)\n")
            print(board)
            log.warning("position check failed: %s", [t for _, t in problems])
            speak.say("please_fix", "Something is not right on the board. Please put it right for me.")
            text = ask("\nPut it right and press Enter ('ok' = it is right as it is, 'q' = quit) > ").lower()
            if text in ("q", "quit"):
                return False
            if text in ("ok", "o"):
                for sq, _ in problems:                    # do not ask about these again
                    self.accepted[sq] = board.piece_at(sq)
                self.baseline = obs
                return True

    def new_problems(self, board, problems):
        """Drop the problems the player has already said are fine (until that square is played on)."""
        self.accepted = {sq: p for sq, p in self.accepted.items() if board.piece_at(sq) == p}
        return [p for p in problems if p[0] not in self.accepted]

    # -- the player's move
    def read_move(self):
        """Infer the player's move from the camera. Returns (move or None, sure, text)."""
        img, obs = self.look(self.board)
        ranked, no_move = vision.rank_moves(self.cal, self.board, obs, self.baseline)
        move, sure, text = vision.decide(ranked, no_move)
        after = self.board.copy(stack=False)
        if move is not None:
            after.push(move)
        show("camera", img)
        show("last_read", vision.draw_grid(obs.warped, obs.occ, obs.white, after))
        if move is not None:
            print("Camera: %s (%s)" % (self.board.san(move), text))
            log.info("read %s sure=%s %s; next: %s", self.board.san(move), sure, text,
                     ", ".join("%s=%d" % (self.board.san(c.move), c.total) for c in ranked[1:4]))
        else:
            print("Camera: %s" % text)
            log.info("no move read: %s", text)
        self._last_obs = obs
        return move, sure, text

    def confirm(self, move):
        """Ask the player whether the best guess is right. Returns a move, None to look again, or 'q'."""
        speak.say(["is_this_your_move"], "Is this your move: %s?" % self.board.san(move))
        text = ask("Did you play %s? Enter = yes, or type your move > " % self.board.san(move))
        if text.lower() in ("", "y", "yes"):
            return move
        if text.lower() in ("q", "quit", "resign"):
            return "q"
        if text.lower() in ("n", "no"):
            return None
        typed_move = parse_typed_move(self.board, text)
        if typed_move is None:
            print("'%s' is not a legal move here." % text)
        return typed_move

    def watch(self):
        """
        --auto: wait until the board has changed and then stood still, and
        read the move. Returns a line as if typed at the prompt: '' to read
        the board now, or whatever the player typed in the meantime.
        """
        print("Make your move. (You can still type a move, 'u' or 'q' and press Enter.)")
        last, steady, tries = None, 0, 0
        while True:
            line = typed(WATCH_PERIOD)
            if line is not None:
                return line
            img = self.cam.frame()
            if img is None:
                continue
            small = cv2.resize(cv2.cvtColor(self.cal.warp(img), cv2.COLOR_BGR2GRAY), (100, 100)).astype(np.int16)
            moving = last is not None and float(np.abs(small - last).mean()) > 2.0
            last = small
            steady = 0 if moving else steady + 1
            if steady < WATCH_STEADY:
                continue
            obs = vision.observe(self.cal, img)
            if vision.unchanged(self.cal, self.board, obs, self.baseline):
                continue                                  # nothing has happened yet
            ranked, no_move = vision.rank_moves(self.cal, self.board, obs, self.baseline)
            move, sure, _ = vision.decide(ranked, no_move)
            if move is not None and sure:
                return ""
            steady = 0
            tries += 1
            if tries >= WATCH_TRIES:
                return ""

    def players_turn(self):
        """Returns False when the game should stop."""
        if self.watching and self.cam is not None:
            text = self.watch()
        else:
            text = ask("Your move: press Enter when done (or type a move, 'u' undo, 'q' quit) > ")
        self.watching = bool(self.args.auto)
        if text.lower() in ("q", "quit", "resign"):
            self.result = "1-0" if self.robot_color == chess.WHITE else "0-1"
            speak.say("good_game", "Good game.")
            return False
        if text.lower() in ("u", "undo"):
            if len(self.board.move_stack) >= 2:
                self.board.pop()
                self.board.pop()
                print("Undid the last two moves. Put the pieces back:\n")
                print(self.board)
                ask("\nPress Enter when the board shows this position > ")
                self.baseline = None
                if not self.check_position(self.board):
                    return False
                autosave(self.board, self.robot_color, self.engine.name, self.started)
            else:
                print("Nothing to undo.")
            return True
        obs = None
        if text:
            move = parse_typed_move(self.board, text)
            if move is None:
                print("'%s' is not a legal move here." % text)
                return True
        elif self.cam is None:
            print("No camera. Type your move, for example e2e4 or Nf3.")
            return True
        else:
            try:
                move, sure, _ = self.read_move()
                obs = self._last_obs
            except Exception as e:
                log.exception("reading the move failed")
                print("Camera problem: %s" % e)
                move, sure = None, False
            if move is None and self.lost_board:
                return self.find_board_again(self.board)
            if move is None:
                speak.say("did_not_understand", "I could not read your move. Keep your hands out of "
                                                "the picture and press Enter again, or type the move.")
                self.watching = False                     # wait for the player instead of nagging
                return True
            if not sure:
                move = self.confirm(move)
                if move == "q":
                    self.result = "1-0" if self.robot_color == chess.WHITE else "0-1"
                    return False
                if move is None:
                    return True
        print("You played %s" % self.board.san(move))
        before = self.board.copy(stack=False)
        self.board.push(move)
        if obs is not None and not self.new_problems(
                self.board, vision.position_problems(self.cal, self.board, obs)):
            self.baseline = obs
        elif self.cam is not None:
            # typed or corrected move: make sure the board really shows it before going on
            if not self.check_position(self.board, before):
                return False
        autosave(self.board, self.robot_color, self.engine.name, self.started)
        return True

    # -- the robot's move
    def robots_turn(self):
        """Returns False when the game should stop."""
        move = self.engine.best_move(self.board)
        if move is None or move not in self.board.legal_moves:
            print("The engine returned no legal move; stopping.")
            log.error("engine returned %r", move)
            return False
        before = self.board.copy(stack=False)
        after = self.board.copy(stack=False)
        after.push(move)
        speak.announce_move(self.board, move)
        log.info("robot plays %s", self.board.san(move))

        done_by_arm = False
        if self.robot.use_arms:
            done_by_arm = self.move_with_arms(move)
        if not done_by_arm:
            if self.cam is None:
                print("(Make my move on your board.)")
            else:
                speak.say("please_move_my_piece", "Please move my piece for me, then press Enter.")
                if ask("Press Enter when you have made my move %s ('q' = quit) > "
                       % self.board.san(move)).lower() in ("q", "quit"):
                    return False
        if move.promotion and done_by_arm:
            speak.say("swap_for_queen", "Please swap my pawn on %s for a %s, then press Enter."
                      % (chess.square_name(move.to_square), chess.piece_name(move.promotion)))
            if ask("Press Enter when the %s stands on %s > "
                   % (chess.piece_name(move.promotion), chess.square_name(move.to_square))).lower() in ("q", "quit"):
                return False
        if not self.check_position(after, before, "my move"):
            return False
        self.board.push(move)
        autosave(self.board, self.robot_color, self.engine.name, self.started)
        if not self.board.is_game_over(claim_draw=True):
            speak.say("your_move", "Your move.")
        return True

    def move_with_arms(self, move):
        """Play the move with the arms. Returns False if the player has to do it."""
        hands = self.robot.hands
        if not hands.ready():
            print("My arms stay down: %s." % hands.why_off)
            return False
        plan, why = hands.plan(self.board, move)
        if plan is None:
            print("I cannot make this move myself: %s." % why)
            log.info("no plan for %s: %s", self.board.san(move), why)
            return False
        if not hands.battery_ok():
            speak.say("battery_low", "My battery is low. I will not move my arms.")
            self.robot.use_arms = False
            return False
        print("My arms: %r" % plan)
        speak.say("keep_clear", "Keep your hands clear. I am moving my piece.")
        self.robot.warn()
        try:
            hands.execute(plan)
            self.arm_failures = 0
            return True
        except KeyboardInterrupt:
            raise
        except chess_arm.ArmError as e:
            print("My arm could not finish the move: %s" % e)
            self.arm_failures += 1
        held = hands.holding()
        if held is not None:
            speak.say("take_piece", "Please take the piece out of my %s hand." % held[0])
            ask("Hold the piece in my %s hand and press Enter; I will let go > " % held[0])
            try:
                hands.release()
            except Exception as e:
                log.error("could not open the gripper: %s", e)
        if self.arm_failures >= 3:
            print("My arms failed three times in a row; I will ask you to move my pieces from now on.")
            print("Check the arms with chess_teach.py before the next game.")
            self.robot.use_arms = False
        self.robot.look_at_board()
        return False

    # -- start to finish
    def run(self):
        print("\n" + str(self.board) + "\n")
        while not self.board.is_game_over(claim_draw=True):
            if self.board.turn == self.robot_color:
                go_on = self.robots_turn()
            else:
                go_on = self.players_turn()
            if not go_on:
                return
            print("\n" + str(self.board) + "\n")
        self.result = self.board.result(claim_draw=True)
        print("Result: " + self.result)
        if self.board.is_checkmate():
            if self.board.turn != self.robot_color:
                speak.say("i_win", "Checkmate. I win. Good game.")
                self.robot.gesture("chest")
            else:
                speak.say("you_win", "You win. Well played.")
                self.robot.gesture("bow")
        else:
            speak.say("draw", "The game is a draw.")
            self.robot.gesture("bow")


def check_arms(game):
    """Hold a hand over one square per arm so the player can see the arms are on target."""
    hands = game.robot.hands
    for name in chess_arm.ARM_NAMES:
        squares = [s for s in chess.SQUARES if name in hands.who_reaches(s)]
        if not squares:
            continue
        mine = [s for s in squares if game.board.color_at(s) == game.robot_color]
        square = (mine or squares)[0]
        arm = hands.arms[name]
        print("My %s hand goes above %s." % (name, chess.square_name(square)))
        game.robot.warn()
        try:
            arm.to_board()
            arm.open()
            arm.move(arm.target(hands.square_uv(square), "hover"))
            arm._last_hover = arm.pose
            good = yes(ask("Is my %s hand centred above %s? [Y/n] > " % (name, chess.square_name(square))))
            arm.to_rest()
        except chess_arm.ArmError as e:
            print("The %s arm failed: %s" % (name, e))
            arm.recover()
            good = False
        if not good:
            print("I will not use my arms in this game. Teach them again with chess_teach.py.")
            game.robot.use_arms = False
            return
    game.robot.look_at_board()


def usable_calibration(cam):
    """The saved board calibration, if it fits the picture the camera gives now; else None."""
    cal = vision.Calibration.load(CONFIG_YAML)
    if cal is None:
        return None
    first = cam.grab(1)
    size = (first.shape[1], first.shape[0])
    if cal.image_size is None:
        print("The saved board calibration is from an older version of this program (made before")
        print("the lens correction); the board has to be photographed again.")
        return None
    if size != tuple(cal.image_size):
        print("The camera picture is %dx%d but the calibration was made at %dx%d; recalibrating."
              % (size + tuple(cal.image_size)))
        return None
    if (cal.K is None) != (getattr(cam, "K", None) is None):
        print("The saved board calibration was made with a different camera set-up; recalibrating.")
        return None
    if getattr(cam, "K", None) is not None:
        cal.K = cam.K
    return cal


def prepare_board(args, robot, cam):
    """Calibration, board position and piece learning. Returns (calibration, first observation)."""
    cal = None if args.recalibrate else usable_calibration(cam)
    while True:
        if cal is None:
            cal = calibrate_board(cam)
        else:
            print("Loaded board calibration from %s (use --recalibrate if the board or camera moved)."
                  % CONFIG_YAML)
            if locate_board(cam, cal) is None:
                cal = None
                continue
        return cal, learn_pieces(cam, cal)


def play(args):
    setup_logging()
    log.info("---- ChessPlay started: %s", vars(args))
    if getattr(args, "web", None):
        serve_debug(args.web)
    if not args.no_robot and chess_arm.service_running():
        print("The tonypi app service is running. It uses the servo port and the camera, and I need")
        print("both to myself. (After the game  sudo systemctl start tonypi  brings the phone app back.)")
        if yes(ask("Shall I stop it now? [Y/n] > ")):
            if chess_arm.stop_service():
                print("Stopped.")
            else:
                print("I could not stop it. Do it yourself:  sudo systemctl stop tonypi")
        if chess_arm.service_running() and not args.no_arms \
                and not yes(ask("Carry on with the service running? My arms may fail. [y/N] > ") or "n"):
            return
    robot = Robot(enabled=not args.no_robot, arms=not args.no_arms, tilt=args.tilt)
    cam = None
    engine = None
    game = None
    try:
        robot.stance()
        cam = None if args.no_camera else Cam(args.camera, use_robot=not args.no_robot)
        engine = chess_engine.make_engine(time_limit=args.time, skill=args.skill,
                                          engine=args.engine, path=args.stockfish)
        speak.set_volume(80)

        resumed = load_unfinished() if args.resume else None
        if args.resume and resumed is None:
            print("There is no interrupted game; starting a new one.")

        cal, first = None, None
        saved = None if (cam is None or args.recalibrate or resumed is None) else usable_calibration(cam)
        if cam is None:
            print("No camera: type your moves at the prompt (e2e4, Nf3, O-O ...).")
        elif resumed is not None and saved is not None and saved.ready:
            cal = saved                                  # the pieces were learned before the interruption
        else:
            if resumed is not None:
                print("I have to learn the board first. Set up the STARTING position for that;")
                print("afterwards I will ask you to set up the interrupted game.")
            cal, first = prepare_board(args, robot, cam)

        game = Game(args, robot, cam, cal, engine)
        game.baseline = first
        if resumed is not None:
            game.board, game.robot_color = resumed
            print("\nCarrying on with the interrupted game. The board should show:\n")
            print(game.board)
            if cam is not None:
                ask("\nPress Enter when the board shows this position > ")
                if locate_board(cam, cal, vision.expected_state(game.board)[0]) is None:
                    print("Recalibrate with --recalibrate (starting position needed), then use --resume again.")
                    return
                game.baseline = None
                if not game.check_position(game.board):
                    return

        if robot.use_arms:
            drift = robot.hands.set_board(cal.oriented_corners()) if cal is not None else None
            if cal is None:
                robot.use_arms = False
                robot.why_no_arms = "there is no camera to find the board with"
            elif not robot.hands.ready():
                print("My arms stay down: %s." % robot.hands.why_off)
                print("Put the board back where it stood when the arms were taught, or teach them again.")
                robot.use_arms = False
            else:
                print("The board is %.2f squares from where my arms were taught. Squares I can reach"
                      " (R right arm, L left arm, B both):\n" % drift)
                print(robot.hands.reach_map())
                if args.check_arms:
                    check_arms(game)
        if not robot.use_arms and robot.enabled:
            print("I will announce my moves and ask you to make them (%s)."
                  % (robot.hands.why_off or robot.why_no_arms))

        if game.robot_color == chess.WHITE:
            speak.say("lets_play", "Let's play chess. I am white, so I will start.")
        elif args.auto:
            speak.say("lets_play", "Let's play chess. You are white. Make your move.")
        else:
            speak.say("lets_play", "Let's play chess. Make your move, then press Enter.")
        game.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        speak.say("good_game", "Good game.")
    except chess_arm.ArmError as e:                         # one that nothing further down could deal with
        log.exception("ChessPlay stopped on an error")
        print("\nI have to stop: %s" % e)
    except Exception:
        log.exception("ChessPlay stopped on an error")      # the arms are brought to rest below
        raise
    finally:
        if game is not None:
            finished = game.result != "*"
            pgn = save_pgn(game.board, game.robot_color, engine.name, game.result, game.started)
            if pgn:
                print("Game saved to %s" % pgn)
            if finished:
                forget_unfinished()
            elif game.board.move_stack:
                print("Carry on later with:  python3 ChessPlay.py --resume")
        if engine is not None:
            engine.quit()
        try:
            robot.shutdown()
        except Exception as e:
            log.error("shutdown: %s", e)
        if cam is not None:
            cam.close()
        if SHOW:
            cv2.destroyAllWindows()
        log.info("---- ChessPlay finished")


def _terminate(signum, frame):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description="TonyPi plays chess")
    ap.add_argument("--white", action="store_true", help="robot plays white (moves first)")
    ap.add_argument("--time", type=float, default=2.0, help="thinking time per move, seconds")
    ap.add_argument("--skill", type=int, default=10, help="strength 0-20")
    ap.add_argument("--engine", choices=["auto", "stockfish", "builtin"], default="auto",
                    help="auto: Stockfish if installed, else built-in search (default)")
    ap.add_argument("--stockfish", default=None, metavar="PATH", help="path to the Stockfish binary")
    ap.add_argument("--recalibrate", action="store_true", help="redo board detection")
    ap.add_argument("--no-arms", action="store_true", help="announce moves only; the player moves the pieces")
    ap.add_argument("--check-arms", action="store_true", help="show that the arms are on target before the game")
    ap.add_argument("--auto", action="store_true", help="notice the player's move without Enter")
    ap.add_argument("--resume", action="store_true", help="carry on with the interrupted game")
    ap.add_argument("--no-robot", action="store_true", help="no servos; for testing on a PC")
    ap.add_argument("--no-camera", action="store_true", help="type every move; no board reading")
    ap.add_argument("--camera", default=None, help="camera index or stream URL (default: robot camera)")
    ap.add_argument("--tilt", type=int, default=None, metavar="PULSE",
                    help="head tilt 500-2500 for this game (default: the value saved by chess_teach.py)")
    ap.add_argument("--web", type=int, nargs="?", const=8081, default=None, metavar="PORT",
                    help="show the pictures in chess_debug in a browser: http://<robot>:8081/")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _terminate)
    try:
        play(args)
    except RuntimeError as e:
        print("Error: %s" % e)
        sys.exit(1)


if __name__ == "__main__":
    main()
