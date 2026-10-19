#!/usr/bin/python3
# coding=utf8
"""
PotCollect: TonyPi tidies small white pots to the centre of the table it stands on.

What happens
    1. The robot stands up, looks down at the table and remembers the colour
       of the table top right in front of its feet.
    2. It turns on the spot until it sees the AprilTag that marks the centre
       of the table. From the tag it knows where it stands.
    3. It looks for a white pot: first where it is looking, then by turning
       half-way round to the right and half-way round to the left.
    4. It walks to the pot, bends down, takes it between both arms and lifts
       it over its head (the robot's own "move_up" action).
    5. It turns until it sees the tag again, walks to it and puts the pot
       down just in front of the tag, never on it (the tag must stay
       visible), and not on a pot that is already there.
    6. It steps back and starts again at 3. Pots standing by the tag are
       "collected" and are left alone; the robot remembers where it put
       them and walks round them. When no pot is left it waits, facing the
       centre, for you to put one where it can see it.

What you need
    * White pots on a table top that is NOT white and not shiny.
    * An AprilTag of the family tag36h11 (number 1 unless you say --tag N)
      at the centre of the table, taped down flat on all four sides: the
      robot may walk over it. Bigger is better, 8 to 10 cm: the robot has
      to recognise it from wherever it picks a pot up. pot_tag_1.png in the
      project folder is such a tag: printed at 100 % its black square is
      10 cm; cut along the grey line. Whatever tag you use, measure the side
      of the black square and write it into SETTINGS["tag_size_cm"], and
      leave only a narrow white border round it.
    * Room. The robot's heels come within  tag_size/2 + drop_gap + 27 cm  of
      the tag (42 cm with the numbers below), so the table has to reach
      50 cm from the tag on every side. Pots at least 15 cm from the edges
      and not further than about 70 cm from the robot (it does not see
      further).
    * Start the robot standing about 40 cm from the tag and facing it.

How it keeps from falling off the table
    The robot walks forward only towards a pot or the tag, and only while
    the picture shows table top all the way from its feet to that pot or
    tag. A step sideways is taken only if the picture shows table top on
    that side. White things that are not surrounded by table top, or that
    run out of the picture (a wall, a white floor), are not pots.
    It cannot see behind itself. It steps back only a little: two steps
    after putting a pot down, and away from the tag when it has come too
    close to it. That is why the table has to be big enough round the tag.
    All of this depends on the robot telling table top from everything
    else by its colour, so look at it with --view before the first run:
    what the robot takes for table top is tinted green, pots pink.

First time on the robot (app service stopped: sudo systemctl stop tonypi)
    cd /home/pi/TonyPi/Functions
    1. python3 PotCollect.py --view
       The robot stands and shows what it sees. Check: the table is green
       right up to its edges and nothing else is; a pot gets a green box;
       the tag gets a yellow outline and its distance. Click on a pot if
       it gets no box (that teaches the pot colour).
    2. python3 PotCollect.py --grab-test
       Put a pot in front of the robot where the two short blue lines are.
       When it stands right the robot beeps, picks it up and puts it down
       again. If the pot slips, change "carry_lock" below.
    3. python3 PotCollect.py --once       (stay close, a hand near the robot)
    4. python3 PotCollect.py
Options:
    --view            do not walk: show what the robot sees. Click on a pot
                      to teach its colour, m hides or shows the colours,
                      t learns the table colour again, q quits
    --grab-test       do not walk: pick up the pot put in front of the robot
                      and put it down again
    --once            stop after one pot
    --tag N           number of the tag at the centre (default 1), or "any"
    --no-table-check  walk without checking for table top (robot can fall!)
    --no-window       no picture window (when there is no screen)
Ctrl-C stops the robot after the movement it is making; if it carries a pot
it puts it down first. Ctrl-C a second time quits at once.

Numbers you may have to change are in SETTINGS below. Instead of editing
this file you can put them in /home/pi/TonyPi/pot_config.yaml, e.g.
    tag_size_cm: 8.0
pot_selftest.py tries the whole program on a PC against a pretend robot.
The real robot has not run it yet: the numbers that describe the robot
(where a pot has to be in the picture to be picked up, how far a step goes)
are taken from the robot's own transport game or guessed.
"""
import os
import sys
import math
import time
import signal
import argparse
import threading
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "HiwonderSDK"))

ACTION_DIR = os.path.join(ROOT, "ActionGroups") + os.sep
CONFIG_YAML = os.path.join(ROOT, "pot_config.yaml")
SERVO_YAML = os.path.join(ROOT, "servo_config.yaml")
VIEW_JPG = os.path.join(ROOT, "pot_view.jpg")
LENS_NPZ = os.path.join(HERE, "CameraCalibration", "calibration_param.npz")
STREAM_URL = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
WINDOW = "PotCollect"

SIZE = (640, 480)               # every picture is brought to this size; all pixel numbers below refer to it
LENS_TRUST = 0.72               # the lens calibration is only good this far from the middle of the picture
STILL = ("stand", "stand_slow", "pick", "put")      # actions that leave the feet where they are

SETTINGS = {
    # ---- the pots: colour range in LAB (OpenCV scale 0-255), size in the picture (pixels x pixels at
    #      640 x 480) and width on the table (cm; only checked once the tag has been seen)
    "pot_lab_min": [170, 108, 104],
    "pot_lab_max": [255, 148, 152],
    "pot_min_area": 500,
    "pot_max_area": 60000,
    "pot_min_cm": 3.0,
    "pot_max_cm": 14.0,

    # ---- the AprilTag at the centre of the table
    "tag_id": 1,                # None = any tag of the family tag36h11
    "tag_size_cm": 10.0,        # side of the black square: MEASURE YOURS
    "drop_gap_cm": 10.0,        # a pot is put down this far from the edge of the tag
    "drop_tol_cm": 4.0,         # ...and the robot steps back if it came this much closer than that
    "home_radius_cm": 28.0,     # a pot this close to the middle of the tag has been collected
    "home_chain_cm": 16.0,      # ...and so has a pot this close to a collected one
    "spot_cm": 9.0,             # a pot this close to where the robot put one down is that pot
    "guess_cm": 8.0,            # extra allowance while the tag is out of view and the robot reckons where it is
    "pot_clear_cm": 10.0,       # room left between the pot being put down and one that is already there

    # ---- where things are in the picture (pixels). Taken from the robot's own transport game.
    "center_x": 350,            # column in which something straight in front of the arms appears
    "turn_px": 80,              # further off-centre than this: turn towards it
    "side_big_px": 40,          # close and further off-centre than this: big step sideways
    "side_small_px": 20,        # ...small step sideways
    "near_y": 250,              # below this row the pot (or tag) is close
    "step_y": 280,              # below this row the robot creeps in single steps
    "grab_y": 340,              # below this row the pot is close enough to be picked up
    "too_close_y": 430,         # below this row it is too close: step back
    "drop_point": [350, 390],   # the spot on the table where a pot ends up if the robot puts it down now
    "feet_y": 470,              # row of the table top right in front of the feet
    "track_px": 130,            # a pot moves less than this in the picture from one look to the next
    "far_y": 100,               # above this row a pot is far away and easily slips out of the top of the picture
    "turn_shift_px": 55,        # how far one turn step moves things sideways in the picture, roughly

    # ---- the table top: how far a pixel may differ from the colour in front of the feet and still be
    #      table (L darker, L brighter, a, b), and how long a stretch that is not table may be (pixels)
    "table_check": True,
    "table_tol": [80, 50, 14, 14],
    "table_gap_px": 24,

    # ---- the head: None = the values in servo_config.yaml, as in the robot's own games
    "head_tilt": None,
    "head_pan": None,

    # ---- action groups (files in ActionGroups/)
    "forward": "go_forward",
    "one_step": "go_forward_one_step",
    "back": "back_fast",
    "turn_left": "turn_left_small_step",
    "turn_right": "turn_right_small_step",
    "left": "left_move",
    "right": "right_move",
    "left_big": "left_move_30",
    "right_big": "right_move_30",
    "pick": "move_up",          # bend down, take the pot between both arms, lift it over the head
    "put": "put_down",          # the reverse
    # arm servos held still while the robot walks with a pot. Elbows 6 and 14 decide how hard the arms
    # squeeze: raise 6 and lower 14 by the same amount to squeeze harder.
    "carry_lock": {"6": 650, "7": 850, "8": 0, "14": 350, "15": 150, "16": 1000},
    "grab_steps": 2,            # single steps towards the pot before bending down
    "put_steps": 2,             # single steps before putting it down
    "leave_steps": 2,           # steps back after putting a pot down
    "clear_ahead_cm": 18.0,     # a pot nearer than this in front of the robot...
    "clear_side_cm": 12.0,      # ...and less than this off its path is in the way: the robot steps round it
    "grab_tries": 2,            # pick-up attempts per pot
    "missed_y": 300,            # a pot still seen below this row after picking up was missed...
    "missed_dx": 130,           # ...if it is this close to the centre column

    # ---- what one run of each walking action does: cm forward, cm to the left, degrees turned to the left.
    #      First guesses only: the robot measures each of them whenever the tag is in view before and after.
    "steps": {
        "forward": [3.0, 0.0, 0.0], "one_step": [2.0, 0.0, 0.0], "back": [-2.5, 0.0, 0.0],
        "turn_left": [0.0, 0.0, 7.5], "turn_right": [0.0, 0.0, -7.5],
        "left": [0.0, 1.0, 0.0], "right": [0.0, -1.0, 0.0],
        "left_big": [0.0, 2.5, 0.0], "right_big": [0.0, -2.5, 0.0],
    },
    "turn_centre_cm": 14.0,     # from the point the robot turns about to the drop point on the table
    "search_turn_steps": 2,     # turn steps between two looks while searching

    # ---- patience
    "lost_looks": 3,            # looks without the pot (or tag) before the robot gives it up
    "max_moves": 70,            # movements per walk; more means something is wrong
    "max_failures": 3,          # pots in a row that could not be fetched before the robot stops
    "rescan_s": 0,              # while waiting for a pot: turn round to look again this often (0 = never)
    "settle_s": 0.4,            # wait after a movement before the picture is used
}


# ----------------------------------------------------------------- settings
def load_settings(path=CONFIG_YAML):
    """SETTINGS, with the values from pot_config.yaml on top when that file exists."""
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in SETTINGS.items()}
    if os.path.isfile(path):
        import yaml
        with open(path, "r", encoding="utf-8") as fp:
            extra = yaml.safe_load(fp) or {}
        for key, value in extra.items():
            if key not in cfg:
                print("pot_config.yaml: unknown setting '%s' ignored" % key)
            elif key == "carry_lock":
                cfg[key] = {str(k): int(v) for k, v in value.items()}
            elif key == "steps":
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def save_settings(changes, path=CONFIG_YAML):
    """Write `changes` into pot_config.yaml, keeping what is already in there."""
    import yaml
    extra = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fp:
            extra = yaml.safe_load(fp) or {}
    extra.update(changes)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        yaml.safe_dump(extra, fp)
    os.replace(tmp, path)


def missing_actions(cfg, folder=ACTION_DIR):
    names = [cfg[k] for k in ("forward", "one_step", "back", "turn_left", "turn_right", "left", "right",
                              "left_big", "right_big", "pick", "put")]
    names += ["stand", "stand_slow", "go_forward_start", "go_forward_end", "back_start", "back_end"]
    return sorted(set(n for n in names if not os.path.isfile(os.path.join(folder, n + ".d6a"))))


def load_lens(path=LENS_NPZ):
    """(camera matrix, distortion) of the robot's camera, or None without a calibration file."""
    try:
        data = np.load(path)
        return np.array(data["mtx_array"], np.float64), np.array(data["dist_array"], np.float64)
    except Exception as e:
        print("No lens calibration (%s): distances will be less exact." % e)
        return None


def wrap(angle):
    """An angle in radians brought into -pi .. pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------- AprilTag
class TagFinder:
    """Finds tag36h11 tags with the robot's apriltag library, or with OpenCV where that is missing."""

    def __init__(self):
        self.detector = None
        self.aruco = None
        try:
            import hiwonder.apriltag as apriltag
            self.detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
            return
        except Exception:
            pass
        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            raise RuntimeError("no AprilTag detector: neither hiwonder.apriltag nor cv2.aruco can be used")
        if hasattr(aruco, "ArucoDetector"):                 # OpenCV 4.7 and later
            finder = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11),
                                         aruco.DetectorParameters())
            self.aruco = finder.detectMarkers
        else:
            dictionary = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
            params = aruco.DetectorParameters_create()
            self.aruco = lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)

    def find(self, gray):
        """[(tag number, 4 corners in the picture)]"""
        if self.detector is not None:
            return [(int(d.tag_id), np.array(d.corners, np.float64)) for d in self.detector.detect(gray)
                    if d.tag_family in (b"tag36h11", "tag36h11")]
        corners, ids, _ = self.aruco(gray)
        if ids is None:
            return []
        return [(int(i), np.array(c, np.float64).reshape(4, 2)) for c, i in zip(corners, ids.ravel())]


# ----------------------------------------------------------------- what the camera sees
class Pot:
    def __init__(self, box, area):
        x, y, w, h = box
        self.box = box
        self.area = area
        self.x, self.y = x + w // 2, y + h // 2     # middle of the pot in the picture
        self.base = (self.x, y + h)                 # where it stands on the table
        self.robot = None                           # that spot seen from the robot: cm forward, cm to the left
        self.cm = None                              # that spot on the table, in cm from the middle of the tag
        self.exact = False                          # cm was measured with the tag in view, not reckoned
        self.home = False                           # stands at the centre already (seen together with the tag)
        self.maybe_home = False                     # ...or so the robot reckons, the tag being out of view
        self.off_table = False                      # no table top beside it, or none on the way to it


class Tag:
    def __init__(self, number, corners):
        self.id = number
        self.corners = corners
        self.x, self.y = (int(round(v)) for v in corners.mean(axis=0))
        self.H = None                               # picture -> table top in cm, middle of the tag = (0, 0)
        self.gap = 0                                # pixels that are not table between the robot and the tag


class Seen:
    """One look at the table."""

    def __init__(self, frame):
        self.frame = frame
        self.tag = None
        self.pots = []
        self.pose = None            # where the robot stands: x, y in cm from the tag, and its heading
        self.drop = None            # the spot where a pot would be put down now, in cm from the tag
        self.gap_cm = None          # ...and how far that is from the middle of the tag
        self.blocked = False        # a collected pot stands where the pot would be put down
        self.lab = None             # half-size pictures:
        self.white = None           # pot-coloured pixels that may be pots
        self.table = None           # pixels that look like the table top
        self.other = None           # pots and tag: they say nothing about the table


class Vision:
    def __init__(self, cfg, lens="file", tags=None):
        self.cfg = cfg
        self.lens = load_lens() if lens == "file" else lens
        self.tags = tags or TagFinder()
        self.table = None           # colour of the table top (L, a, b), learned in front of the feet
        self.ground = None          # picture -> table top as seen from the robot (cm forward, cm left)
        self._ground_area = 0.0     # how big the tag was in the picture that `ground` comes from

    # -- geometry
    def _ideal(self, points):
        """Picture points with the lens distortion taken out; None if one is where the lens model fails."""
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        if self.lens is None:
            return pts
        K, dist = self.lens
        r = np.hypot((pts[:, 0] - K[0, 2]) / K[0, 0], (pts[:, 1] - K[1, 2]) / K[1, 1])
        if (r > LENS_TRUST).any():
            return None
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    def _map(self, H, point):
        pts = None if H is None else self._ideal(point)
        if pts is None:
            return None
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)[0]
        return float(out[0]), float(out[1])

    def on_table(self, tag, point):
        """Where a point of the picture is on the table top: (cm, cm) from the middle of the tag, or None."""
        return None if tag is None else self._map(tag.H, point)

    def from_robot(self, point):
        """Where a point of the picture is on the table top: (cm forward, cm to the left) of the robot."""
        return self._map(self.ground, point)

    # -- one look
    def see(self, frame):
        c = self.cfg
        if (frame.shape[1], frame.shape[0]) != SIZE:
            frame = cv2.resize(frame, SIZE)
        seen = Seen(frame)
        small = cv2.resize(frame, (SIZE[0] // 2, SIZE[1] // 2), interpolation=cv2.INTER_NEAREST)
        seen.lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 3), cv2.COLOR_BGR2LAB)
        white = cv2.inRange(seen.lab, tuple(int(v) for v in c["pot_lab_min"]), tuple(int(v) for v in c["pot_lab_max"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        seen.white = self._without_background(cv2.dilate(cv2.erode(white, kernel), kernel))
        seen.tag = self._find_tag(frame)
        self._locate(seen)
        seen.pots = self._find_pots(seen)
        self._measure(seen)
        self._check_table(seen)
        return seen

    @staticmethod
    def _without_background(white):
        """
        White that runs out of the picture at the top or at the sides is a wall, the floor or a sheet
        of paper, not a pot, and neither is white that all but touches it (tiles of a white floor).
        """
        joined = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        labels = cv2.connectedComponents(joined, connectivity=8)[1]
        background = set()
        for edge in (np.s_[0, :], np.s_[:, 0], np.s_[:, -1]):
            background.update(labels[edge][white[edge] > 0].tolist())
        if background:
            white = white.copy()
            white[np.isin(labels, list(background))] = 0
        return white

    def _find_tag(self, frame):
        c = self.cfg
        best, best_area = None, 0.0
        for number, corners in self.tags.find(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):
            if c["tag_id"] is not None and number != c["tag_id"]:
                continue
            area = abs(cv2.contourArea(corners.astype(np.float32)))
            if area > best_area:
                best, best_area = Tag(number, corners), area
        if best is not None:
            ideal = self._ideal(best.corners)
            if ideal is not None:
                # Corners in the same sense as the square below, so that the table is measured the way it
                # looks from above (not mirrored) whichever way round the detector lists them.
                x, y = ideal[:, 0], ideal[:, 1]
                if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0:
                    ideal = ideal[::-1]
                s = c["tag_size_cm"] / 2.0
                best.H = cv2.getPerspectiveTransform(ideal.astype(np.float32),
                                                     np.float32([[-s, -s], [s, -s], [s, s], [-s, s]]))
        return best

    def _locate(self, seen):
        """With the tag in view: where the robot stands and where it would put a pot down."""
        c = self.cfg
        tag = seen.tag
        if tag is None or tag.H is None:
            return
        x, y = c["drop_point"]
        near, far = self.on_table(tag, (x, y)), self.on_table(tag, (x, y - 140))
        if near is None or far is None:
            return
        heading = math.atan2(far[1] - near[1], far[0] - near[0])
        cos, sin = math.cos(heading), math.sin(heading)
        px, py = near[0] - c["turn_centre_cm"] * cos, near[1] - c["turn_centre_cm"] * sin
        seen.drop = near
        seen.gap_cm = math.hypot(*near)
        seen.pose = (px, py, heading)
        area = abs(cv2.contourArea(tag.corners.astype(np.float32)))
        if area >= 0.6 * self._ground_area:         # the bigger the tag in the picture, the better it measures
            to_robot = np.array([[cos, sin, -cos * px - sin * py], [-sin, cos, sin * px - cos * py], [0.0, 0.0, 1.0]])
            self.ground, self._ground_area = to_robot @ tag.H, max(area, self._ground_area)

    @staticmethod
    def _grown(corners, factor):
        centre = corners.mean(axis=0)
        return centre + (corners - centre) * factor

    def _find_pots(self, seen):
        c = self.cfg
        white = seen.white
        if seen.tag is not None:                    # the white paper the tag is printed on is not a pot
            white = white.copy()
            cv2.fillConvexPoly(white, (self._grown(seen.tag.corners, 1.6) / 2.0).astype(np.int32), 0)
        pots = []
        contours = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for contour in contours:
            area = cv2.contourArea(contour) * 4.0
            if not c["pot_min_area"] <= area <= c["pot_max_area"]:
                continue
            x, y, w, h = (2 * v for v in cv2.boundingRect(contour))
            if not 0.4 <= w / float(h) <= 2.5 or area < 0.35 * w * h:
                continue                            # a streak or a frame, not a pot
            left, right = self.from_robot((x, y + h)), self.from_robot((x + w, y + h))
            if left is not None and right is not None:
                if not c["pot_min_cm"] <= math.dist(left, right) <= c["pot_max_cm"]:
                    continue                        # too small or too big on the table to be a pot
            pots.append(Pot((x, y, w, h), area))
        return pots

    def _measure(self, seen):
        """Where the pots stand, and with the tag in view which of them stand at the centre."""
        c = self.cfg
        tag = seen.tag
        for pot in seen.pots:
            pot.robot = self.from_robot(pot.base)
            pot.cm = self.on_table(tag, pot.base)
            pot.exact = pot.cm is not None
            pot.home = pot.exact and math.hypot(*pot.cm) <= c["home_radius_cm"]
        grew = True
        while grew:                                 # pots standing next to collected pots are collected too
            grew = False
            for pot in seen.pots:
                if pot.home or not pot.exact:
                    continue
                if any(o.home and math.dist(o.cm, pot.cm) <= c["home_chain_cm"] for o in seen.pots):
                    pot.home = grew = True
        if seen.drop is not None:
            seen.blocked = any(p.home and math.dist(p.cm, seen.drop) <= c["pot_clear_cm"] for p in seen.pots)

    # -- table top
    def learn_table(self, seen):
        """Remember the colour of the table top in front of the feet. None if it cannot be seen there."""
        h, w = seen.lab.shape[:2]
        rows, cols = slice(int(h * 0.70), int(h * 0.97)), slice(int(w * 0.25), int(w * 0.75))
        pixels = seen.lab[rows, cols][seen.white[rows, cols] == 0]
        if len(pixels) < 300:
            self.table = None
            return None
        self.table = np.median(pixels, axis=0)
        return self.table

    def table_looks_like_pot(self):
        c = self.cfg
        return self.table is not None and all(lo <= v <= hi for lo, v, hi in
                                              zip(c["pot_lab_min"], self.table, c["pot_lab_max"]))

    def _check_table(self, seen):
        c = self.cfg
        if self.table is None or not c["table_check"]:
            return
        dark, bright, da, db = c["table_tol"]
        L0, a0, b0 = (float(v) for v in self.table)
        L, a, b = cv2.split(seen.lab.astype(np.float32))
        # In shadow a surface keeps its colour but loses colourfulness along with brightness, so the
        # a and b the table should have are scaled with L before they are compared.
        k = (L + 41.0) / (L0 + 41.0)
        table = ((L >= L0 - dark) & (L <= L0 + bright)
                 & (np.abs(a - 128.0 - (a0 - 128.0) * k) <= da) & (np.abs(b - 128.0 - (b0 - 128.0) * k) <= db))
        seen.table = table.astype(np.uint8) * 255
        # pots, with the dark inside of their opening, and a little room round them
        seen.other = np.zeros_like(seen.white)
        for contour in cv2.findContours(seen.white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]:
            cv2.fillConvexPoly(seen.other, cv2.convexHull(contour), 255)
        seen.other = cv2.dilate(seen.other, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        if seen.tag is not None:
            cv2.fillConvexPoly(seen.other, (self._grown(seen.tag.corners, 1.6) / 2.0).astype(np.int32), 255)
            seen.tag.gap = self.gap_on_way(seen, (seen.tag.x, seen.tag.y))
        for pot in seen.pots:
            pot.off_table = (self.gap_on_way(seen, pot.base) >= c["table_gap_px"]
                             or not self._table_beside(seen, pot))

    @staticmethod
    def _table_beside(seen, pot):
        """A pot stands on the table: left and right of its foot the picture shows table top."""
        x, y, w, h = (v // 2 for v in pot.box)
        width = seen.table.shape[1]
        rows = slice(y + h - max(3, h // 4), y + h + 1)
        for cols in (slice(max(0, x - 8), max(0, x - 3)), slice(min(width, x + w + 3), min(width, x + w + 8))):
            ground = seen.other[rows, cols] == 0
            if ground.sum() >= 6 and 2 * ((seen.table[rows, cols] > 0) & ground).sum() < ground.sum():
                return False
        return True

    def side_clear(self, seen, side):
        """Before a step sideways: does the picture show table top in the near corner on that side?"""
        if seen.table is None:
            return True
        h, w = seen.table.shape
        x = self.cfg["center_x"] // 2
        if side == "right":
            cols = slice(min(w - 1, x + 50), min(w, x + 140))
        else:
            cols = slice(max(0, x - 140), max(1, x - 50))
        rows = slice(int(h * 0.62), int(h * 0.97))
        ground = seen.other[rows, cols] == 0
        return ground.sum() < 50 or ((seen.table[rows, cols] > 0) & ground).sum() >= 0.8 * ground.sum()

    def gap_on_way(self, seen, point):
        """Longest stretch, in pixels, between the feet and `point` that does not look like table top."""
        if seen.table is None:
            return 0
        h, w = seen.table.shape
        x0, y0 = self.cfg["center_x"] / 2.0, min(h - 1, self.cfg["feet_y"] / 2.0)
        x1, y1 = point[0] / 2.0, min(point[1] / 2.0, y0)
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        t = np.linspace(0.0, 1.0, n)
        xs = np.rint(x0 + (x1 - x0) * t).astype(int)
        ys = np.clip(np.rint(y0 + (y1 - y0) * t).astype(int), 0, h - 1)
        cols = np.clip(xs[:, None] + np.arange(-6, 7)[None, :], 0, w - 1)      # a strip 13 pixels wide
        ground = seen.other[ys[:, None], cols] == 0
        table = (seen.table[ys[:, None], cols] > 0) & ground
        ground, table = ground.sum(axis=1), table.sum(axis=1)
        worst = run = 0
        for k in range(n):
            if ground[k] < 4:                       # a pot or the tag: says nothing about the table
                continue
            if 2 * table[k] >= ground[k]:
                run = 0
            else:
                run += 1
                worst = max(worst, run)
        return 2 * worst

    # -- pictures for people
    def draw(self, seen, text="", masks=False):
        c = self.cfg
        img = seen.frame.copy()
        if masks:
            tint = np.zeros_like(img)
            if seen.table is not None:
                tint[cv2.resize(seen.table, SIZE, interpolation=cv2.INTER_NEAREST) > 0] = (0, 150, 0)
            tint[cv2.resize(seen.white, SIZE, interpolation=cv2.INTER_NEAREST) > 0] = (255, 0, 255)
            img = cv2.addWeighted(img, 0.6, tint, 0.4, 0)
        cv2.line(img, (c["center_x"], 0), (c["center_x"], SIZE[1]), (255, 200, 0), 1)
        for row in (c["grab_y"], c["too_close_y"]):
            cv2.line(img, (c["center_x"] - c["side_small_px"], row), (c["center_x"] + c["side_small_px"], row),
                     (255, 200, 0), 2)
        cv2.drawMarker(img, tuple(int(v) for v in c["drop_point"]), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for pot in seen.pots:
            x, y, w, h = pot.box
            if pot.home or pot.maybe_home:
                colour, label = (160, 160, 160), "collected"
            elif pot.off_table:
                colour, label = (0, 0, 255), "not on the table"
            else:
                colour, label = (0, 220, 0), "pot %d,%d" % (pot.x, pot.y)
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(img, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        if seen.tag is not None:
            cv2.polylines(img, [seen.tag.corners.astype(np.int32)], True, (0, 255, 255), 2)
            label = "tag %d" % seen.tag.id
            if seen.gap_cm is not None:
                label += "  %.0f cm" % seen.gap_cm
            cv2.putText(img, label, (seen.tag.x + 8, seen.tag.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if text:
            cv2.putText(img, text, (8, SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, text, (8, SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img


# ----------------------------------------------------------------- the robot
def service_running(name="tonypi"):
    """True when the robot's app service runs (it shares the serial port and the camera)."""
    import subprocess
    try:
        r = subprocess.run(["systemctl", "is-active", "--quiet", name], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


class _Stream:
    """Newest picture of a video stream. Reading all the time keeps old pictures from piling up."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            ok, img = self.cap.read()
            if ok:
                self.frame = img
            else:
                time.sleep(0.05)

    def read(self):
        return self.frame is not None, self.frame

    def camera_close(self):
        self.alive = False
        time.sleep(0.2)
        self.cap.release()


class Robot:
    """The real TonyPi: action groups, head, buzzer, camera, picture window."""

    WALKS = {"go_forward": "go", "go_forward_fast": "go", "go_forward_slow": "go", "back": "back", "back_fast": "back"}

    def __init__(self, cfg, window=True):
        import hiwonder.ActionGroupControl as AGC
        self.cfg = cfg
        self.AGC = AGC
        self.board = AGC.board
        self.gait = None
        self.window = window and bool(os.environ.get("DISPLAY"))
        if service_running():
            print("Note: the tonypi service is running. If the robot moves strangely, stop it first:")
            print("      sudo systemctl stop tonypi")
        self.cam = self._open_camera()
        if self.window:
            cv2.namedWindow(WINDOW)

    @staticmethod
    def _open_camera():
        if service_running():                       # the service owns the camera and passes it on as a stream
            stream = _Stream(cv2.VideoCapture(STREAM_URL))
            deadline = time.time() + 3
            while time.time() < deadline and stream.frame is None:
                time.sleep(0.05)
            if stream.frame is not None:
                return stream
            stream.camera_close()
        import hiwonder.Camera as Camera
        cam = Camera.Camera()
        cam.camera_open()
        return cam

    def head(self):
        """Point the head the way the robot's own games do: forward and down at the floor in front."""
        tilt, pan = self.cfg["head_tilt"], self.cfg["head_pan"]
        if tilt is None or pan is None:
            import yaml
            with open(SERVO_YAML, "r", encoding="utf-8") as fp:
                servo = yaml.safe_load(fp)
            tilt = servo["servo1"] if tilt is None else tilt
            pan = servo["servo2"] if pan is None else pan
        self.board.pwm_servo_set_position(0.5, [[1, int(tilt)], [2, int(pan)]])
        time.sleep(0.6)

    def act(self, name, times=1, lock=None):
        """Run an action group. `lock` = arm servos that keep their place (while a pot is carried)."""
        gait = self.WALKS.get(name)
        if gait and self.gait and gait != self.gait:        # finish walking one way before walking the other
            self.AGC.runActionGroup("stand", lock_servos=lock or "", path=ACTION_DIR)
        self.gait = gait
        self.AGC.runActionGroup(name, times=times, lock_servos=lock or "", path=ACTION_DIR)

    def frame(self, settle=None):
        """A picture taken after the robot has come to rest."""
        time.sleep(self.cfg["settle_s"] if settle is None else settle)
        deadline = time.time() + 8
        while time.time() < deadline:
            ok, img = self.cam.read()
            if ok and img is not None:
                return img.copy()
            time.sleep(0.05)
        raise RuntimeError("no picture from the camera")

    def beep(self):
        try:
            self.board.set_buzzer(1900, 0.1, 0.9, 1)
        except Exception:
            pass

    def wait(self, seconds):
        time.sleep(seconds)

    def show(self, img):
        """Show the picture. Returns the key pressed in the window (or -1)."""
        if not self.window:
            return -1
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(1)
        return -1 if key == -1 else key & 0xFF

    def close(self):
        try:
            self.cam.camera_close()
        except Exception:
            pass
        if self.window:
            cv2.destroyAllWindows()


# ----------------------------------------------------------------- what the robot does
class Collector:
    def __init__(self, robot, vision, cfg, say=print):
        self.robot = robot
        self.vision = vision
        self.cfg = cfg
        self.say = say
        self.lock = None            # arm servos held while a pot is carried
        self.carrying = False
        self.delivered = 0
        self.failures = 0
        self.stop = False           # set by Ctrl-C or q: finish the movement, then stop
        self.no_tag = False         # the tag cannot be found: nothing can be done
        self.status = ""
        self._told = ""
        # Where the robot stands (cm from the tag) and which way it faces. Exact whenever the tag is in
        # view; in between it is reckoned from the steps taken, and `loose` says how far off that may be.
        self.pose = None
        self.loose = [0.0, 0.0]     # cm, radians
        self.steps = {name: list(step) for name, step in cfg["steps"].items()}
        self._measured = set()      # actions whose step has been measured at the tag
        self._fix = None            # the pose when the tag was last in view...
        self._since = []            # ...and the steps taken since then: [(action, times)]
        self.spots = []             # where it has put pots down, in cm from the tag
        self.shy = ["", 0]          # side on which a pot was just passed, and for how many more movements

    # -- seeing and moving
    def look(self):
        seen = self.vision.see(self.robot.frame())
        self.locate(seen)
        if self.robot.show(self.vision.draw(seen, self.status)) in (27, ord("q")):
            self.say("Stopping (q pressed).")
            self.stop = True
        return seen

    def locate(self, seen):
        """Keep track of where the robot stands, and settle which pots stand at the centre already."""
        c = self.cfg
        if seen.pose is not None:
            self._learn_step(seen.pose)
            self.pose, self.loose = list(seen.pose), [0.0, 0.0]
            self._fix, self._since = seen.pose, []
        for pot in seen.pots:
            if pot.exact:
                pot.home = pot.home or any(math.dist(pot.cm, s) <= c["spot_cm"] for s in self.spots)
            elif self.pose is not None and pot.robot is not None:
                pot.cm = self.to_table(pot.robot)       # tag out of view: reckon where the pot stands
                slack = c["guess_cm"] + self.loose[0] + self.loose[1] * math.hypot(*pot.robot)
                pot.maybe_home = (math.hypot(*pot.cm) <= c["home_radius_cm"] + slack
                                  or any(math.dist(pot.cm, s) <= c["spot_cm"] + slack for s in self.spots))
        if seen.drop is not None and any(math.dist(seen.drop, s) <= c["pot_clear_cm"] for s in self.spots):
            seen.blocked = True

    def to_table(self, spot):
        """(cm forward, cm left) of the robot -> cm from the tag."""
        x, y, heading = self.pose
        cos, sin = math.cos(heading), math.sin(heading)
        return x + spot[0] * cos - spot[1] * sin, y + spot[0] * sin + spot[1] * cos

    def to_robot(self, place):
        """cm from the tag -> (cm forward, cm left) of the robot."""
        x, y, heading = self.pose
        cos, sin = math.cos(heading), math.sin(heading)
        return (place[0] - x) * cos + (place[1] - y) * sin, -(place[0] - x) * sin + (place[1] - y) * cos

    def _learn_step(self, pose):
        """The tag is in view before and after one kind of step: measure what that step really does."""
        if self._fix is None or not self._since or len(set(name for name, _ in self._since)) != 1:
            return
        name, count = self._since[0][0], sum(times for _, times in self._since)
        x, y, heading = self._fix
        cos, sin = math.cos(heading), math.sin(heading)
        done = ((pose[0] - x) * cos + (pose[1] - y) * sin, -(pose[0] - x) * sin + (pose[1] - y) * cos,
                math.degrees(wrap(pose[2] - heading)))
        guess = self.steps[name]
        k = max(range(3), key=lambda i: abs(guess[i]))          # what this action is for
        value = done[k] / count
        if value * guess[k] > 0 and 0.3 <= value / guess[k] <= 3.0:
            guess[k] = value if name not in self._measured else 0.5 * (guess[k] + value)
            self._measured.add(name)

    def act(self, name, times=1):
        self.robot.act(self.cfg.get(name, name), times, lock=self.lock)
        step = self.steps.get(name)
        if step is None:
            return
        self._since.append((name, times))
        if self.pose is not None:
            for _ in range(times):
                self.pose[0], self.pose[1] = self.to_table(step[:2])
                self.pose[2] += math.radians(step[2])
        self.loose[0] += 0.25 * math.hypot(step[0], step[1]) * times
        self.loose[1] += (0.15 * abs(math.radians(step[2])) + (0.02 if step[0] or step[1] else 0.0)) * times

    def move(self, what, name, times=1):
        self.shy[1] -= 1
        self.say("  %-30s -> %s%s" % (what, self.cfg.get(name, name), " x%d" % times if times > 1 else ""))
        self.act(name, times)

    def back(self, what, times=1):
        self.move(what, "back", times)
        self.act("stand")

    def sidestep(self, seen, what, side, big=True):
        """Step sideways, but only if the picture shows table top on that side and no pot was just passed there."""
        if not self.vision.side_clear(seen, side) or (self.shy[0] == side and self.shy[1] > 0):
            return False
        self.move(what, side + "_big" if big else side)
        return True

    def in_the_way(self, seen, target=None):
        """A pot the robot would walk into (not the one it is fetching): how far it is to the left, or None."""
        c = self.cfg
        reach = None if target is None or target.robot is None else target.robot[0] - 4.0
        things = [pot.robot for pot in seen.pots if pot is not target and pot.robot is not None]
        if self.pose is not None:           # a pot it has put down may be too close to its feet to be in the picture
            things += [self.to_robot(spot) for spot in self.spots]
        for forward, left in things:
            if (0.0 < forward <= c["clear_ahead_cm"] and abs(left) <= c["clear_side_cm"]
                    and (reach is None or forward < reach)):
                return left
        return None

    def step_round(self, seen, where, left):
        """Step sideways away from a pot that is in the way. False if there is no table top to step on."""
        side = "right" if left >= 0 else "left"
        if self.sidestep(seen, where + ": pot in the way", side):
            self.shy = ["left" if side == "right" else "right", 6]      # and do not step back towards it at once
            return True
        self.say("  a pot is in my way and I see no table top to step round it.")
        return False

    def target(self, seen):
        """The pot to fetch: the nearest one that is not at the centre and stands on this table."""
        best, best_d = None, 0
        for pot in seen.pots:
            if pot.home or pot.maybe_home or pot.off_table:
                continue
            d = (pot.x - SIZE[0] / 2) ** 2 + (pot.y - SIZE[1]) ** 2
            if best is None or d < best_d:
                best, best_d = pot, d
        return best

    def same_pot(self, seen, last):
        """The pot that was at `last` in the picture one look ago."""
        near = [p for p in seen.pots if math.dist((p.x, p.y), last) <= self.cfg["track_px"]]
        return min(near, key=lambda p: math.dist((p.x, p.y), last)) if near else None

    def tell(self, text):
        """Say something that would otherwise be repeated at every look only once."""
        if text != self._told:
            self._told = text
            if text:
                self.say(text)

    # -- the steps of one trip
    def get_ready(self):
        c = self.cfg
        self.status = "getting ready"
        self.robot.head()
        self.act("stand_slow")
        if not c["table_check"]:
            self.say("NOT checking for table top: keep a hand near the robot.")
            return True
        for _ in range(3):
            colour = self.vision.learn_table(self.look())
            if colour is not None:
                break
            self.robot.wait(1.0)
        if colour is None:
            self.say("I cannot see the table top in front of my feet (is something white there?).")
            self.say("Clear that spot and start again.")
            return False
        self.say("Table top colour (L a b): %d %d %d" % tuple(int(v) for v in colour))
        if self.vision.table_looks_like_pot():
            self.say("The table top looks as white as a pot to me. I cannot tell pots from this table:")
            self.say("use a darker surface, or teach the pot colour with --view.")
            return False
        return True

    def find_tag(self):
        """Turn on the spot until the tag is in view (hands empty: until the robot knows where it stands)."""
        c = self.cfg
        self.status = "looking for the tag"
        turned = 0.0
        most = 400.0 if "turn_right" in self._measured else 600.0       # a full turn, generously
        degrees = abs(self.steps["turn_right"][2]) * c["search_turn_steps"]
        while not self.stop:
            seen = self.look()
            if seen.tag is not None and (self.carrying or seen.pose is not None):
                return True
            if turned >= most:
                break
            self.act("turn_right", c["search_turn_steps"])
            turned += degrees
        return False

    def spot(self):
        """Look, twice (no walking after a reflection): a pot to fetch, or None."""
        seen = self.look()
        pot = self.target(seen)
        if pot is not None:
            again = self.target(self.look())
            if again is not None and abs(again.x - pot.x) < 60 and abs(again.y - pot.y) < 60:
                self.say("Pot in sight at (%d, %d)." % (again.x, again.y))
                return again
        off_table = [p for p in seen.pots if not p.home and not p.maybe_home and p.off_table]
        self.tell("  something white at (%d, %d), but not on table top I can walk over: left alone"
                  % (off_table[0].x, off_table[0].y) if off_table else "")
        return None

    def find_pot(self):
        """Look for a pot to fetch: straight ahead, then half-way round to either side. None = none."""
        c = self.cfg
        if (self.pose is None or self.loose[0] > 10.0) and not self.find_tag():
            if not self.stop:
                self.no_tag = True
            return None
        self.status = "looking for a pot"
        pot = self.spot()
        # half-way round to the right, all the way back and on to the left, then back to face the tag
        for side, most in (("turn_right", 180.0), ("turn_left", 360.0), ("turn_right", 180.0)):
            turned = 0.0
            while pot is None and not self.stop and turned < most:
                self.act(side, c["search_turn_steps"])
                turned += abs(self.steps[side][2]) * c["search_turn_steps"]
                pot = self.spot()
        return pot

    def wait_for_pot(self):
        """Stand still and watch until a pot appears."""
        c = self.cfg
        self.status = "waiting for a pot"
        self.say("No pot to fetch. Put one on the table where I can see it.")
        self.act("stand")
        waited = 0.0
        while not self.stop and self.target(self.look()) is None:
            self.robot.wait(0.6)
            waited += 1.0
            if c["rescan_s"] and waited >= c["rescan_s"]:
                break

    def walk_to_pot(self, pot):
        """Walk until the pot stands where the arms can take it. False if it was lost or is not to be fetched."""
        c = self.cfg
        self.status = "walking to the pot"
        last, foot = (pot.x, pot.y), pot.base
        missing = unseen = 0
        for _ in range(c["max_moves"]):
            if self.stop:
                return False
            seen = self.look()
            pot = self.same_pot(seen, last)
            if pot is None and last[1] < c["far_y"] and unseen < 5:
                # It was far away, at the top of the picture, and slipped out of view as the robot turned
                # to it. Go on towards where it was, as long as there is table top all the way there.
                unseen += 1
                dx = last[0] - c["center_x"]
                if abs(dx) > c["turn_px"]:
                    self.move("pot out of view at the top", "turn_right" if dx > 0 else "turn_left")
                    last = (last[0] - math.copysign(c["turn_shift_px"], dx), last[1])
                    foot = (foot[0] - math.copysign(c["turn_shift_px"], dx), foot[1])
                    continue
                blocked = self.in_the_way(seen)
                if blocked is not None and self.step_round(seen, "pot out of view at the top", blocked):
                    continue
                if blocked is None and self.vision.gap_on_way(seen, foot) < c["table_gap_px"]:
                    self.move("pot out of view at the top", "forward", 2)
                    last, foot = (last[0], last[1] + 25), (foot[0], foot[1] + 25)
                    continue
            if pot is not None and pot.home:
                self.say("  that pot stands at the centre already.")
                self.spots.append(pot.cm)           # so that it is not taken for a new pot again
                return False
            if pot is None or pot.off_table:
                missing += 1
                if missing >= c["lost_looks"]:
                    self.say("  I lost the pot." if pot is None else "  no table top on the way to the pot.")
                    self.failures += 1
                    return False
                continue
            missing = 0
            last, foot = (pot.x, pot.y), pot.base
            where = "pot at (%d, %d)" % last
            dx = pot.x - c["center_x"]
            side = "right" if dx > 0 else "left"
            blocked = self.in_the_way(seen, pot)
            if abs(dx) > c["turn_px"]:                      # face it first
                self.move(where, "turn_" + side)
            elif pot.y > c["too_close_y"]:
                self.back(where + ": too close")
            elif blocked is not None:
                if not self.step_round(seen, where, blocked):
                    break
            elif pot.y <= c["near_y"]:                      # still far: walk
                self.move(where, "forward", 2 if pot.y < c["near_y"] - 100 else 1)
            elif abs(dx) >= c["side_small_px"]:             # close: line up sideways, then creep in
                if not self.sidestep(seen, where, side, abs(dx) >= c["side_big_px"]):
                    self.move(where, "turn_" + side)        # no table top that side: turn instead
            elif pot.y <= c["step_y"]:
                self.move(where, "forward")
            elif pot.y <= c["grab_y"]:
                self.move(where, "one_step")
            else:
                self.say("  %s: in reach." % where)
                return True
        self.say("  I do not get any closer to that pot.")
        self.failures += 1
        return False

    def pick_up(self):
        """Bend down, take the pot, and check that it is no longer standing on the table."""
        c = self.cfg
        for attempt in range(c["grab_tries"]):
            self.status = "picking the pot up"
            self.act("one_step", c["grab_steps"])
            self.act("stand")
            self.say("  picking it up (%s)" % c["pick"])
            self.act("pick")
            self.lock = dict(c["carry_lock"])
            self.carrying = True
            left = [p for p in self.look().pots
                    if not p.home and p.y > c["missed_y"] and abs(p.x - c["center_x"]) < c["missed_dx"]]
            if not left:
                self.failures = 0
                return True
            self.say("  the pot is still on the table: I missed it.")
            self.put_down(steps=0)                          # arms back down
            self.back("stepping back to try again", 2)
            pot = self.target(self.look())
            if attempt + 1 == c["grab_tries"] or pot is None or not self.walk_to_pot(pot):
                break
        self.failures += 1
        return False

    def walk_to_tag(self):
        """
        Walk until the pot would be put down just in front of the tag, or of the pots already there.
        Returns that place (cm from the tag), or None if the robot cannot get there.
        """
        c = self.cfg
        self.status = "carrying the pot to the centre"
        want = c["tag_size_cm"] / 2.0 + c["drop_gap_cm"]
        missing = backs = 0
        for _ in range(c["max_moves"]):
            if self.stop:
                return None
            seen = self.look()
            tag = seen.tag
            if tag is None:
                missing += 1
                if missing >= c["lost_looks"]:
                    self.say("  I lost the tag.")
                    if not self.find_tag():
                        return None
                    missing = 0
                continue
            missing = 0
            gap = seen.gap_cm
            where = "tag at (%d, %d)" % (tag.x, tag.y) + ("" if gap is None else ", %.0f cm" % gap)
            dx = tag.x - c["center_x"]
            side = "right" if dx > 0 else "left"
            blocked = self.in_the_way(seen)
            if abs(dx) > c["turn_px"]:                      # face it first
                self.move(where, "turn_" + side)
            elif blocked is not None:
                if not self.step_round(seen, where, blocked):
                    return None
            elif gap is not None and gap < want - c["drop_tol_cm"] and backs < 8:
                self.back(where + ": too close")
                backs += 1
            elif gap is not None and (gap <= want or seen.blocked):
                self.say("  %s: this is the place%s." % (where, " (next to the pots already here)"
                                                         if gap > want else ""))
                return seen.drop
            elif tag.y > c["near_y"] and abs(dx) >= c["side_small_px"]:
                if not self.sidestep(seen, where, side, abs(dx) >= c["side_big_px"]):
                    self.move(where, "turn_" + side)
            elif tag.gap >= c["table_gap_px"]:
                self.say("  %s, but I see no table top on the way to it." % where)
                return None
            elif gap is None or gap > want + 10:
                self.move(where, "forward")
            else:
                self.move(where, "one_step")
        self.say("  I do not get any closer to the tag.")
        return None

    def put_down(self, steps=None):
        c = self.cfg
        self.status = "putting the pot down"
        steps = c["put_steps"] if steps is None else steps
        if steps:
            self.act("one_step", steps)
        self.act("stand")
        self.lock = None                    # the put-down action moves the arms itself
        self.act("put")
        self.carrying = False

    def leave(self):
        """Step back from the pot just put down, so that turning does not knock it over."""
        self.back("stepping back from the centre", self.cfg["leave_steps"])

    def finish(self):
        try:
            if self.carrying:
                self.say("Putting the pot down before I stop.")
                self.put_down(steps=0)
            self.act("stand")
        except Exception as e:
            self.say("Could not come to a tidy stop: %s" % e)

    # -- the programs
    def run(self, once=False, wait=True):
        """Collect pots until stopped. wait=False: return when there is no pot left to fetch."""
        c = self.cfg
        try:
            if not self.get_ready():
                return
            while not self.stop:
                pot = self.find_pot()
                if pot is None:
                    if self.no_tag:
                        self.say("I cannot see the tag at the centre of the table from here. Stand me about")
                        self.say("40 cm from it, facing it, and check with --view that I recognise it.")
                    if self.stop or self.no_tag or not wait:
                        break
                    self.wait_for_pot()
                    continue
                self.robot.beep()
                if not (self.walk_to_pot(pot) and self.pick_up()):
                    if self.failures >= c["max_failures"]:
                        self.say("I could not fetch a pot %d times in a row. Stopping: try --grab-test." % self.failures)
                        break
                    continue
                place = self.walk_to_tag() if self.find_tag() else None
                if place is None:
                    if not self.stop:
                        self.say("I cannot get to the tag at the centre of the table. Stopping.")
                    break
                self.say("  putting it down (%s)" % c["put"])
                self.put_down()
                self.spots.append(place)
                self.delivered += 1
                self.say("Pot number %d is at the centre." % self.delivered)
                self.robot.beep()
                self.leave()
                if once:
                    break
        except Exception:
            traceback.print_exc()
        finally:
            self.finish()
            self.say("Done: %d pot%s collected." % (self.delivered, "" if self.delivered == 1 else "s"))

    def grab_test(self):
        """Without walking: wait until a pot stands where the arms can take it, pick it up, put it down."""
        c = self.cfg
        c["table_check"] = False
        c["grab_tries"] = 1
        try:
            self.get_ready()
            self.say("Put a pot in front of the robot. It has to appear between the two short blue lines.")
            steady = 0
            while not self.stop and steady < 2:
                self.status = "grab test: waiting for a pot between the blue lines"
                pot = self.target(self.look())
                if pot is None:
                    self.tell("  no pot in sight")
                    steady = 0
                    continue
                dx = pot.x - c["center_x"]
                if abs(dx) >= c["side_small_px"]:
                    hint = "move it %d pixels to the %s in the picture" % (abs(dx), "left" if dx > 0 else "right")
                elif pot.y <= c["grab_y"]:
                    hint = "bring it closer to the robot"
                elif pot.y > c["too_close_y"]:
                    hint = "move it away from the robot"
                else:
                    hint = ""
                self.tell("  pot at (%d, %d): %s" % (pot.x, pot.y, hint or "good, hands off"))
                steady = 0 if hint else steady + 1
                self.robot.wait(0.5)
            if self.stop:
                return
            self.robot.beep()
            self.robot.wait(1.5)
            if self.pick_up():
                self.say("I think I hold it. Putting it down again in 3 seconds.")
                self.robot.wait(3.0)
                self.put_down(steps=0)
        except Exception:
            traceback.print_exc()
        finally:
            self.finish()


# ----------------------------------------------------------------- looking without walking
def learn_pot_colour(cfg, seen, x, y, path=CONFIG_YAML):
    """Set the pot colour from the pixels round a click, and store it in pot_config.yaml."""
    patch = seen.lab[max(0, y // 2 - 4): y // 2 + 5, max(0, x // 2 - 4): x // 2 + 5].reshape(-1, 3).astype(int)
    lo, hi = patch.min(axis=0), patch.max(axis=0)
    cfg["pot_lab_min"] = [int(max(0, lo[0] - 30)), int(max(0, lo[1] - 10)), int(max(0, lo[2] - 10))]
    cfg["pot_lab_max"] = [255 if lo[0] > 150 else int(min(255, hi[0] + 30)),
                          int(min(255, hi[1] + 10)), int(min(255, hi[2] + 10))]
    save_settings({"pot_lab_min": cfg["pot_lab_min"], "pot_lab_max": cfg["pot_lab_max"]}, path)
    print("Pot colour is now LAB %s to %s (saved in pot_config.yaml)." % (cfg["pot_lab_min"], cfg["pot_lab_max"]))


def view(robot, vision, cfg, collector):
    print("Looking only. Click on a pot = learn its colour, m = show pot (pink) and table (green),")
    print("t = learn the table colour again, q = quit.")
    robot.head()
    robot.act("stand_slow")
    vision.learn_table(vision.see(robot.frame()))
    clicks = []
    if robot.window:
        cv2.setMouseCallback(WINDOW, lambda event, x, y, flags, param:
                             clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
    else:
        print("No window: the picture is written to %s once a second." % VIEW_JPG)
    masks, told = True, 0.0
    while not collector.stop:
        seen = vision.see(robot.frame(settle=0.03))
        collector.locate(seen)
        if clicks:
            learn_pot_colour(cfg, seen, *clicks[-1])
            clicks.clear()
            continue
        pot = collector.target(seen)
        text = "no pot" if pot is None else "pot at (%d, %d)" % (pot.x, pot.y)
        text += ", %d collected" % sum(p.home or p.maybe_home for p in seen.pots)
        text += ", no tag" if seen.tag is None else ", tag %d" % seen.tag.id
        if seen.gap_cm is not None:
            text += " %.0f cm from the cross" % seen.gap_cm
        if vision.table is None:
            text += ", TABLE NOT SEEN"
        img = vision.draw(seen, text, masks)
        key = robot.show(img)
        if key in (27, ord("q")):
            break
        if key == ord("m"):
            masks = not masks
        if key == ord("t"):
            print("Table top colour (L a b): %s" % vision.learn_table(seen))
        if time.time() - told > 1.0:
            told = time.time()
            print(text)
            if not robot.window:
                cv2.imwrite(VIEW_JPG, img)


# ----------------------------------------------------------------- start
def main():
    parser = argparse.ArgumentParser(description="TonyPi collects white pots at the centre of its table.")
    parser.add_argument("--view", action="store_true", help="only look: show what the robot sees")
    parser.add_argument("--grab-test", action="store_true", help="pick up and put down a pot without walking")
    parser.add_argument("--once", action="store_true", help="stop after one pot")
    parser.add_argument("--tag", default=None, help="number of the tag at the centre, or 'any'")
    parser.add_argument("--no-table-check", action="store_true", help="walk without checking for table top")
    parser.add_argument("--no-window", action="store_true", help="no picture window")
    args = parser.parse_args()

    cfg = load_settings()
    if args.tag is not None:
        cfg["tag_id"] = None if args.tag == "any" else int(args.tag)
    if args.no_table_check:
        cfg["table_check"] = False
    missing = missing_actions(cfg)
    if missing:
        print("These action groups are missing in %s: %s" % (ACTION_DIR, ", ".join(missing)))
        sys.exit(1)

    try:
        robot = Robot(cfg, window=not args.no_window)
    except Exception as e:
        print("Cannot reach the robot's controller board or camera: %s" % (e or type(e).__name__))
        print("This program runs on the robot itself (pot_selftest.py is the one for a PC).")
        sys.exit(1)
    vision = Vision(cfg)
    collector = Collector(robot, vision, cfg)

    def on_signal(signum, frame):
        if collector.stop:
            os._exit(1)
        collector.stop = True
        print("\nStopping after this movement (Ctrl-C again quits at once).")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        if args.view:
            view(robot, vision, cfg, collector)
        elif args.grab_test:
            collector.grab_test()
        else:
            collector.run(once=args.once)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
