#!/usr/bin/python3
# coding=utf8
"""
A pretend robot, table and camera for testing TonyPi chess on a PC.

Nothing in here runs on the robot during a game. chess_selftest.py uses it
to exercise the real vision, arm and game code against something that
behaves like the real thing in the ways that matter:

  camera   pinhole camera on the robot's head looking down at an angle;
           pieces are drawn as upright cylinders, so they lean over the
           squares behind them exactly as real pieces do. Optional lens
           distortion, brightness / contrast changes and sensor noise.
  board    can be shifted and turned on the table, and set up with white
           or black on the robot's side.
  arms     two 3-joint arms with a gripper (made-up but plausible
           lengths). Poses are "taught" by solving for the servo values
           that put the gripper over a square, the way a person would by
           moving the arm there by hand.
  physics  closing the gripper picks up a piece if the gripper is around
           it; opening puts it down where the gripper is. Hitting a
           standing piece, dropping from too high, missing the square or
           using the camera while an arm is over the board are recorded
           as errors.

All lengths are in squares. The table top (board surface) is z = 0; x is
to the robot's right, y is forward, z is up. The robot stands at x = y = 0.
"""
import numpy as np
import cv2
import chess

import chess_vision as vision
import chess_arm

IMAGE_SIZE = (640, 480)
KING_HEIGHT = 1.6
HEIGHT = {p: KING_HEIGHT * rel for p, rel in vision.PIECE_REL_HEIGHT.items()}
RADIUS = 0.28

GRASP_Z = 0.7                   # height at which the pretend person teaches "grasp"
HOVER_Z = 2.8                   # ...and "hover"
GRAB_TOL = 0.30                 # how far off-centre the gripper may be and still catch the piece
PLACE_TOL = 0.30
JAW_INSIDE = 0.36               # an open gripper closer than this has the piece between its jaws
JAW_REACH = 0.60                # anything lower than the gripper and closer than this gets hit

DEG_PER_PULSE = np.array([0.24, 0.24, 0.36])      # shoulder, upper arm, elbow
SHOULDER_Z = 4.5
SHOULDER_X = 2.2
UPPER, FOREARM = 2.8, 4.4

# gripper: pulse -> opening in squares
GRIPPER = {
    "right": {"servo": 17, "closed": 500, "open": 760},
    "left": {"servo": 18, "closed": 500, "open": 240},
}
JOINTS = {"right": (16, 15, 14), "left": (8, 7, 6)}
REST = {"right": (275, 200, 425), "left": (725, 800, 575)}
BIN = {"right": (5.8, 4.0), "left": (-5.8, 4.0)}
BIN_RADIUS = 1.0


# ---------------------------------------------------------------- camera
class SimCamera:
    def __init__(self, position=(0.0, 0.5, 8.0), pitch=57.0, f=425.0, size=IMAGE_SIZE, distortion=None,
                 K=None):
        self.size = size
        self.K = np.array([[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1]], np.float64)
        if K is not None:
            self.K = np.array(K, np.float64)
        self.C = np.array(position, np.float64)
        a = np.radians(pitch)
        zc = np.array([0.0, np.cos(a), -np.sin(a)])       # optical axis: forward and down
        xc = np.array([1.0, 0.0, 0.0])
        yc = np.cross(zc, xc)
        self.R = np.vstack([xc, yc, zc])
        self.distortion = None if distortion is None else np.array(distortion, np.float64)
        self._dist_map = None

    def project(self, pts):
        pts = np.array(pts, np.float64).reshape(-1, 3)
        q = (pts - self.C) @ self.R.T @ self.K.T
        return q[:, :2] / q[:, 2:3]

    def distort(self, img):
        """What the real lens would make of an ideal image."""
        if self.distortion is None:
            return img
        if self._dist_map is None:
            w, h = self.size
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, self.K, self.distortion, P=self.K).reshape(h, w, 2)
            self._dist_map = (np.ascontiguousarray(und[..., 0]), np.ascontiguousarray(und[..., 1]))
        return cv2.remap(img, self._dist_map[0], self._dist_map[1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- arm geometry
def angles(arm, pulses):
    """
    Joint angles (radians) of the pretend arm:
      turn   about the vertical through the shoulder, positive = outwards
      raise  of the upper arm, 0 = hanging down, 90 degrees = level
      bend   of the elbow, 0 = straight
    """
    p = np.array(pulses, np.float64)
    r = np.array(REST[arm], np.float64)
    if arm == "right":
        d = np.array([p[0] - r[0], p[1] - r[1], r[2] - p[2]])
    else:
        d = np.array([r[0] - p[0], r[1] - p[1], p[2] - r[2]])
    return np.radians(d * DEG_PER_PULSE)


def pulses(arm, turn, lift, bend):
    """The servo values for the given joint angles (radians)."""
    d = np.degrees([turn, lift, bend]) / DEG_PER_PULSE
    r = np.array(REST[arm], np.float64)
    if arm == "right":
        return np.array([r[0] + d[0], r[1] + d[1], r[2] - d[2]])
    return np.array([r[0] - d[0], r[1] - d[1], r[2] + d[2]])


def tip(arm, pose):
    """Position of the gripper for the given servo values."""
    turn, lift, bend = angles(arm, pose)
    side = 1.0 if arm == "right" else -1.0
    fore = lift - bend                                   # angle of the forearm from straight down
    out = UPPER * np.sin(lift) + FOREARM * np.sin(fore)
    drop = UPPER * np.cos(lift) + FOREARM * np.cos(fore)
    return np.array([side * (SHOULDER_X + out * np.sin(turn)), out * np.cos(turn), SHOULDER_Z - drop])


def solve(arm, target, guess=None):
    """Servo values that put the gripper at `target` (what a person does by hand), or None."""
    x, y, z = (float(v) for v in target)
    side = 1.0 if arm == "right" else -1.0
    dx = side * x - SHOULDER_X
    out, drop = np.hypot(dx, y), SHOULDER_Z - z
    dist = np.hypot(out, drop)
    if dist > UPPER + FOREARM - 1e-6 or dist < abs(FOREARM - UPPER) + 1e-6 or out < 0.3:
        return None
    bend = np.arccos(np.clip((dist ** 2 - UPPER ** 2 - FOREARM ** 2) / (2 * UPPER * FOREARM), -1, 1))
    lift = np.arctan2(out, drop) + np.arctan2(FOREARM * np.sin(bend), UPPER + FOREARM * np.cos(bend))
    turn = np.arctan2(dx, y)
    p = pulses(arm, turn, lift, bend)
    if (p < 5).any() or (p > 995).any() or abs(np.degrees(turn)) > 75:
        return None
    p = np.round(p)
    if np.linalg.norm(tip(arm, p) - np.array([x, y, z])) > 0.05:
        return None
    return p.astype(int)


# ---------------------------------------------------------------- the world
class SimWorld:
    def __init__(self, robot_white=False, centre=(0.0, 6.5), angle=0.0, camera=None, seed=0):
        self.camera = camera or SimCamera()
        self.centre = np.array(centre, np.float64)
        self.angle = float(angle)                 # degrees the board is turned on the table
        self.turn = 0 if robot_white else 2       # which side of the board faces the robot
        self.rng = np.random.default_rng(seed)
        self.pieces = {}                          # square -> [symbol, dx, dy]
        self.binned = []
        self.errors = []
        self.events = []
        self.held = {}                            # arm -> [symbol, height it was grasped at]
        self.joint = {n: np.array(REST[n], np.float64) for n in chess_arm.ARM_NAMES}
        self.grip = {"right": 500.0, "left": 500.0}
        self.gain, self.offset, self.noise = 1.0, 0.0, 5
        self.contact = {}                         # arm -> gripper pulse at which it touches the piece

    # -- setting up
    def set_position(self, board, jitter=0.06):
        self.pieces = {}
        for s, p in board.piece_map().items():
            dx, dy = self.rng.uniform(-jitter, jitter, 2)
            self.pieces[s] = [p.symbol(), float(dx), float(dy)]

    def clear(self):
        self.pieces = {}

    def position(self):
        """{square: symbol} of what stands on the board."""
        return {s: v[0] for s, v in self.pieces.items()}

    def matches(self, board):
        return self.position() == {s: p.symbol() for s, p in board.piece_map().items()}

    def human_move(self, board, move, jitter=0.08):
        """A person plays `move` (board is the position BEFORE the move)."""
        after = board.copy(stack=False)
        after.push(move)
        want = {s: p.symbol() for s, p in after.piece_map().items()}
        for s in list(self.pieces):
            if want.get(s) != self.pieces[s][0]:
                del self.pieces[s]
        for s, sym in want.items():
            if s not in self.pieces:
                dx, dy = self.rng.uniform(-jitter, jitter, 2)
                self.pieces[s] = [sym, float(dx), float(dy)]

    def fix(self, board):
        """A person tidies the board so that it shows `board`."""
        want = {s: p.symbol() for s, p in board.piece_map().items()}
        self.pieces = {s: v for s, v in self.pieces.items() if want.get(s) == v[0]}
        for s, sym in want.items():
            if s not in self.pieces:
                self.pieces[s] = [sym, 0.0, 0.0]
        for v in self.pieces.values():
            v[1] = float(np.clip(v[1], -0.1, 0.1))
            v[2] = float(np.clip(v[2], -0.1, 0.1))
        self.held = {}

    # -- geometry
    def _rot(self):
        a = np.radians(self.angle + 90.0 * self.turn)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def board_to_world(self, f, r):
        """(file, rank) in 0..8 (a1 corner = 0, 0) to table coordinates."""
        return self.centre + self._rot() @ (np.array([f, r], np.float64) - 4.0)

    def world_to_board(self, x, y):
        return self._rot().T @ (np.array([x, y], np.float64) - self.centre) + 4.0

    def square_world(self, square, z=0.0):
        xy = self.board_to_world(chess.square_file(square) + 0.5, chess.square_rank(square) + 0.5)
        return np.array([xy[0], xy[1], z])

    def corners_image(self):
        """The board's outer corners as the camera sees them, ordered TL, TR, BR, BL."""
        pts = [np.append(self.board_to_world(f, r), 0.0) for f, r in ((0, 0), (8, 0), (8, 8), (0, 8))]
        return vision.order_corners(self.camera.project(pts))

    def arms_at_rest(self):
        return all(np.abs(self.joint[n] - np.array(REST[n])).max() < 3 for n in chess_arm.ARM_NAMES)

    # -- picture
    def render(self, empty=False, seed=None):
        cam = self.camera
        w, h = cam.size
        img = np.full((h, w, 3), (170, 150, 140), np.uint8)
        cell, rim = 40, 16
        side = 8 * cell + 2 * rim
        flat = np.full((side, side, 3), (40, 60, 85), np.uint8)
        for f in range(8):
            for r in range(8):
                col = (205, 215, 225) if (f + r) % 2 else (60, 90, 120)
                x0, y0 = rim + f * cell, rim + (7 - r) * cell
                cv2.rectangle(flat, (x0, y0), (x0 + cell - 1, y0 + cell - 1), col, -1)
        k = rim / float(cell)
        src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        outline = [(-k, 8 + k), (8 + k, 8 + k), (8 + k, -k), (-k, -k)]      # texture corners in (file, rank)
        dst = cam.project([np.append(self.board_to_world(f, r), 0.0) for f, r in outline])
        M = cv2.getPerspectiveTransform(src, np.float32(dst))
        cv2.warpPerspective(flat, M, (w, h), dst=img, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
        if not empty:
            items = []
            for s, (sym, dx, dy) in self.pieces.items():
                xy = self.board_to_world(chess.square_file(s) + 0.5 + dx, chess.square_rank(s) + 0.5 + dy)
                items.append((np.linalg.norm(np.append(xy, 0.0) - cam.C), xy, sym))
            ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            for _, xy, sym in sorted(items, key=lambda t: -t[0]):           # far pieces first
                ptype = chess.Piece.from_symbol(sym).piece_type
                white = sym.isupper()
                ring = np.column_stack([xy[0] + RADIUS * np.cos(ang), xy[1] + RADIUS * np.sin(ang)])
                base = cam.project(np.column_stack([ring, np.zeros(len(ring))]))
                top = cam.project(np.column_stack([ring, np.full(len(ring), HEIGHT[ptype])]))
                hull = cv2.convexHull(np.vstack([base, top]).astype(np.float32)).astype(np.int32)
                body = (225, 228, 232) if white else (34, 32, 30)
                cap = (245, 245, 245) if white else (58, 56, 54)
                edge = (120, 120, 120) if white else (10, 10, 10)
                cv2.fillConvexPoly(img, hull, body, lineType=cv2.LINE_AA)
                cv2.fillConvexPoly(img, cv2.convexHull(top.astype(np.float32)).astype(np.int32), cap,
                                   lineType=cv2.LINE_AA)
                cv2.polylines(img, [hull], True, edge, 1, lineType=cv2.LINE_AA)
        img = cam.distort(img)
        out = img.astype(np.float32) * self.gain + self.offset
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)) if seed is None else seed)
        if self.noise:
            out += rng.normal(0, self.noise, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    # -- physics
    def opening(self, arm, pulse=None):
        """How far the gripper is open: 0 closed .. 1 fully open."""
        g = GRIPPER[arm]
        pulse = self.grip[arm] if pulse is None else pulse
        return float(np.clip((pulse - g["closed"]) / float(g["open"] - g["closed"]), 0.0, 1.0))

    def contact_pulse(self, arm):
        """Gripper pulse at which the jaws touch a piece."""
        g = GRIPPER[arm]
        return g["closed"] + (2 * RADIUS) * (g["open"] - g["closed"])

    def _problem(self, text):
        self.errors.append(text)
        self.events.append("ERROR " + text)

    def _standing(self):
        for s, (sym, dx, dy) in self.pieces.items():
            xy = self.board_to_world(chess.square_file(s) + 0.5 + dx, chess.square_rank(s) + 0.5 + dy)
            yield s, sym, xy, HEIGHT[chess.Piece.from_symbol(sym).piece_type]

    def move_joints(self, arm, target):
        start = self.joint[arm].copy()
        target = np.array(target, np.float64)
        n = max(2, int(np.abs(target - start).max() / 8) + 1)
        hit = set()
        for i in range(1, n + 1):
            pose = start + (target - start) * i / n
            p = tip(arm, pose)
            low = p[2] - (self.held[arm][1] if arm in self.held else 0.0)   # lowest point being moved
            is_open = self.opening(arm) > 2 * RADIUS + 0.05 and arm not in self.held
            for s, sym, xy, height in self._standing():
                d = float(np.linalg.norm(p[:2] - xy))
                if low >= height + 0.02 or d >= JAW_REACH:
                    continue
                if is_open and d < JAW_INSIDE:
                    continue                                  # the piece is between the open jaws
                hit.add(s)
            if p[2] < 0.15 and 0 <= self.world_to_board(p[0], p[1]).min() \
                    and self.world_to_board(p[0], p[1]).max() <= 8:
                hit.add("board")
        self.joint[arm] = target
        for s in hit:
            self._problem("%s arm %s" % (arm, "pressed into the board" if s == "board"
                                          else "knocked the piece on " + chess.square_name(s)))

    def move_gripper(self, arm, pulse):
        before = self.opening(arm)
        self.grip[arm] = float(pulse)
        after = self.opening(arm)
        p = tip(arm, self.joint[arm])
        if arm not in self.held and after < before and after <= 2 * RADIUS:       # closing on something?
            best = None
            for s, sym, xy, height in self._standing():
                d = float(np.linalg.norm(p[:2] - xy))
                if d <= GRAB_TOL and 0.2 <= p[2] <= height and (best is None or d < best[0]):
                    best = (d, s, sym)
            if best is None:
                self.events.append("%s gripper closed on nothing" % arm)
                return
            if before <= 2 * RADIUS + 0.05:
                self._problem("%s gripper was not open when it came down on %s"
                              % (arm, chess.square_name(best[1])))
            del self.pieces[best[1]]
            self.held[arm] = [best[2], float(p[2])]
            self.events.append("%s arm picked up %s from %s" % (arm, best[2], chess.square_name(best[1])))
        elif arm in self.held and after > 2 * RADIUS + 0.02:                       # letting go
            sym, grasped_at = self.held.pop(arm)
            bottom = p[2] - grasped_at
            for name, centre in BIN.items():
                if np.linalg.norm(p[:2] - np.array(centre)) <= BIN_RADIUS:
                    self.binned.append(sym)
                    self.events.append("%s arm dropped %s in the %s box" % (arm, sym, name))
                    return
            f, r = self.world_to_board(p[0], p[1])
            if not (0 <= f < 8 and 0 <= r < 8):
                self._problem("%s arm dropped %s on the table" % (arm, sym))
                return
            s = chess.square(int(f), int(r))
            dx, dy = f - int(f) - 0.5, r - int(r) - 0.5
            name = chess.square_name(s)
            if bottom > 0.3:
                self._problem("%s arm dropped %s onto %s from %.1f squares up" % (arm, sym, name, bottom))
            elif bottom < -0.3:
                self._problem("%s arm pushed %s into the board on %s" % (arm, sym, name))
            if np.hypot(dx, dy) > PLACE_TOL:
                self._problem("%s arm put %s %.2f squares off the centre of %s" % (arm, sym, np.hypot(dx, dy), name))
            if s in self.pieces:
                self._problem("%s arm put %s on top of the piece on %s" % (arm, sym, name))
                return
            self.pieces[s] = [sym, float(dx), float(dy)]
            self.events.append("%s arm put %s on %s" % (arm, sym, name))


# ---------------------------------------------------------------- the bus
class SimBus(chess_arm.NullBus):
    """Servo bus wired to a SimWorld instead of a robot."""

    def __init__(self, world, battery=11800):
        super().__init__()
        self.world = world
        self.battery = battery
        self.blocked = {}               # servo id -> position it cannot get past
        self.dead = False               # True: servos do not answer reads
        self.moves = 0
        self.actions = []
        self.head_pose = None
        for arm in chess_arm.ARM_NAMES:
            for i, p in zip(JOINTS[arm], REST[arm]):
                self.pos[i] = p
            self.pos[GRIPPER[arm]["servo"]] = 500

    def move(self, positions, ms):
        self.moves += 1
        positions = {int(i): float(p) for i, p in positions.items()}
        for i, stop in self.blocked.items():
            if i in positions:
                positions[i] = stop
        for arm in chess_arm.ARM_NAMES:
            ids = JOINTS[arm]
            if any(i in positions for i in ids):
                target = [positions.get(i, self.pos[i]) for i in ids]
                self.world.move_joints(arm, target)
            g = GRIPPER[arm]["servo"]
            if g in positions:
                self.world.move_gripper(arm, positions[g])
        for i, p in positions.items():
            self.pos[i] = int(round(p))
            self.limp.discard(i)

    def hand_moves(self, arm, pulses):
        """A person moves the limp arm by hand."""
        self.world.joint[arm] = np.array(pulses, np.float64)
        for i, p in zip(JOINTS[arm], pulses):
            self.pos[i] = int(round(p))

    def read(self, servo_id, tries=3):
        if self.dead:
            self.read_failures += 1
            return None
        servo_id = int(servo_id)
        for arm in chess_arm.ARM_NAMES:
            if servo_id == GRIPPER[arm]["servo"] and arm in self.world.held:
                g = GRIPPER[arm]
                contact = self.world.contact_pulse(arm)
                cmd = self.pos[servo_id]
                # the jaws stop on the piece
                tighter = cmd < contact if g["open"] > g["closed"] else cmd > contact
                return int(round(contact)) if tighter else cmd
        return self.pos.get(servo_id)

    def battery_mv(self):
        return self.battery

    def head(self, tilt, pan, ms=600):
        self.head_pose = (int(tilt), int(pan))

    def run_action(self, name):
        self.actions.append(name)
        if name in ("stand", "stand_slow"):
            for arm in chess_arm.ARM_NAMES:
                self.move({i: p for i, p in zip(JOINTS[arm], REST[arm])}, 1000)
                self.move({GRIPPER[arm]["servo"]: 500}, 300)


# ---------------------------------------------------------------- a pretend person teaching the arms
PATH = {
    # turn outwards, raise the folded arm at the robot's side, swing it round over the board
    "right": [(650, 200, 425), (650, 742, 147), (275, 742, 147)],
    "left": [(350, 800, 575), (350, 258, 853), (725, 258, 853)],
}


def taught_targets(world, arm, squares):
    """{square: (grasp pulses, hover pulses)} for the squares this arm can physically get to."""
    out = {}
    guess = None
    for s in squares:
        g = solve(arm, world.square_world(s, GRASP_Z), guess)
        if g is None:
            continue
        h = solve(arm, world.square_world(s, HOVER_Z), g)
        if h is None:
            continue
        out[s] = (g, h)
        guess = g
    return out


def bin_pose(arm):
    x, y = BIN[arm]
    return solve(arm, (x, y, HOVER_Z + 0.4))
