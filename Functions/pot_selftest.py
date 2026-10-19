#!/usr/bin/python3
# coding=utf8
"""
Tries PotCollect.py on a PC against a pretend robot, table and camera.

Nothing in here runs on the robot while it collects pots, and nothing is
written into the project. The pretend world behaves like the real one in
the ways that matter to the program:

  table    a wooden table top with an edge, the floor 75 cm below it, an
           AprilTag on a white card at the centre, white pots (upright
           cylinders with a darker opening and a shadow)
  camera   in the robot's head, looking forward and down, a little to the
           left of the middle of the body, with lens distortion and noise
  robot    every action group moves it by a plausible, slightly random
           amount. move_up takes a pot only if it stands where the arms
           close; walking with a pot without the arm servos held drops it;
           walking into a pot, putting a pot on the tag or on another pot,
           and stepping over the edge of the table are recorded as errors.

The step lengths and the camera are guesses about the real robot. What this
checks is the program's logic, not the numbers in SETTINGS.

    python3 pot_selftest.py              all checks (about three minutes)
    python3 pot_selftest.py --quick      skip the random tables and the robots with other step lengths
    python3 pot_selftest.py --show DIR   also save what the camera saw, as pictures in DIR
    python3 pot_selftest.py -v           print what the robot says
"""
import os
import sys
import math
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True      # leave nothing behind in the project

import PotCollect as pc

TABLE = (120.0, 100.0)          # cm, the tag lies at (0, 0)
FLOOR_Z = -75.0
POT_R, POT_H = 3.0, 6.0
TAG_CM, CARD_CM = 10.0, 13.0

CAM_H, CAM_FWD, CAM_LEFT, PITCH = 33.0, 3.0, 3.7, 55.0
K = np.array([[425.0, 0.0, 302.6], [0.0, 425.6, 276.3], [0.0, 0.0, 1.0]])
DIST = np.array([[-0.30, 0.09, 0.0, 0.0, 0.0]])
PX_PER_CM = 8

FEET = (-7.0, 2.5, 8.0)         # the robot stands on: back, front, half width (cm from the middle of the body)
REACH = (6.0, 15.5, 4.0)        # the arms close on a pot between these distances in front, and this far off-centre
PUT_AT = 10.5                   # a pot is put down this far in front

# forward cm, left cm, turn degrees for one run of an action group
MOVES = {
    "go_forward": (3.0, 0.0, 0.0), "go_forward_one_step": (2.0, 0.0, 0.0),
    "back_fast": (-2.5, 0.0, 0.0), "back": (-2.0, 0.0, 0.0),
    "turn_left_small_step": (0.0, 0.0, 7.5), "turn_right_small_step": (0.0, 0.0, -7.5),
    "left_move": (0.0, 1.0, 0.0), "right_move": (0.0, -1.0, 0.0),
    "left_move_30": (0.0, 2.5, 0.0), "right_move_30": (0.0, -2.5, 0.0),
    "stand": (0.0, 0.0, 0.0), "stand_slow": (0.0, 0.0, 0.0),
}

WOOD = (80, 120, 165)
GREY_FLOOR = (140, 140, 140)
WHITE_FLOOR = (232, 232, 230)


# ----------------------------------------------------------------- the world
class World:
    def __init__(self, pots, robot=(-10.0, -25.0, 80.0), tag=True, table=WOOD, floor=GREY_FLOOR, tiles=False,
                 buckets=(), slips=0, tag_vanishes=False, stride=1.0, swing=1.0, seed=1):
        self.pots = [tuple(map(float, p)) for p in pots]
        self.x, self.y, self.heading = map(float, robot)
        self.tag = tag
        self.floor = floor
        self.buckets = list(buckets)        # white things standing on the floor: (x, y, radius, height)
        self.slips = slips                  # how many times the arms close without holding the pot
        self.tag_vanishes = tag_vanishes    # somebody takes the tag away while the robot picks a pot up
        self.stride, self.swing = stride, swing     # this robot's steps and turns, compared with MOVES
        self._table_colour = table
        self.rng = np.random.default_rng(seed)
        self.carried = False
        self.fallen = False
        self.errors = []
        self.grabs = self.air_grabs = self.puts = 0
        self.lock = pc.SETTINGS["carry_lock"]
        self.tiles = tiles                  # dark joints every 30 cm on the floor
        self._table_tex = self._make_table(table)
        self._rays = None 

    # -- textures
    def _make_table(self, colour):
        w, h = int(TABLE[0] * PX_PER_CM), int(TABLE[1] * PX_PER_CM)
        rng = np.random.default_rng(7)
        grain = cv2.resize(rng.normal(0, 1, (h // 40, w // 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        light = np.linspace(0.88, 1.10, w, dtype=np.float32)[None, :]       # the lamp is at one end of the table
        tex = np.clip((np.float32(colour)[None, None, :] + 6.0 * grain[:, :, None]) * light[:, :, None], 0, 255)
        tex = tex.astype(np.uint8)
        if self.tag:
            card, mark = int(CARD_CM * PX_PER_CM), int(TAG_CM * PX_PER_CM)
            x0, y0 = (w - card) // 2, (h - card) // 2
            tex[y0:y0 + card, x0:x0 + card] = 240
            aruco = cv2.aruco
            if hasattr(aruco, "generateImageMarker"):
                img = aruco.generateImageMarker(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), 1, mark)
            else:
                img = aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11), 1, mark)
            x0, y0 = (w - mark) // 2, (h - mark) // 2
            tex[y0:y0 + mark, x0:x0 + mark] = img[:, :, None]
        return tex

    # -- geometry
    def axes(self):
        a = math.radians(self.heading)
        return np.array([math.cos(a), math.sin(a), 0.0]), np.array([-math.sin(a), math.cos(a), 0.0])

    def to_world(self, forward, left):
        f, l = self.axes()
        return self.x + forward * f[0] + left * l[0], self.y + forward * f[1] + left * l[1]

    def to_robot(self, point):
        f, l = self.axes()
        d = np.array([point[0] - self.x, point[1] - self.y, 0.0])
        return float(d @ f), float(d @ l)

    def camera(self):
        """Where the camera is, and the rotation from world to camera axes (x right, y down, z ahead)."""
        f, l = self.axes()
        up = np.array([0.0, 0.0, 1.0])
        centre = np.array([self.x, self.y, 0.0]) + CAM_FWD * f + CAM_LEFT * l + CAM_H * up
        p = math.radians(PITCH)
        zc = f * math.cos(p) - up * math.sin(p)
        xc = -l
        return centre, np.vstack([xc, np.cross(zc, xc), zc])

    @staticmethod
    def project(cam, points):
        """World points -> pixels of the camera picture, lens distortion included."""
        centre, R = cam
        rvec = cv2.Rodrigues(R)[0]
        pix = cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 1, 3), rvec, -R @ centre, K, DIST)[0]
        return np.clip(np.nan_to_num(pix.reshape(-1, 2), nan=-9000.0, posinf=9000.0, neginf=-9000.0), -9000, 9000)

    @staticmethod
    def in_view(cam, point):
        """Roughly: could something at this world point show in the picture?"""
        centre, R = cam
        q = R @ (np.asarray(point, np.float64) - centre)
        return q[2] > 5.0 and math.hypot(q[0], q[1]) < 1.6 * q[2]

    @staticmethod
    def _hit(cam, rays, z):
        """Where the rays of the camera's pixels meet the plane at height z: x, y."""
        centre, R = cam
        d = rays @ R                                    # the rays in world axes
        t = (z - centre[2]) / np.minimum(d[..., 2], -1e-6)
        return centre[0] + t * d[..., 0], centre[1] + t * d[..., 1]

    def ground_cm(self, pixel):
        """Where a pixel of the camera picture is on the table top (world cm)."""
        n = cv2.undistortPoints(np.float64(pixel).reshape(1, 1, 2), K, DIST).reshape(2)
        x, y = self._hit(self.camera(), np.array([n[0], n[1], 1.0]), 0.0)
        return np.array([x, y])

    # -- the camera picture
    def _cylinder(self, img, cam, x, y, z, radius, height, colour, opening=None):
        if not self.in_view(cam, (x, y, z + height / 2.0)):
            return
        ang = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        ring = np.stack([x + radius * np.cos(ang), y + radius * np.sin(ang)], axis=1)
        base = self.project(cam, np.column_stack([ring, np.full(40, z)]))
        top = self.project(cam, np.column_stack([ring, np.full(40, z + height)]))
        hull = cv2.convexHull(np.vstack([base, top]).astype(np.float32))
        cv2.fillConvexPoly(img, hull.astype(np.int32), colour, cv2.LINE_AA)
        if opening is not None:
            inner = np.stack([x + 0.8 * radius * np.cos(ang), y + 0.8 * radius * np.sin(ang)], axis=1)
            inner = self.project(cam, np.column_stack([inner, np.full(40, z + height)]))
            cv2.fillConvexPoly(img, inner.astype(np.int32), opening, cv2.LINE_AA)

    def render(self):
        """The camera picture: every pixel looks along its ray at the table or, past its edge, at the floor."""
        if self._rays is None:
            xs, ys = np.meshgrid(np.arange(pc.SIZE[0], dtype=np.float64), np.arange(pc.SIZE[1], dtype=np.float64))
            n = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2), K, DIST)
            self._rays = np.dstack([n.reshape(pc.SIZE[1], pc.SIZE[0], 2), np.ones((pc.SIZE[1], pc.SIZE[0]))])
        cam = self.camera()
        img = np.empty((pc.SIZE[1], pc.SIZE[0], 3), np.uint8)
        img[:] = self.floor
        if self.tiles:
            fx, fy = self._hit(cam, self._rays, FLOOR_Z)
            img[(np.mod(fx, 30.0) < 2.0) | (np.mod(fy, 30.0) < 2.0)] = 60
        for x, y, radius, height in self.buckets:
            self._cylinder(img, cam, x, y, FLOOR_Z, radius, height, (236, 236, 236))
        tx, ty = self._hit(cam, self._rays, 0.0)
        on_table = (np.abs(tx) <= TABLE[0] / 2) & (np.abs(ty) <= TABLE[1] / 2)
        top = cv2.remap(self._table_tex, ((tx + TABLE[0] / 2) * PX_PER_CM).astype(np.float32),
                        ((TABLE[1] / 2 - ty) * PX_PER_CM).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        img[on_table] = top[on_table]
        eye = cam[0][:2]
        ang = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        for x, y in sorted(self.pots, key=lambda p: -math.hypot(p[0] - eye[0], p[1] - eye[1])):
            if not self.in_view(cam, (x, y, 0.0)):
                continue
            shade = np.zeros(img.shape[:2], np.uint8)
            ring = np.column_stack([x + 2.5 + 1.2 * POT_R * np.cos(ang), y - 1.5 + 1.2 * POT_R * np.sin(ang), np.zeros(40)])
            cv2.fillConvexPoly(shade, self.project(cam, ring).astype(np.int32), 255, cv2.LINE_AA)
            img[shade > 0] = (img[shade > 0] * 0.6).astype(np.uint8)
            self._cylinder(img, cam, x, y, 0.0, POT_R, POT_H, (236, 236, 236), (150, 150, 150))
        noisy = img.astype(np.float32) * self.rng.uniform(0.96, 1.04) + self.rng.normal(0, 2.5, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # -- the body
    def error(self, text):
        if text not in self.errors:
            self.errors.append(text)

    def on_table(self, point, margin=0.0):
        return abs(point[0]) <= TABLE[0] / 2 - margin and abs(point[1]) <= TABLE[1] / 2 - margin

    def do(self, name, times, lock):
        if self.fallen:
            return
        if self.carried and name != "put_down" and lock != self.lock:
            self.error("dropped the pot: the arms were not held during %s" % name)
            self.carried = False
        if name == "move_up":
            return self._pick()
        if name == "put_down":
            return self._put()
        if name not in MOVES:
            return self.error("action group %s is not known to the pretend robot" % name)
        forward, left, turn = MOVES[name]
        forward, left, turn = forward * self.stride, left * self.stride, turn * self.swing
        for _ in range(times):
            wobble = 1.0 + self.rng.normal(0, 0.12)
            self.x, self.y = self.to_world(forward * wobble + (self.rng.normal(0, 0.15) if left else 0.0),
                                           left * wobble + (self.rng.normal(0, 0.15) if forward else 0.0))
            self.heading += turn * wobble + (self.rng.normal(0, 0.6) if forward or left else 0.0)
            back, front, half = FEET
            if not all(self.on_table(self.to_world(f, l)) for f in (back, front) for l in (-half, half)):
                self.error("FELL OFF THE TABLE during %s" % name)
                self.fallen = True
                return
            for k, pot in enumerate(self.pots):
                f, l = self.to_robot(pot)
                if math.hypot(f - min(max(f, back), front), l - min(max(l, -half), half)) < POT_R:
                    self.error("walked into a pot during %s" % name)
                    self.pots[k] = self.to_world(front + POT_R + 0.5, l)

    def _pick(self):
        if self.carried:
            return self.error("picked up while holding a pot")
        near, far, side = REACH
        for pot in self.pots:
            f, l = self.to_robot(pot)
            if near <= f <= far and abs(l) <= side:
                if self.slips > 0:
                    self.slips -= 1
                else:
                    self.pots.remove(pot)
                    self.carried = True
                    self.grabs += 1
                    if self.tag_vanishes:
                        self.tag = False
                        self._table_tex = self._make_table(self._table_colour)
                return
        self.air_grabs += 1

    def _put(self):
        if not self.carried:
            return
        spot = self.to_world(PUT_AT, 0.0)
        if any(math.dist(spot, p) < 2 * POT_R for p in self.pots):
            self.error("put a pot on top of another pot")
        if self.tag and max(abs(spot[0]), abs(spot[1])) < TAG_CM / 2 + POT_R:
            self.error("put a pot on the tag")
        if not self.on_table(spot, POT_R):
            self.error("put a pot down over the edge of the table")
        self.pots.append(spot)
        self.carried = False
        self.puts += 1


class SimRobot:
    """What PotCollect.Collector needs from a robot, done by the pretend world."""

    def __init__(self, world, show=None, stop_after=None):
        self.world = world
        self.show_dir = show
        self.frames = 0
        self.beeps = 0
        self.actions = []
        self.stop_after = stop_after        # (number of actions, collector): ask the collector to stop then

    def head(self):
        pass

    def act(self, name, times=1, lock=None):
        self.actions.append(name)
        self.world.do(name, times, lock)
        if self.stop_after and len(self.actions) >= self.stop_after[0]:
            self.stop_after[1].stop = True

    def frame(self, settle=None):
        return self.world.render()

    def beep(self):
        self.beeps += 1

    def wait(self, seconds):
        pass

    def show(self, img):
        self.frames += 1
        if self.show_dir:
            cv2.imwrite(os.path.join(self.show_dir, "%04d.jpg" % self.frames), img)
        return -1


# ----------------------------------------------------------------- checks
class Checks:
    def __init__(self, verbose=False, show=None):
        self.verbose = verbose
        self.show = show
        self.passed = self.failed = 0
        self.tags = pc.TagFinder()

    def check(self, what, ok, detail=""):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print("%s  %s%s" % ("ok  " if ok else "FAIL", what, "" if ok or not detail else "   <- %s" % detail))

    def vision(self, cfg):
        return pc.Vision(cfg, lens=(K, DIST), tags=self.tags)

    def start(self, name, world, stop_after=None, **settings):
        cfg = pc.load_settings(path=os.devnull + ".none")
        cfg.update(settings)
        show = None
        if self.show:
            show = os.path.join(self.show, name.replace(" ", "_"))
            os.makedirs(show, exist_ok=True)
        robot = SimRobot(world, show)
        lines = []

        def say(text):
            lines.append(text)
            if self.verbose:
                print("        " + text)

        collector = pc.Collector(robot, self.vision(cfg), cfg, say=say)
        if stop_after:
            robot.stop_after = (stop_after, collector)
        return collector, robot, lines

    def run(self, name, world, expect, once=False, stop_after=None, **settings):
        collector, robot, lines = self.start(name, world, stop_after, **settings)
        if self.verbose:
            print("---- %s" % name)
        collector.run(once=once, wait=False)
        detail = "; ".join(world.errors) or "collected %d, %d pots on the table, robot at (%.0f, %.0f)" % (
            collector.delivered, len(world.pots), world.x, world.y)
        self.check("%s: no accident" % name, not world.errors, detail)
        self.check("%s: %d collected" % (name, expect), collector.delivered == expect, detail)
        self.check("%s: ends standing, hands empty" % name,
                   robot.actions[-1] == "stand" and not world.carried and not collector.carrying, detail)
        return collector, robot, lines

    def at_centre(self, name, world, pots):
        """The pots that were fetched stand round the tag, none on it."""
        far = [p for p in pots if math.hypot(*p) > 48]
        on_tag = [p for p in pots if max(abs(p[0]), abs(p[1])) < TAG_CM / 2 + POT_R]
        self.check("%s: the pots stand round the tag" % name, not far and not on_tag,
                   "distances from the tag: %s" % ", ".join("%.0f" % math.hypot(*p) for p in pots))


def test_seeing(t):
    cfg = pc.load_settings(path=os.devnull + ".none")
    vision = t.vision(cfg)

    world = World([(0, -14), (-32, -8)], robot=(0, -42, 90))
    seen = vision.see(world.render())
    vision.learn_table(seen)
    seen = vision.see(world.render())
    t.check("seeing: the tag is found", seen.tag is not None and seen.tag.id == 1)
    t.check("seeing: two pots, the card of the tag is not one", len(seen.pots) == 2, "%d pots" % len(seen.pots))
    home = sorted(seen.pots, key=lambda p: abs(p.x - cfg["center_x"]))
    t.check("seeing: the pot next to the tag counts as collected, the other does not",
            len(home) == 2 and home[0].home and not home[1].home)
    t.check("seeing: both pots stand on the table", not any(p.off_table for p in seen.pots))
    true_gap = float(np.hypot(*world.ground_cm(cfg["drop_point"])))
    t.check("seeing: distance to the tag is measured to 2 cm",
            seen.gap_cm is not None and abs(seen.gap_cm - true_gap) < 2.0,
            "measured %s, true %.1f" % (seen.gap_cm, true_gap))
    far = [p for p in seen.pots if not p.home]
    if far and far[0].cm is not None:
        true_far = math.hypot(-32, -8)
        t.check("seeing: where the far pot stands is measured to 5 cm", abs(math.hypot(*far[0].cm) - true_far) < 5.0,
                "measured %.1f, true %.1f" % (math.hypot(*far[0].cm), true_far))

    f, l = world.to_robot((-32, -8))
    t.check("seeing: the far pot is measured from the robot to 5 cm",
            far and far[0].robot is not None and math.dist(far[0].robot, (f - POT_R, l)) < 5.0,
            "measured %s, true (%.1f, %.1f)" % (far and far[0].robot, f - POT_R, l))
    before = seen.pose
    world.heading += 12.0
    after = vision.see(world.render()).pose
    turned = None if before is None or after is None else math.degrees(pc.wrap(after[2] - before[2]))
    t.check("seeing: a turn to the left is measured as a turn to the left, to 2 degrees",
            turned is not None and abs(turned - 12.0) < 2.0, "measured %s" % turned)
    t.check("seeing: the robot stays where it is while it turns, to 3 cm",
            turned is not None and math.dist(before[:2], after[:2]) < 3.0,
            "moved %.1f cm" % (math.dist(before[:2], after[:2]) if turned is not None else -1))

    world = World([], robot=(0, 5, 90), buckets=[(0, 95, 10, 45)])
    seen = vision.see(world.render())
    vision.learn_table(seen)
    seen = vision.see(world.render())
    t.check("seeing: a white bucket on the floor beyond the table is seen, and is not on the table",
            len(seen.pots) == 1 and seen.pots[0].off_table, "%d pots" % len(seen.pots))

    world = World([(8, 20)], robot=(0, -5, 90), floor=WHITE_FLOOR, tiles=True)
    seen = vision.see(world.render())
    vision.learn_table(seen)
    seen = vision.see(world.render())
    ok = [p for p in seen.pots if not p.off_table]
    t.check("seeing: on a white tiled floor only the pot on the table is a pot", len(ok) == 1,
            "%d of %d white things accepted" % (len(ok), len(seen.pots)))

    world = World([], robot=(0, -30, 90), table=(236, 236, 236))
    vision.learn_table(vision.see(world.render()))
    t.check("seeing: a white table is recognised as a problem", vision.table is None or vision.table_looks_like_pot())


def test_settings(t):
    """The parts that touch files: settings, the pot colour taught by a click, the robot's own lens file."""
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "pot_config.yaml")
        pc.save_settings({"tag_size_cm": 8.0, "steps": {"forward": [2.0, 0.0, 0.0]}}, path)
        cfg = pc.load_settings(path)
        t.check("settings: pot_config.yaml overrides single numbers and keeps the rest",
                cfg["tag_size_cm"] == 8.0 and cfg["steps"]["forward"][0] == 2.0 and cfg["steps"]["back"][0] == -2.5
                and cfg["drop_gap_cm"] == pc.SETTINGS["drop_gap_cm"])

        cfg = pc.load_settings(path=os.devnull + ".none")
        cfg["pot_lab_min"], cfg["pot_lab_max"] = [0, 0, 0], [10, 10, 10]        # a colour no pot has
        vision = t.vision(cfg)
        world = World([(0, -5)], robot=(0, -42, 90), tag=False)
        img = world.render()
        before = vision.see(img)
        x, y = (int(v) for v in world.project(world.camera(), [[0.0, -8.0, POT_H / 2]])[0])
        pc.learn_pot_colour(cfg, before, x, y, path)
        after = vision.see(img)
        t.check("settings: a click on a pot teaches its colour",
                len(before.pots) == 0 and len(after.pots) == 1 and "pot_lab_min" in pc.load_settings(path),
                "%d pots before, %d after" % (len(before.pots), len(after.pots)))

    lens = pc.load_lens()
    t.check("the robot's lens calibration file is found", lens is not None)
    if lens is not None:
        vision = pc.Vision(pc.load_settings(path=os.devnull + ".none"), lens=lens, tags=t.tags)
        seen = vision.see(World([(-20, 0)], robot=(0, -40, 90)).render())
        t.check("a picture can be read with the robot's lens calibration",
                seen.tag is not None and seen.gap_cm is not None and len(seen.pots) == 1)


def test_runs(t, quick):
    world = World([(30, 22)])
    _, robot, _ = t.run("one pot", world, 1)
    t.at_centre("one pot", world, world.pots)
    t.check("one pot: one pick-up, none into the air", world.grabs == 1 and world.air_grabs == 0)

    world = World([(-38, 22), (40, -20), (-30, -28)], robot=(8, -22, 100))
    t.run("three pots", world, 3)
    t.at_centre("three pots", world, world.pots)

    world = World([(0, -15), (14, 4)], robot=(-20, -25, 60))
    t.run("pots that are at the centre already", world, 0)
    t.check("pots at the centre: left alone", world.grabs == 0 and world.air_grabs == 0 and len(world.pots) == 2)

    world = World([(32, 14)], floor=WHITE_FLOOR, tiles=True, buckets=[(-20, 95, 10, 45), (115, 0, 10, 45)])
    t.run("white floor, white buckets beside the table", world, 1)
    t.at_centre("white floor", world, world.pots)

    world = World([(-48, 36), (50, -38)], robot=(0, -20, 90))
    t.run("pots near the edge of the table", world, 2)
    t.at_centre("pots near the edge", world, world.pots)

    world = World([(30, 22)], slips=1)
    _, _, lines = t.run("a pot that slips out of the arms once", world, 1)
    t.check("slipped pot: the robot noticed and tried again", any("missed" in line for line in lines))

    world = World([(30, 22)], tag=False)
    _, _, lines = t.run("no tag on the table", world, 0)
    t.check("no tag: nothing is picked up and the robot says why",
            world.grabs == 0 and any("cannot see the tag" in line for line in lines))

    world = World([(30, 22)], tag_vanishes=True)
    _, _, lines = t.run("the tag disappears while a pot is carried", world, 0)
    t.check("tag disappears: the pot is back on the table and the robot says why",
            world.grabs == 1 and len(world.pots) == 1 and any("cannot get to the tag" in line for line in lines))

    world = World([(30, 22)])
    collector, robot, lines = t.start("stopped while carrying", world)
    collector.get_ready()
    pot = collector.find_pot()
    ok = pot is not None and collector.walk_to_pot(pot) and collector.pick_up()
    collector.stop = True                           # what Ctrl-C does
    collector.run(wait=False)
    t.check("stopped while carrying: the pot is put down, nothing broken",
            ok and not world.errors and not world.carried and len(world.pots) == 1 and robot.actions[-1] == "stand",
            "; ".join(world.errors))

    world = World([(30, 22), (-36, 20)])
    t.run("--once", world, 1, once=True)

    world = World([(30, 5)], robot=(0, -10, 90), table=(236, 236, 236))
    collector, robot, lines = t.run("white table", world, 0)
    t.check("white table: the robot does not walk", not any(a in MOVES and MOVES[a][0] for a in robot.actions))

    if quick:
        return
    # The program's first guesses of step length and turn angle happen to be the pretend robot's. They
    # must not matter: try robots that step and turn quite differently.
    for name, stride, swing in (("a robot with short steps and small turns", 0.6, 0.65),
                                ("a robot with long steps and big turns", 1.4, 1.5)):
        world = World([(-38, 22), (40, -20), (-30, -28)], robot=(8, -30, 100), stride=stride, swing=swing)
        t.run(name, world, 3)
        t.at_centre(name, world, world.pots)

    rng = np.random.default_rng(2026)
    for n in range(8):
        pots = []
        while len(pots) < 3:
            p = (rng.uniform(-45, 45), rng.uniform(-35, 35))
            if math.hypot(*p) > 34 and all(math.dist(p, q) > 14 for q in pots):
                pots.append(p)
        while True:                                 # anywhere near the middle, facing anywhere, clear of tag and pots
            start = (rng.uniform(-35, 35), rng.uniform(-30, 30), rng.uniform(0, 360))
            if math.hypot(*start[:2]) > 25 and all(math.dist(start[:2], p) > 16 for p in pots):
                break
        world = World(pots, robot=start, seed=100 + n)
        name = "random table %d" % (n + 1)
        t.run(name, world, 3)
        t.at_centre(name, world, world.pots)


def main():
    parser = argparse.ArgumentParser(description="Tries PotCollect.py against a pretend robot.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--show", default=None, metavar="DIR")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    t = Checks(args.verbose, args.show)
    missing = pc.missing_actions(pc.SETTINGS)
    t.check("the action groups the program uses are in ActionGroups/", not missing, ", ".join(missing))
    test_seeing(t)
    test_settings(t)
    test_runs(t, args.quick)
    print("\n%d checks passed, %d failed" % (t.passed, t.failed))
    sys.exit(1 if t.failed else 0)


if __name__ == "__main__":
    main()
