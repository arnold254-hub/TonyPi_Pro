#!/usr/bin/python3
# coding=utf8
"""
chess_teach: teach the TonyPi where the chess squares are.

Run this once, on the robot, before the first game with moving arms (and
again whenever the robot's place, the table or the board has changed):

    sudo systemctl stop tonypi
    cd /home/pi/TonyPi/Functions && python3 chess_teach.py

Type 'wizard' for a guided tour, or use the commands one by one ('help').
Everything is stored in /home/pi/TonyPi/chess_arm.yaml.

What is taught
    head      where the head has to point so the whole board is in view.
              The arms find the squares through the camera, so everything
              else is taught for ONE direction of the head: changing it
              later means photographing the board and teaching the squares
              again.
    gripper   how far to open, and how far to close on a piece
    path      way-points that take the hand from the rest pose to a spot
              above the board without sweeping through the pieces
    squares   for a few squares per arm: the gripper around the piece
              ("grasp") and straight above it ("hover", high enough to
              carry a piece over all the others). Squares in between are
              worked out from their neighbours; 'test' shows you how well.
    bin       where captured pieces are dropped (a box next to the board)

How a pose is taught
    limp      the arm's servos are switched off; you move the arm by hand
              and the program reads where you put it
    jog       the servos stay on and you nudge them: 'jog right 2 -15'
    Both can be mixed: put the arm roughly in place by hand, 'hold', then
    jog the last millimetres.

SAFETY  The arm drops when it goes limp: hold it first. Keep fingers out
of the gripper. 'rest' brings both arms down the way they came.
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "HiwonderSDK"))
sys.path.append(ROOT)

import chess

import chess_vision as vision
import chess_arm
import ChessPlay as CP

HELP = """
Commands (arm = right | left, or r | l)
  wizard                      guided tour through everything below
  stance                      playing stance, arms down, head on the board
  head [tilt] [pan]           point the head (500-2500); saves chess_debug/camera.jpg
  calibrate                   photograph the empty board (after the head or the table changed)
  board                       find the board again (after moving it) and show the reach map
  corner <square>             tell me which square is in the corner nearest my left foot

  limp <arm>                  switch the arm's servos off (HOLD THE ARM FIRST)
  hold <arm>                  switch them on again where the arm is now
  jog <arm> <1|2|3|g> <n>     nudge a joint (1 shoulder, 2 upper arm, 3 elbow, g gripper) by n
  where <arm>                 read the servo positions

  open <arm> [pulse]          gripper: set / go to the open position
  grip <arm> [pulse]          gripper: set / go to the position that holds a piece
  relax <arm> [pulse]         gripper: set / go to the closed-but-not-squeezing position
  feel <arm>                  find out whether the gripper can feel a piece

  path <arm> add|clear|show|run   way-points from the rest pose to above the board
  square <arm> <sq> grasp|hover   store where the arm is now for this square
  forget <arm> <sq>           remove a taught square
  bin <arm>                   store where the arm is now as the drop-off for captured pieces
  restpose <arm>              store where the arm is now as the rest pose

  goto <arm> <sq> [grasp]     go above a square (or down to the piece)
  take <arm> <sq>             pick the piece up from a square
  put <arm> <sq>              put the piece in the gripper down on a square
  drop <arm>                  drop the piece in the gripper into the bin
  move <sq> <sq>              carry a piece the way it is done in a game
  rest                        both arms back to the rest pose

  test                        reach map and how well the squares in between are estimated
  show                        print the configuration
  save                        write chess_arm.yaml          quit   leave (asks to save)
"""

CORNER_TO_ROTATION = {"a1": 0, "h1": 1, "h8": 2, "a8": 3}


def arm_name(text):
    text = text.lower()
    for name in chess_arm.ARM_NAMES:
        if text in (name, name[0]):
            return name
    raise ValueError("'%s' is not an arm; use right or left" % text)


def square_of(text):
    try:
        return chess.parse_square(text.lower())
    except ValueError:
        raise ValueError("'%s' is not a square; use a1 .. h8" % text)


class Teacher:
    """Everything chess_teach can do, without the command line around it."""

    def __init__(self, cfg, bus, cam, cal, path=chess_arm.ARM_YAML):
        self.cfg = cfg
        self.bus = bus
        self.cam = cam
        self.cal = cal
        self.path = path
        self.hands = chess_arm.Hands(cfg, bus)
        self.pending = {}               # (arm, square name) -> half-taught point
        self.dirty = False
        self.limp_checked = False

    # -- board
    def see_board(self, follow=True):
        """Find the board and tell the arms where it is. Returns the drift in squares (None: nothing taught)."""
        if self.cal is None:
            raise ValueError("the board has not been photographed yet: type 'calibrate'")
        if follow and self.cam is not None:
            img = self.cam.grab()
            reg = vision.register(self.cal, img)
            if reg.ok and vision.REG_APPLY_MIN <= reg.shift <= vision.REG_APPLY_MAX:
                vision.apply_registration(self.cal, reg)
            CP.show("camera", vision.draw_corners(img, self.cal.oriented_corners()))
        if not self.cfg.get("board"):
            # first time: from now on every taught place is measured against this outline
            self.cfg["board"] = {
                "corners": [[float(x), float(y)] for x, y in self.cal.corners],
                "image_size": list(self.cal.image_size or (640, 480)),
            }
            self.dirty = True
        return self._aim()

    def _aim(self):
        """Point the arms' map at the board."""
        return self.hands.set_board(self.cal.oriented_corners())

    def set_corner(self, square_name):
        """Say which square sits in the corner of the board nearest the robot's left foot."""
        name = square_name.lower()
        if name not in CORNER_TO_ROTATION:
            raise ValueError("the corner square is one of a1, h1, h8, a8")
        k = CORNER_TO_ROTATION[name]
        if k != self.cal.rotation and self.cal.reference is not None:
            turn = (k - self.cal.rotation) % 4
            self.cal.reference = np.ascontiguousarray(np.rot90(self.cal.reference, -turn))
        self.cal.rotation = k
        self._aim()

    def near_left_corner(self):
        for name, k in CORNER_TO_ROTATION.items():
            if k == self.cal.rotation % 4:
                return name

    def taught_squares(self):
        return sum(len(self.cfg["arms"][n]["points"]) for n in chess_arm.ARM_NAMES)

    def forget_view(self):
        """The camera sees the table differently now: what was taught through it is void."""
        for name in chess_arm.ARM_NAMES:
            self.cfg["arms"][name]["points"] = []
            self.hands.arms[name].rebuild()
        self.pending = {}
        self.cfg["board"] = None
        self.hands.to_teach = None
        self.cal = None
        self.dirty = True

    def uv(self, square):
        if self.hands.to_teach is None:
            self.see_board(follow=False)
        return [float(v) for v in self.hands.square_uv(square)]

    # -- head
    def head(self, tilt=None, pan=None):
        """Point the head. A new direction voids the board calibration and the taught squares."""
        head = self.cfg["head"]
        new = dict(head)
        if tilt is not None:
            new["tilt"] = int(min(max(int(tilt), 500), 2500))
        if pan is not None:
            new["pan"] = int(min(max(int(pan), 500), 2500))
        if new != head:
            head.update(new)
            self.forget_view()
        self.hands.look()
        if self.cam is not None:
            img = self.cam.grab()
            CP.show("camera", img)
            return img
        return None

    # -- arm by hand / by nudging
    def limp(self, name):
        self.hands.arms[name].limp()

    def hold(self, name):
        return self.hands.arms[name].hold()

    def flip_limp_value(self):
        self.cfg["limp_value"] = 1 - int(self.cfg["limp_value"])
        if hasattr(self.bus, "limp_value"):
            self.bus.limp_value = self.cfg["limp_value"]
        self.dirty = True

    def where(self, name):
        """Servo positions of the arm (3 joints) or raises if a servo does not answer."""
        arm = self.hands.arms[name]
        read = [self.bus.read(i) for i in arm.joints]
        if any(r is None for r in read):
            dead = [i for i, r in zip(arm.joints, read) if r is None]
            raise chess_arm.ArmError(
                "servo %s does not answer. Is the tonypi service stopped (sudo systemctl stop tonypi)?"
                % ", ".join(str(i) for i in dead))
        return [int(r) for r in read]

    def jog(self, name, joint, delta):
        arm = self.hands.arms[name]
        if str(joint).lower() == "g":
            now = self.bus.read(arm.gripper)
            now = arm.arm["relax"] if now is None else now
            return arm.set_gripper(now + delta, wait=0.3)
        k = int(joint) - 1
        if k not in (0, 1, 2):
            raise ValueError("joint is 1, 2, 3 or g")
        if arm.pose is None:
            arm.pose = np.array(self.where(name), np.float64)
        pose = arm.pose.copy()
        pose[k] += delta
        arm.move(pose, check=False)
        arm.where = "unknown"
        return [int(round(p)) for p in arm.pose]

    # -- gripper
    def gripper(self, name, which, pulse=None):
        """Go to (and with `pulse`, set) the open / grip / relax position."""
        arm = self.hands.arms[name]
        if which not in ("open", "grip", "relax"):
            raise ValueError("open, grip or relax")
        if pulse is not None:
            pulse = int(pulse)
            if not 0 <= pulse <= 1000:
                raise ValueError("the gripper position is between 0 and 1000")
            lim = arm.arm["gripper_limits"]
            arm.arm["gripper_limits"] = [min(lim[0], pulse), max(lim[1], pulse)]
            arm.arm[which] = pulse
            self.dirty = True
        return arm.set_gripper(arm.arm[which])

    def feel(self, name, ask):
        """Measure what the gripper reads closed on nothing and closed on a piece."""
        arm = self.hands.arms[name]
        arm.open()
        ask("Nothing in the %s gripper. Press Enter; it will close > " % name)
        arm.set_gripper(arm.arm["grip"], wait=0.8)
        empty = self.bus.read(arm.gripper)
        arm.open()
        ask("Hold a PAWN between the jaws. Press Enter; it will close on it > ")
        arm.set_gripper(arm.arm["grip"], wait=0.8)
        held = self.bus.read(arm.gripper)
        ask("Take hold of the pawn. Press Enter; the gripper opens > ")
        arm.open()
        arm.relax()
        if empty is None or held is None:
            raise chess_arm.ArmError("the gripper servo does not answer")
        arm.arm["grip_empty"], arm.arm["grip_held"] = int(empty), int(held)
        self.dirty = True
        return int(empty), int(held)

    # -- storing poses
    def store_square(self, name, square, kind):
        if kind not in ("grasp", "hover"):
            raise ValueError("grasp or hover")
        pose = self.where(name)
        sq_name = chess.square_name(square)
        arm = self.hands.arms[name]
        uv = self.uv(square)
        points = arm.arm["points"]
        found = [p for p in points if np.hypot(p["uv"][0] - uv[0], p["uv"][1] - uv[1]) < 0.25]
        if found:
            found[0][kind] = pose
            found[0]["name"] = sq_name
            complete = True
        else:
            pt = self.pending.setdefault((name, sq_name), {"uv": uv, "name": sq_name})
            pt[kind] = pose
            complete = "grasp" in pt and "hover" in pt
            if complete:
                points.append(self.pending.pop((name, sq_name)))
        if complete:
            arm.rebuild()
        self.dirty = True
        return pose, complete

    def forget(self, name, square):
        arm = self.hands.arms[name]
        uv = self.uv(square)
        before = len(arm.arm["points"])
        arm.arm["points"] = [p for p in arm.arm["points"]
                             if np.hypot(p["uv"][0] - uv[0], p["uv"][1] - uv[1]) >= 0.25]
        self.pending.pop((name, chess.square_name(square)), None)
        arm.rebuild()
        self.dirty = True
        return before - len(arm.arm["points"])

    def path_add(self, name):
        arm = self.hands.arms[name]
        pose = self.where(name)
        arm.arm["path"].append(pose)
        arm.path.append(np.array(pose, np.float64))
        self.dirty = True
        return pose

    def path_clear(self, name):
        arm = self.hands.arms[name]
        arm.arm["path"] = []
        arm.path = []
        self.dirty = True

    def store_bin(self, name):
        arm = self.hands.arms[name]
        pose = self.where(name)
        arm.arm["bin"] = pose
        arm.bin = np.array(pose, np.float64)
        self.dirty = True
        return pose

    def store_rest(self, name):
        arm = self.hands.arms[name]
        pose = self.where(name)
        arm.arm["rest"] = pose
        arm.rest = np.array(pose, np.float64)
        self.dirty = True
        return pose

    # -- trying it out
    def goto(self, name, square, down=False):
        arm = self.hands.arms[name]
        uv = self.uv(square)
        if not arm.reachable(uv):
            raise chess_arm.ArmError("%s is %.1f squares outside what the %s arm was taught"
                                     % (chess.square_name(square),
                                        arm.fields["grasp"].distance_outside(*uv) if arm.fields else 99, name))
        self._other_rests(name)
        arm.to_board()
        hover = arm.target(uv, "hover")
        arm.move(hover)
        arm._last_hover = hover
        if down:
            arm.open()
            arm.move(arm.target(uv, "grasp"), self.cfg["slow_speed"])

    def _other_rests(self, name):
        for other, arm in self.hands.arms.items():
            if other != name and arm.where not in ("rest", "unknown"):
                arm.to_rest()

    def take(self, name, square, piece_type=None):
        self._other_rests(name)
        self.hands.arms[name].pick(self.uv(square), piece_type)

    def put(self, name, square, piece_type=None):
        self._other_rests(name)
        self.hands.arms[name].place(self.uv(square), piece_type)

    def drop(self, name):
        self.hands.arms[name].drop_in_bin()

    def carry(self, src, dst):
        """Carry a piece the way a game does. Returns the plan that was carried out."""
        board = chess.Board(None)
        board.set_piece_at(src, chess.Piece(chess.PAWN, chess.WHITE))
        steps, why = self.hands._carry(src, dst, chess.PAWN, {src})
        if steps is None:
            raise chess_arm.ArmError(why)
        plan = chess_arm.Plan(chess.Move(src, dst), steps)
        self.hands.execute(plan)
        return plan

    def run_path(self, name):
        arm = self.hands.arms[name]
        arm.where = "rest" if arm.where == "unknown" else arm.where
        arm.to_rest()
        arm.to_board()

    def rest(self):
        for arm in self.hands.arms.values():
            if arm.where == "unknown":
                arm.sync()
            arm.to_rest()

    # -- reports
    def report(self):
        lines = []
        for name in chess_arm.ARM_NAMES:
            arm = self.hands.arms[name]
            pts = arm.arm["points"]
            lines.append("%s arm: %d squares taught (%s), %d way-points, bin %s, gripper open %s grip %s"
                         % (name, len(pts), " ".join(p.get("name", "?") for p in pts) or "none",
                            len(arm.arm["path"]), "taught" if arm.arm["bin"] else "NOT taught",
                            arm.arm["open"], arm.arm["grip"]))
            loo = chess_arm.leave_one_out(pts)
            if loo:
                worst = max(loo, key=lambda r: max(r[1], r[2]))
                lines.append("   squares in between are estimated to within about %d pulses "
                             "(worst near %s)" % (round(max(worst[1], worst[2])), worst[0]))
                if max(worst[1], worst[2]) > 25:
                    lines.append("   that is coarse: teach a square near %s, or check it with 'goto'" % worst[0])
            if not arm.arm["path"]:
                lines.append("   WARNING no way-points: the hand goes straight from the rest pose to the board")
            if not arm.arm["bin"]:
                lines.append("   no bin: this arm cannot capture")
        for (name, sq), pt in self.pending.items():
            missing = "hover" if "grasp" in pt else "grasp"
            lines.append("%s arm: %s still needs its '%s' pose" % (name, sq, missing))
        if self.hands.to_teach is not None:
            lines.append("")
            lines.append("reach (R right arm, L left arm, B both, . neither), %s in the corner by my left foot:"
                         % self.near_left_corner())
            lines.append(self.hands.reach_map())
        return "\n".join(lines)

    def save(self):
        chess_arm.save_config(self.cfg, self.path)
        self.dirty = False


# ----------------------------------------------------------------- command line
class Shell:
    def __init__(self, teacher, ask=None):
        self.t = teacher
        self.ask = ask or CP.ask

    def say(self, text=""):
        print(text)

    def confirm_limp(self, name):
        """Switch the arm off and make sure it really went limp (the command differs between firmwares)."""
        t = self.t
        self.ask("HOLD the %s arm, it is about to go limp. Press Enter > " % name)
        t.limp(name)
        if t.limp_checked:
            return True
        if CP.yes(self.ask("Is the %s arm limp now? [Y/n] > " % name)):
            t.limp_checked = True
            return True
        t.flip_limp_value()
        t.limp(name)
        if CP.yes(self.ask("And now? [Y/n] > ")):
            t.limp_checked = True
            return True
        t.flip_limp_value()
        t.hold(name)
        self.say("This robot's servos cannot be switched off from here. Use 'jog' to move the arm.")
        return False

    def run(self, line):
        """Carry out one command. Returns False when the shell should end."""
        words = line.split()
        if not words:
            return True
        cmd, args = words[0].lower(), words[1:]
        t = self.t
        try:
            if cmd in ("q", "quit", "exit"):
                return False
            elif cmd in ("help", "h", "?"):
                self.say(HELP)
            elif cmd == "wizard":
                self.wizard()
            elif cmd == "stance":
                t.hands.stance()
            elif cmd == "head":
                if args and t.taught_squares() and not self.ask(
                        "%d squares were taught with the head where it is; they will be forgotten. "
                        "Go on? [y/N] > " % t.taught_squares()).lower() in ("y", "yes"):
                    return True
                t.head(*(int(a) for a in args[:2]))
                self.say("head tilt %d pan %d; picture in chess_debug/camera.jpg"
                         % (t.cfg["head"]["tilt"], t.cfg["head"]["pan"]))
                if t.cal is None:
                    self.say("The board has to be photographed again: 'calibrate'.")
            elif cmd == "calibrate":
                self.calibrate()
            elif cmd == "board":
                drift = t.see_board()
                if drift is not None:
                    self.say("the board is %.2f squares from where the arms were taught" % drift)
                self.say(t.report())
            elif cmd == "corner":
                t.set_corner(args[0])
                self.say("%s is in the corner by my left foot" % args[0].lower())
            elif cmd == "limp":
                self.confirm_limp(arm_name(args[0]))
            elif cmd == "hold":
                self.say("holding at %s" % t.hold(arm_name(args[0])))
            elif cmd == "jog":
                self.say("now at %s" % (t.jog(arm_name(args[0]), args[1], int(args[2])),))
            elif cmd == "where":
                name = arm_name(args[0])
                self.say("%s arm %s, gripper %s" % (name, t.where(name), t.bus.read(t.hands.arms[name].gripper)))
            elif cmd in ("open", "grip", "relax"):
                pulse = t.gripper(arm_name(args[0]), cmd, args[1] if len(args) > 1 else None)
                self.say("gripper at %d" % pulse)
            elif cmd == "feel":
                empty, held = t.feel(arm_name(args[0]), self.ask)
                if abs(empty - held) >= 8:
                    self.say("closed on nothing %d, on a pawn %d: the gripper can feel a piece" % (empty, held))
                else:
                    self.say("closed on nothing %d, on a pawn %d: too close to tell apart. "
                             "Make 'grip' a little tighter if the pawn was loose; otherwise the "
                             "camera does the checking." % (empty, held))
            elif cmd == "path":
                name, what = arm_name(args[0]), (args[1].lower() if len(args) > 1 else "show")
                if what == "add":
                    self.say("way-point %d: %s" % (len(t.hands.arms[name].path) + 1, t.path_add(name)))
                elif what == "clear":
                    t.path_clear(name)
                elif what == "run":
                    self.ask("The %s arm goes to rest and back along its path. Press Enter > " % name)
                    t.run_path(name)
                else:
                    self.say("%s" % t.hands.arms[name].arm["path"])
            elif cmd == "square":
                name, sq, kind = arm_name(args[0]), square_of(args[1]), args[2].lower()
                pose, complete = t.store_square(name, sq, kind)
                self.say("%s %s for the %s arm: %s%s" % (chess.square_name(sq), kind, name, pose,
                                                         "" if complete else "  (now the other pose)"))
            elif cmd == "forget":
                self.say("removed %d" % t.forget(arm_name(args[0]), square_of(args[1])))
            elif cmd == "bin":
                self.say("bin: %s" % t.store_bin(arm_name(args[0])))
            elif cmd == "restpose":
                self.say("rest pose: %s" % t.store_rest(arm_name(args[0])))
            elif cmd == "goto":
                t.goto(arm_name(args[0]), square_of(args[1]), len(args) > 2 and args[2].lower() == "grasp")
            elif cmd == "take":
                t.take(arm_name(args[0]), square_of(args[1]))
            elif cmd == "put":
                t.put(arm_name(args[0]), square_of(args[1]))
            elif cmd == "drop":
                t.drop(arm_name(args[0]))
            elif cmd == "move":
                self.say("done: %r" % t.carry(square_of(args[0]), square_of(args[1])))
            elif cmd == "rest":
                t.rest()
            elif cmd == "test":
                self.say(t.report())
            elif cmd == "show":
                import yaml
                self.say(yaml.safe_dump(chess_arm._plain(t.cfg), default_flow_style=None, sort_keys=False))
            elif cmd == "save":
                t.save()
                self.say("saved %s" % t.path)
            else:
                self.say("unknown command '%s' (try 'help')" % cmd)
        except IndexError:
            self.say("something is missing in that command (try 'help')")
        except (ValueError, chess_arm.ArmError) as e:
            self.say("! %s" % e)
        return True

    def calibrate(self):
        """Photograph the empty board; the taught squares stay if the head has not moved."""
        t = self.t
        if t.cam is None:
            raise ValueError("there is no camera")
        t.cal = CP.calibrate_board(t.cam)
        drift = t.see_board(follow=False)
        if drift is not None and t.taught_squares():
            self.say("the board is %.2f squares from where the arms were taught" % drift)

    # -- guided tour
    def wizard(self):
        t, ask, say = self.t, self.ask, self.say
        say("\n1  STANCE AND VIEW")
        say("Stand the robot in front of the board where it will stand in every game and mark")
        say("the place of its feet and of the board (tape on the table works well).")
        ask("Press Enter; the robot takes its stance > ")
        t.hands.stance()
        while True:
            t.head()
            say("Look at chess_debug/camera.jpg. The whole board must be in the picture, with a")
            say("little room around it.")
            text = ask("Enter = good, or type a new tilt (now %d; lower looks further down) > "
                       % t.cfg["head"]["tilt"])
            if not text:
                break
            if t.taught_squares() and ask("%d squares were taught with the head where it is; they "
                                          "will be forgotten. Go on? [y/N] > "
                                          % t.taught_squares()).lower() not in ("y", "yes"):
                break
            try:
                t.head(int(text))
            except ValueError:
                say("a number between 500 and 2500, please")
        say("")
        if t.cal is None:
            self.calibrate()
        drift = t.see_board()
        if drift is not None and drift > 0.1:
            say("The board is %.2f squares from where the arms were taught before." % drift)
        corner = ask("Which square is in the corner of the board nearest my LEFT foot? [%s] > "
                     % t.near_left_corner())
        if corner:
            self.run("corner " + corner)

        for name in chess_arm.ARM_NAMES:
            if not CP.yes(ask("\nTeach the %s arm? [Y/n] > " % name.upper())):
                continue
            self.teach_arm(name)
        say("\n" + t.report())
        if CP.yes(ask("\nSave? [Y/n] > ")):
            t.save()
            say("saved %s" % t.path)

    def teach_arm(self, name):
        t, ask, say = self.t, self.ask, self.say
        arm = t.hands.arms[name]

        say("\n2  GRIPPER (%s)" % name)
        for which, text in (("open", "open wide enough to go over a piece without touching it"),
                            ("grip", "hold a pawn firmly; stop as soon as it cannot slip"),
                            ("relax", "closed without pressing (nothing in the gripper)")):
            t.gripper(name, which)
            while True:
                ans = ask("%s = %d: %s.\n   Enter = good, +n / -n to nudge, or a number > "
                          % (which, arm.arm[which], text))
                if not ans:
                    break
                try:
                    value = arm.arm[which] + int(ans) if ans[0] in "+-" else int(ans)
                    t.gripper(name, which, value)
                except ValueError as e:
                    say("! %s" % e)
        t.gripper(name, "relax")

        say("\n3  PATH (%s)" % name)
        say("Way-points that take the hand from hanging at the side to ABOVE the near edge of")
        say("the board, without passing through the pieces: usually elbow out to the side first,")
        say("then forearm forward. The last way-point must be higher than the tallest piece.")
        again = True
        if arm.arm["path"]:
            again = ask("There are %d way-points already. Teach them again? [y/N] > "
                        % len(arm.arm["path"])).lower() in ("y", "yes")
        if again:
            t.path_clear(name)
            limp = arm.is_limp or self.confirm_limp(name)
            while True:
                ans = ask("Put the arm at way-point %d and press Enter ('done' after the last one%s) > "
                          % (len(arm.arm["path"]) + 1, "" if limp else "; jog with: 1 +20"))
                if ans.lower() in ("done", "d"):
                    break
                if ans and not limp:
                    self.run("jog %s %s" % (name, ans))
                    continue
                try:
                    say("   way-point %d: %s" % (len(arm.arm["path"]) + 1, t.path_add(name)))
                except chess_arm.ArmError as e:
                    say("! %s" % e)
                    break

        say("\n4  SQUARES (%s)" % name)
        say("Put a pawn on a square. Bring the open gripper around it: that is 'grasp'. Then lift")
        say("the arm straight up until the pawn would clear the king: that is 'hover'.")
        say("Start with the corners of the area this arm can reach, then one or two in the middle.")
        limp = arm.is_limp or self.confirm_limp(name)
        arm.open()
        while True:
            ans = ask("Square to teach (Enter = finished with the %s arm) > " % name)
            if not ans:
                break
            try:
                sq = square_of(ans)
            except ValueError as e:
                say("! %s" % e)
                continue
            for kind in ("grasp", "hover"):
                while True:
                    extra = ask("   %s %s: put the arm there and press Enter%s > "
                                % (chess.square_name(sq), kind, "" if limp else " (jog with: 1 +20)"))
                    if extra and not limp:
                        self.run("jog %s %s" % (name, extra))
                        continue
                    break
                self.run("square %s %s %s" % (name, chess.square_name(sq), kind))

        say("\n5  BIN (%s)" % name)
        if CP.yes(ask("Teach where this arm drops captured pieces? [Y/n] > ")):
            ask("Hold the hand above the box, higher than the pieces, and press Enter > ")
            self.run("bin %s" % name)

        say("\n6  BACK TO REST")
        ask("Hold the arm above the board (near the last way-point) and press Enter; "
            "the servos switch on > ")
        t.hold(name)
        arm.where = "board"
        t.rest()
        say("Try it: 'goto %s <square>', 'take', 'put', 'move e7 e5', then 'rest'." % name)


def main():
    ap = argparse.ArgumentParser(description="Teach the TonyPi's arms the chess board")
    ap.add_argument("--camera", default=None, help="camera index or stream URL (default: robot camera)")
    ap.add_argument("--recalibrate", action="store_true", help="photograph the empty board again first")
    ap.add_argument("--no-robot", action="store_true", help="no servos (only useful for trying the commands)")
    ap.add_argument("--web", type=int, nargs="?", const=8081, default=None, metavar="PORT",
                    help="show the pictures in chess_debug in a browser: http://<robot>:8081/")
    args = ap.parse_args()
    CP.setup_logging()
    if args.web:
        CP.serve_debug(args.web)
    if not args.no_robot and chess_arm.service_running():
        print("The tonypi app service is running and shares the servo port; positions cannot be")
        print("read reliably while it does. (sudo systemctl start tonypi  brings it back afterwards.)")
        if CP.yes(CP.ask("Shall I stop it now? [Y/n] > ")):
            if chess_arm.stop_service():
                print("Stopped.")
            else:
                print("I could not stop it. Do it yourself:  sudo systemctl stop tonypi")
        if chess_arm.service_running() and not CP.yes(CP.ask("Carry on anyway? [y/N] > ") or "n"):
            return
    try:
        cfg = chess_arm.load_config() or chess_arm.default_config()
    except chess_arm.ArmError as e:
        print("%s\nStarting from the defaults." % e)
        cfg = chess_arm.default_config()
    fresh = not os.path.isfile(chess_arm.ARM_YAML)
    bus = chess_arm.NullBus() if args.no_robot else chess_arm.HiwonderBus(cfg["limp_value"])
    if fresh and not args.no_robot:
        cfg["head"]["pan"] = CP.Robot._pan_centre(cfg["head"]["pan"])
    cam = None
    teacher = None
    try:
        hands = chess_arm.Hands(cfg, bus)
        if not args.no_robot:
            print("The robot takes its playing stance.")
            hands.stance()
        cam = CP.Cam(args.camera, use_robot=not args.no_robot)
        cal = None if args.recalibrate else CP.usable_calibration(cam)
        teacher = Teacher(cfg, bus, cam, cal)
        teacher.hands = hands
        shell = Shell(teacher)
        print(__doc__)
        if cal is None:
            print("The board has not been photographed yet; the wizard starts with that.")
        else:
            try:
                drift = teacher.see_board()
                if drift is not None and teacher.taught_squares():
                    print("The board is %.2f squares from where the arms were taught." % drift)
            except (RuntimeError, ValueError) as e:
                print("Camera problem: %s" % e)
        print("Type 'wizard' for the guided tour, 'help' for the commands.")
        while True:
            if not shell.run(CP.ask("teach> ")):
                break
        if teacher.dirty and CP.yes(CP.ask("Save the changes? [Y/n] > ")):
            teacher.save()
            print("saved %s" % teacher.path)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if teacher is not None and teacher.dirty:
            try:
                teacher.save()
                print("saved %s" % teacher.path)
            except chess_arm.ArmError as e:
                print("not saved: %s" % e)
    finally:
        if teacher is not None:
            for arm in teacher.hands.arms.values():
                try:
                    arm.hold()                       # never leave an arm limp and forgotten
                except Exception:
                    pass
            if CP.yes(CP.ask("Bring the arms down to rest? [Y/n] > ")):
                try:
                    teacher.rest()
                except Exception as e:
                    print("could not: %s" % e)
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    main()
