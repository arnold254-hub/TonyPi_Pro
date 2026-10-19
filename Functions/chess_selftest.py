#!/usr/bin/python3
# coding=utf8
"""
Self-test for TonyPi chess. Needs no robot and no camera.

    python3 chess_selftest.py            # everything (about two minutes on a PC)
    python3 chess_selftest.py --quick    # skip the full games
    python3 chess_selftest.py --keep     # keep the pictures of failed readings in ./selftest_debug

A pretend robot (chess_sim.py) stands in for the real one: a camera on the
head looking down at an angle at upright pieces, two arms that are taught
by "hand", and a table on which pieces can be knocked over.

  vision   the board is calibrated and a game is read back move by move,
           with the robot on either side, the board nudged and turned,
           the light changed and the lens distortion of the real camera
  engine   mate in one, built-in search and Stockfish (if installed)
  speech   every sentence the robot can say has its audio
  arms     teaching, reach, carrying, capturing, castling, en passant,
           handing over between the arms, and every way it can go wrong
  board    the robot's own SDK talking to a pretend controller board over
           a pretend serial line (needs pyserial, as on the robot)
  games    ChessPlay.play() start to finish with the arms moving pieces
"""
import os
import sys
import glob
import shutil
import tempfile
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import chess
except ImportError:
    print("python-chess is missing. It comes with the project as the folder Functions/chess;")
    print("copy that folder to the robot (or, with internet: python3 -m pip install chess).")
    sys.exit(1)

import chess_vision as vision
import chess_engine
import chess_speech as speak
import chess_arm
import chess_sim as sim
import chess_teach

chess_arm._sleep = lambda seconds: None         # the pretend robot moves instantly

FAILURES = []
KEEP = None


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)
    return bool(cond)


def keep(name, img):
    if KEEP:
        os.makedirs(KEEP, exist_ok=True)
        cv2.imwrite(os.path.join(KEEP, name + ".png"), img)


# ---------------------------------------------------------------- vision
GAME = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6",
        "Be3", "e5", "Nb3", "Be6", "f3", "Be7", "Qd2", "O-O", "O-O-O", "Nbd7",
        "g4", "b5", "g5", "b4", "Ne2", "Ne8", "f4", "a5", "f5", "a4", "Nbd4", "exd4", "Nxd4", "b3",
        "Kb1", "bxc2+", "Nxc2", "Bb3", "axb3", "axb3", "Na3", "Ne5", "h4", "Ra4"]

LENS = os.path.join(HERE, "CameraCalibration", "calibration_param.npz")


class SimCam:
    """The pretend robot's camera, with the same interface as ChessPlay.Cam."""

    def __init__(self, world, lens=False):
        self.world = world
        self.K = world.camera.K
        self.grabs = 0
        self.undistort = None
        self.on_frame = None            # called before every single frame (the player acts in between)
        self.hand = False               # a hand is over the board
        if lens:
            import ChessPlay as CP
            cam = CP.Cam.__new__(CP.Cam)            # only its lens correction is used
            cam.cap = cam.hw = cam.K = cam._maps = cam._size = None
            cam._load_lens()
            self.undistort = cam._undistort
            self._cam = cam

    def _picture(self, empty=False):
        self.grabs += 1
        if not self.world.arms_at_rest():
            self.world.errors.append("the camera was used while an arm was over the board")
        img = self.world.render(empty=empty)
        if self.hand:
            cv2.ellipse(img, (330, 260), (70, 150), 20, 0, 360, (150, 170, 210), -1)
        if self.undistort is not None:
            img = self.undistort(img)
            self.K = self._cam.K
        return img

    def grab(self, n=1):
        return self._picture()

    def frame(self):
        if self.on_frame is not None:
            self.on_frame()
        return self._picture()

    def close(self):
        pass


def make_world(robot_white, lens=False, seed=0):
    camera = None
    if lens:
        data = np.load(LENS)
        camera = sim.SimCamera(K=data["mtx_array"], distortion=data["dist_array"], pitch=55.0)
    return sim.SimWorld(robot_white=robot_white, camera=camera, seed=seed)


def calibrate(world, cam):
    """What STEP 1 of ChessPlay does."""
    world_pieces, world.pieces = world.pieces, {}
    empty = cam.grab()
    world.pieces = world_pieces
    corners = vision.find_board_corners(empty)
    if corners is None:
        return None
    cal = vision.Calibration(corners, K=cam.K, image_size=(empty.shape[1], empty.shape[0]))
    vision.set_reference(cal, empty)
    return cal


def read_game(world, cam, cal, moves, label):
    """Play `moves` on the pretend board and read every one back. Returns (wrong, unsure, guessed wrong)."""
    board = chess.Board()
    world.set_position(board)
    img = cam.grab()
    reg = vision.register(cal, img)
    if reg.ok and reg.shift <= vision.REG_APPLY_MAX:
        vision.apply_registration(cal, reg)
    rep = vision.learn_from_start(cal, img)
    baseline = vision.observe(cal, img)
    start_ok = not vision.position_problems(cal, board, baseline)
    wrong = unsure = misguess = 0
    for i, san in enumerate(moves):
        m = board.parse_san(san)
        world.human_move(board, m)
        img = cam.grab()
        vision.track(cal, img, vision.expected_state(board)[0])
        obs = vision.observe(cal, img)
        ranked, no_move = vision.rank_moves(cal, board, obs, baseline)
        got, sure, text = vision.decide(ranked, no_move)
        if sure and got != m:
            wrong += 1
            keep("%s_%02d_%s" % (label, i, san.replace("+", "")), vision.draw_grid(obs.warped, obs.occ, obs.white))
        elif not sure:
            unsure += 1
            if got != m:
                misguess += 1
        board.push(m)
        baseline = obs
    return rep, reg, start_ok, wrong, unsure, misguess


def test_vision():
    print("vision")
    scenes = [
        ("robot black", dict(robot_white=False), {}),
        ("robot white", dict(robot_white=True), {}),
        ("board nudged and turned", dict(robot_white=False), dict(shift=(0.25, -0.2), turn=2.0)),
        ("darker light", dict(robot_white=True), dict(gain=0.7, offset=10)),
        ("real lens, nudged, brighter", dict(robot_white=False, lens=True),
         dict(shift=(-0.2, 0.15), turn=-1.5, gain=1.15, offset=-10)),
    ]
    for label, make, change in scenes:
        world = make_world(seed=5, **make)
        cam = SimCam(world, lens=make.get("lens", False))
        cal = calibrate(world, cam)
        if not check(cal is not None, "%s: board found in the picture" % label):
            continue
        # between calibration and the game the board gets moved and the light changes
        world.centre = world.centre + np.array(change.get("shift", (0, 0)))
        world.angle += change.get("turn", 0.0)
        world.gain, world.offset = change.get("gain", 1.0), change.get("offset", 0.0)
        rep, reg, start_ok, wrong, unsure, misguess = read_game(world, cam, cal, GAME, label.replace(" ", "_"))
        check(reg.ok, "%s: board located (%r)" % (label, reg))
        check(start_ok and rep["occupancy_separation"] > 0 and rep["colour_separation"] > 0,
              "%s: starting position read (separation %.1f / %.1f)"
              % (label, rep["occupancy_separation"], rep["colour_separation"]))
        check(wrong == 0, "%s: %d of %d moves read wrongly" % (label, wrong, len(GAME)))
        check(unsure <= len(GAME) // 6 and misguess == 0,
              "%s: asked the player to confirm %d moves, guessed %d of them wrong" % (label, unsure, misguess))

    world = make_world(False, seed=2)
    cam = SimCam(world)
    cal = calibrate(world, cam)
    board = chess.Board()
    world.set_position(board)
    img = cam.grab()
    vision.learn_from_start(cal, img)
    base = vision.observe(cal, img)

    # nothing moved: no move must be read
    obs = vision.observe(cal, cam.grab())
    ranked, no_move = vision.rank_moves(cal, board, obs, base)
    got, sure, _ = vision.decide(ranked, no_move)
    check(got is None and vision.unchanged(cal, board, obs, base), "an untouched board reads as 'no move'")

    # a hand over the board: never a confident move
    hand = cam.grab().copy()
    cv2.ellipse(hand, (330, 260), (70, 150), 20, 0, 360, (150, 170, 210), -1)
    obs = vision.observe(cal, hand)
    ranked, no_move = vision.rank_moves(cal, board, obs, base)
    got, sure, _ = vision.decide(ranked, no_move)
    check(not sure, "a hand over the board is not taken for a move")

    # the board pushed too far away is reported, not followed
    world.centre = world.centre + np.array([1.4, 0.0])
    reg = vision.register(cal, cam.grab())
    check((not reg.ok) or reg.shift > vision.REG_APPLY_MAX, "a board pushed 1.4 squares away is reported")
    world.centre = world.centre - np.array([1.4, 0.0])

    # save / reload the calibration and learn the pieces again (second run of ChessPlay)
    tmp = tempfile.mkdtemp(prefix="tonypi_chess_")
    try:
        cal.save(os.path.join(tmp, "c.yaml"), os.path.join(tmp, "r.png"))
        cal2 = vision.Calibration.load(os.path.join(tmp, "c.yaml"))
        ok = cal2 is not None and cal2.rotation == cal.rotation and cal2.ready and cal2.K is not None
        check(ok, "calibration saved and reloaded")
        if ok:
            world2 = make_world(False, seed=9)
            rep, reg, start_ok, wrong, unsure, misguess = read_game(world2, SimCam(world2), cal2, GAME[:20], "reload")
            check(start_ok and wrong == 0 and misguess == 0,
                  "re-learning pieces from a reloaded calibration still reads the board")
        check(vision.Calibration.load(os.path.join(tmp, "missing.yaml")) is None, "a missing calibration is no error")
        with open(os.path.join(tmp, "bad.yaml"), "w") as fp:
            fp.write("corners: [[1, 2\n")
        check(vision.Calibration.load(os.path.join(tmp, "bad.yaml")) is None, "a damaged calibration file is ignored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # promotion: the four promotion pieces look the same; a queen is the guess and the player is asked
    world = make_world(True, seed=4)
    cam = SimCam(world)
    cal = calibrate(world, cam)
    world.set_position(chess.Board())
    vision.learn_from_start(cal, cam.grab())
    b = chess.Board("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    world.set_position(b)
    base = vision.observe(cal, cam.grab())
    m = b.parse_san("b8=Q")
    world.human_move(b, m)
    ranked, no_move = vision.rank_moves(cal, b, vision.observe(cal, cam.grab()), base)
    got, sure, _ = vision.decide(ranked, no_move)
    check(got == m and not sure, "promotion read as %s and put to the player" % (b.san(got) if got else None))

    # corner ordering must not depend on how the board is turned in the picture
    sq = np.float32([[100, 100], [300, 100], [300, 300], [100, 300]])
    ok = True
    for deg in range(0, 360, 15):
        a = np.radians(deg)
        R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = (sq - 200) @ R.T + 200
        o = vision.order_corners(pts[[2, 0, 3, 1]])
        cross = np.cross(o[1] - o[0], o[2] - o[1])
        ok = ok and cross > 0 and len({tuple(np.round(p)) for p in o}) == 4
    check(ok, "board corners are ordered the same way at every angle")


# ---------------------------------------------------------------- engine
MATE_IN_ONE = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4"


def test_engine():
    print("engine")
    eng = chess_engine.make_engine(time_limit=1.5, engine="builtin")
    b = chess.Board(MATE_IN_ONE)
    m = eng.best_move(b)
    check(b.is_checkmate() is False and b.san(m) == "Qxf7#", "built-in engine finds mate in one (%s)" % b.san(m))
    eng.quit()
    path = chess_engine.find_stockfish()
    if path:
        try:
            eng = chess_engine.make_engine(time_limit=1.0, engine="stockfish")
            m = eng.best_move(chess.Board(MATE_IN_ONE))
            check(b.san(m) == "Qxf7#", "%s finds mate in one (%s)" % (eng.name, b.san(m)))
            eng.quit()
        except Exception as e:
            check(False, "Stockfish at %s failed: %s" % (path, e))
    else:
        print("  info  Stockfish not installed; the robot will use the built-in engine (sudo apt install stockfish)")


# ---------------------------------------------------------------- speech
SENTENCES = ["lets_play", "please_move_my_piece", "did_not_understand", "good_game", "i_win", "you_win", "draw",
             "your_move"]
NEW_SENTENCES = ["keep_clear", "please_fix", "board_moved", "take_piece", "swap_for_queen", "is_this_your_move",
                 "battery_low"]


def test_speech():
    print("speech")
    b = chess.Board()
    keys_seen = set(SENTENCES)
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nxe5", "Nxe5", "O-O"]:
        m = b.parse_san(san)
        keys, _ = speak.move_fragments(b, m)
        keys_seen.update(keys)
        b.push(m)
    b = chess.Board(MATE_IN_ONE)
    keys_seen.update(speak.move_fragments(b, b.parse_san("Qxf7#"))[0])
    keys_seen.update(["a", "b", "c", "d", "e", "f", "g", "h", "1", "2", "3", "4", "5", "6", "7", "8",
                      "pawn", "knight", "bishop", "rook", "queen", "king", "takes", "check", "checkmate",
                      "castles_kingside", "castles_queenside", "en_passant", "promotes_to_queen"])
    missing = sorted(k for k in keys_seen if not os.path.isfile(speak._frag_path(k)))
    check(not missing, "all %d recorded audio fragments present in %s%s"
          % (len(keys_seen), speak.FRAG_DIR, (" (missing: %s)" % ", ".join(missing)) if missing else ""))
    check("checkmate" in speak.move_fragments(b, b.parse_san("Qxf7#"))[0], "mate is announced")
    check(all(k in speak.FRAGMENT_TEXT for k in keys_seen | set(NEW_SENTENCES)),
          "every fragment has its text for text-to-speech")
    unplayable = []
    for key in NEW_SENTENCES:
        if os.path.isfile(speak._frag_path(key)):
            continue
        for k in speak.FALLBACK.get(key, [key]):
            if not os.path.isfile(speak._frag_path(k)):
                unplayable.append(key)
    check(not unplayable, "the new sentences have audio or fall back on a recorded one")
    todo = [k for k in NEW_SENTENCES if not os.path.isfile(speak._frag_path(k))]
    if todo:
        print("  info  %d sentences have no audio of their own yet (%s);\n"
              "        on the robot run:  python3 chess_speech.py --make-missing" % (len(todo), ", ".join(todo)))


# ---------------------------------------------------------------- arms
DENSE = {        # squares the pretend person teaches: every other square of what each arm can get to
    "right": "a8 c8 e8 g8 b7 d7 f7 a6 c6 e6 b5 c5 a7".split(),
    "left": "h8 f8 d8 b8 g7 e7 c7 h6 f6 d6 g5 f5 h7".split(),
}


class Rig:
    """A pretend robot with its arms taught, the way chess_teach.py does it."""

    def __init__(self, robot_white=False, squares=None, seed=1, tmp=None):
        self.world = sim.SimWorld(robot_white=robot_white, seed=seed)
        self.bus = sim.SimBus(self.world)
        self.cam = SimCam(self.world)
        self.cal = calibrate(self.world, self.cam)
        self.world.set_position(chess.Board())
        vision.learn_from_start(self.cal, self.cam.grab())
        self.cfg = chess_arm.default_config()
        self.path = os.path.join(tmp or tempfile.gettempdir(), "chess_arm.yaml")
        t = self.teacher = chess_teach.Teacher(self.cfg, self.bus, self.cam, self.cal, path=self.path)
        t.see_board()
        squares = squares or DENSE
        if robot_white:                                   # the same places on the table, other names
            squares = {a: [chess.square_name(chess.square_mirror(chess.parse_square(n)) ^ 7) for n in v]
                       for a, v in squares.items()}
        for arm in chess_arm.ARM_NAMES:
            g = sim.GRIPPER[arm]
            tight = self.world.contact_pulse(arm) + (-12 if g["open"] > g["closed"] else 12)
            t.gripper(arm, "open", g["open"])
            t.gripper(arm, "grip", int(tight))
            t.gripper(arm, "relax", 500)
            t.limp(arm)
            for wp in sim.PATH[arm]:
                self.bus.hand_moves(arm, wp)
                t.path_add(arm)
            targets = sim.taught_targets(self.world, arm, [chess.parse_square(n) for n in squares[arm]])
            for s, (grasp, hover) in targets.items():
                self.bus.hand_moves(arm, grasp)
                t.store_square(arm, s, "grasp")
                self.bus.hand_moves(arm, hover)
                t.store_square(arm, s, "hover")
            self.bus.hand_moves(arm, sim.bin_pose(arm))
            t.store_bin(arm)
            self.bus.hand_moves(arm, sim.PATH[arm][-1])
            t.hold(arm)
            t.hands.arms[arm].where = "board"
            t.rest()
        self.hands = t.hands

    def fresh_hands(self):
        """New Hands from the saved configuration, as ChessPlay builds them."""
        self.teacher.save()
        cfg = chess_arm.load_config(self.path)
        hands = chess_arm.Hands(cfg, self.bus)
        hands.stance()
        hands.set_board(self.cal.oriented_corners())
        return hands

    def position(self, fen):
        board = chess.Board(fen)
        self.world.set_position(board)
        self.world.errors.clear()
        return board

    def play(self, hands, board, san):
        """Plan and carry out a move. Returns (plan, reason); the pretend table is left as the arms left it."""
        move = board.parse_san(san)
        plan, why = hands.plan(board, move)
        if plan is not None:
            hands.execute(plan)
        after = board.copy(stack=False)
        after.push(move)
        return plan, why, after


def test_arms():
    print("arms")
    tmp = tempfile.mkdtemp(prefix="tonypi_chess_")
    try:
        _test_arms(tmp)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, "the arm tests raised %r" % e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _test_arms(tmp):
    # -- configuration
    cfg = chess_arm.default_config()
    check(chess_arm.validate_config(cfg) == [] and not chess_arm.is_taught(cfg), "default configuration is valid")
    bad = chess_arm.default_config()
    bad["arms"]["left"]["joints"] = [16, 7, 6]
    bad["arms"]["right"]["rest"] = [275, 200, 1425]
    bad["speed"] = 5000
    check(len(chess_arm.validate_config(bad)) >= 3, "a broken configuration is refused")
    try:
        chess_arm.save_config(bad, os.path.join(tmp, "bad.yaml"))
        saved = True
    except chess_arm.ArmError:
        saved = os.path.exists(os.path.join(tmp, "bad.yaml"))
    check(not saved, "a broken configuration is never written to disk")
    check(chess_arm.load_config(os.path.join(tmp, "none.yaml")) is None, "no chess_arm.yaml means 'not taught'")

    # -- interpolation
    pts = [(0, 0), (4, 0), (0, 4), (4, 4), (2, 2)]
    vals = [(100, 200, 300), (180, 220, 310), (90, 300, 350), (200, 330, 380), (150, 260, 333)]
    f = chess_arm.PoseField(pts, vals, margin=0.3)
    exact = all(np.allclose(f(*p), v, atol=1e-6) for p, v in zip(pts, vals))
    check(exact, "taught poses are reproduced exactly")
    check(f.reachable(2, 1) and f.reachable(4.2, 4.2) and not f.reachable(5, 2) and not f.reachable(-1, -1),
          "reach is the taught area plus a small margin")
    one = chess_arm.PoseField([(3, 3)], [(1, 2, 3)], margin=0.3)
    line = chess_arm.PoseField([(0, 0), (2, 0), (4, 0)], [(0, 0, 0), (20, 10, 0), (40, 20, 0)], margin=0.3)
    check(one.reachable(3.2, 3) and not one.reachable(4, 3) and line.reachable(1, 0.2)
          and not line.reachable(1, 1) and np.allclose(line(1, 0), (10, 5, 0)),
          "one taught square, or squares in a row, work too")

    # -- teaching
    rig = Rig(tmp=tmp)
    world = rig.world
    check(not world.errors, "teaching by hand disturbs nothing%s" % (": %s" % world.errors if world.errors else ""))
    n = {a: len(rig.cfg["arms"][a]["points"]) for a in chess_arm.ARM_NAMES}
    check(min(n.values()) >= 10 and chess_arm.is_taught(rig.cfg), "squares taught: right %(right)d, left %(left)d" % n)
    hands = rig.fresh_hands()
    check(hands.ready() and hands.drift < 0.05, "configuration saved, reloaded and lined up with the board")
    reach = {a: [s for s in chess.SQUARES if a in hands.who_reaches(s)] for a in chess_arm.ARM_NAMES}
    both = [s for s in chess.SQUARES if len(hands.who_reaches(s)) == 2]
    check(len(reach["right"]) >= 15 and len(reach["left"]) >= 15 and len(both) >= 4,
          "reach: right %d squares, left %d, both %d" % (len(reach["right"]), len(reach["left"]), len(both)))
    worst = 0.0
    for a in chess_arm.ARM_NAMES:
        for s in reach[a]:
            for kind, z in (("grasp", sim.GRASP_Z), ("hover", sim.HOVER_Z)):
                e = sim.tip(a, np.round(hands.arms[a].target(hands.square_uv(s), kind))) - world.square_world(s, z)
                worst = max(worst, float(np.linalg.norm(e[:2])))
    check(worst < 0.2, "squares in between are hit to within %.2f of a square" % worst)
    report = rig.teacher.report()
    check("R" in report and "L" in report and "B" in report, "teaching report shows the reach map")

    def clean(board_after, what):
        ok = world.matches(board_after) and not world.errors and hands.at_rest() and hands.holding() is None
        check(ok, what + ("" if ok else "  [%s]" % "; ".join(world.errors or ["board differs"])))

    # -- ordinary move
    b = rig.position(chess.STARTING_FEN.replace(" w ", " b "))
    plan, why, after = rig.play(hands, b, "Nf6")
    clean(after, "knight g8-f6 carried by the %s arm" % (plan.steps[0].arm if plan else "?"))

    # -- capture: the captured piece goes into the box first
    b = rig.position("rnbqkbnr/ppp1pppp/8/8/8/2P5/PP1pPPPP/RNBQKBNR b KQkq - 0 1")
    plan, why, after = rig.play(hands, b, "dxc1=Q")
    check(plan is None, "a capture on the far side is left to the player (%s)" % why)
    b = rig.position("rnbqkbnr/pp1ppppp/8/2p5/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1")
    b.set_piece_at(chess.B6, chess.Piece(chess.KNIGHT, chess.WHITE))
    world.set_position(b)
    binned = len(world.binned)
    plan, why, after = rig.play(hands, b, "axb6")
    ok = plan is not None and plan.steps[0].kind == "bin" and len(world.binned) == binned + 1
    check(ok, "capture a7xb6: the knight is dropped in the box")
    clean(after, "...and the pawn stands on b6")

    # -- en passant: the captured pawn is not on the square the pawn moves to
    b = rig.position("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    b = rig.position("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    b.set_piece_at(chess.C5, chess.Piece(chess.PAWN, chess.WHITE))
    b.set_piece_at(chess.B5, chess.Piece(chess.PAWN, chess.BLACK))
    b.ep_square = chess.C6
    b.turn = chess.BLACK
    # en passant as black happens on rank 3; set the same thing up where the arms reach instead
    ep = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    ep.set_piece_at(chess.B5, chess.Piece(chess.PAWN, chess.WHITE))
    ep.set_piece_at(chess.C7, chess.Piece(chess.PAWN, chess.BLACK))
    ep.turn = chess.BLACK
    ep.push(chess.Move.from_uci("c7c5"))
    world.set_position(ep)
    world.errors.clear()
    move = chess.Move.from_uci("b5c6")
    check(ep.is_en_passant(move), "(en passant position is set up)")
    plan, why = hands.plan(ep, move)
    if check(plan is not None and plan.steps[0].kind == "bin" and plan.steps[0].src == chess.C5,
             "en passant: the pawn beside is taken off, not the empty square"):
        hands.execute(plan)
        after = ep.copy(stack=False)
        after.push(move)
        clean(after, "...and the capturing pawn stands on c6")

    # -- castling: king and rook
    b = rig.position("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
    plan, why, after = rig.play(hands, b, "O-O")
    check(plan is not None and len(plan.steps) >= 2, "castling is two pieces (%s)" % (plan or why))
    clean(after, "castled king side")
    b = rig.position("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
    plan, why, after = rig.play(hands, b, "O-O-O")
    clean(after, "castled queen side")

    # -- from one arm's side to the other's: handed over on a free square both can reach
    b = rig.position("4k3/8/r7/8/8/8/8/4K3 b - - 0 1")
    plan, why, after = rig.play(hands, b, "Rh6")
    ok = plan is not None and len(plan.steps) == 2 and plan.steps[0].arm != plan.steps[1].arm
    check(ok, "rook a6-h6 is handed from one arm to the other (%s)" % (plan or why))
    clean(after, "...and arrives on h6")
    hands.cfg["allow_relay"] = False
    b = rig.position("4k3/8/r7/8/8/8/8/4K3 b - - 0 1")
    plan, why = hands.plan(b, b.parse_san("Rh6"))
    check(plan is None, "with hand-overs switched off that move is left to the player")
    hands.cfg["allow_relay"] = True

    # -- out of reach
    b = rig.position("4k3/8/8/8/8/q7/8/4K3 b - - 0 1")
    plan, why = hands.plan(b, b.parse_san("Qa1+"))
    check(plan is None and "reach" in why, "a piece out of reach is left to the player (%s)" % why)

    # -- promotion: the pawn is carried, the player swaps it
    b = rig.position("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    plan, why = hands.plan(chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1"), chess.Move.from_uci("a2a1q"))
    check(plan is None, "a promotion on the far side is left to the player")

    # -- the same places with the robot playing white
    rig_w = Rig(robot_white=True, tmp=tmp, seed=3)
    hw = rig_w.fresh_hands()
    bw = rig_w.position(chess.STARTING_FEN)
    plan, why, after = rig_w.play(hw, bw, "Nf3")
    ok = plan is not None and rig_w.world.matches(after) and not rig_w.world.errors
    check(ok, "robot as white: knight g1-f3 (%s)" % (plan or why))

    # -- the board was nudged: the arms follow the camera
    b = rig.position(chess.STARTING_FEN.replace(" w ", " b "))
    world.centre = world.centre + np.array([0.3, -0.2])
    world.angle += 2.0
    reg, moved = vision.track(rig.cal, rig.cam.grab(), vision.expected_state(b)[0])
    drift = hands.set_board(rig.cal.oriented_corners())
    check(moved and hands.ready() and 0.2 < drift < 0.75, "board nudged by a third of a square: noticed (%.2f)" % drift)
    world.errors.clear()
    plan, why, after = rig.play(hands, b, "e6")
    clean(after, "...and the pawn is still picked up and put down in the right place")
    world.centre = world.centre + np.array([1.0, 0.0])
    corners = vision.order_corners(world.corners_image())
    far = hands.set_board(vision.rotate_corners(corners, rig.cal.rotation))
    check(not hands.ready() and far > 0.75, "board a whole square away: the arms stay down (%s)" % hands.why_off)
    world.centre = world.centre - np.array([1.3, -0.2])
    world.angle -= 2.0
    vision.apply_registration(rig.cal, vision.register(rig.cal, rig.cam.grab(), vision.expected_state(after)[0]))
    hands.set_board(rig.cal.oriented_corners())
    check(hands.ready(), "board put back: the arms are in business again")

    # -- things that go wrong
    b = rig.position(chess.STARTING_FEN.replace(" w ", " b "))
    del world.pieces[chess.G8]                          # somebody took the knight away
    rig.cfg_feel(hands) if hasattr(rig, "cfg_feel") else None
    for a in chess_arm.ARM_NAMES:                       # this gripper can feel whether it holds something
        g = hands.arms[a].arm
        g["grip_empty"], g["grip_held"] = g["grip"], int(round(world.contact_pulse(a)))
    plan, why = hands.plan(b, b.parse_san("Nf6"))
    try:
        hands.execute(plan)
        failed = False
    except chess_arm.GraspError:
        failed = True
    check(failed and hands.at_rest() and not world.errors, "closing on thin air is noticed; the arm backs out")

    b = rig.position(chess.STARTING_FEN.replace(" w ", " b "))
    plan, why = hands.plan(b, b.parse_san("Nf6"))
    arm = hands.arms[plan.steps[0].arm]
    rig.bus.blocked = {arm.joints[1]: int(arm.rest[1])}       # a servo that will not move
    try:
        hands.execute(plan)
        failed = False
    except chess_arm.ArmError:
        failed = True
    rig.bus.blocked = {}
    arm.sync()
    arm.to_rest()
    check(failed and plan.done == 0 and world.matches(b), "a blocked servo stops the move before a piece is touched")

    rig.bus.battery = 9500
    b = rig.position(chess.STARTING_FEN.replace(" w ", " b "))
    plan, why = hands.plan(b, b.parse_san("Nf6"))
    moves_before = rig.bus.moves
    try:
        hands.execute(plan)
        failed = False
    except chess_arm.ArmError:
        failed = True
    check(failed and rig.bus.moves == moves_before and not hands.battery_ok(), "with a flat battery the arms do not move")
    rig.bus.battery = 11800

    rig.bus.dead = True                                       # the servos do not answer (service running)
    b = rig.position(chess.STARTING_FEN.replace(" w ", " b "))
    plan, why, after = rig.play(hands, b, "Nf6")
    rig.bus.dead = False
    clean(after, "servos that do not answer: the move is made without the arrival check")

    # -- limits
    arm = hands.arms["right"]
    arm.limits = [(200, 800)] * 3
    c = arm.clamp([0, 500, 1000])
    check(list(c) == [200, 500, 800], "every target is clamped to the servo limits")
    arm.limits = [(0.0, 1000.0)] * 3

    # -- the teaching shell
    rig2 = Rig(tmp=tmp, squares={"right": ["a8", "c8", "a6", "c6"], "left": ["h8", "f8", "h6", "f6"]})
    said = []
    answers = iter(["", "y", ""])
    shell = chess_teach.Shell(rig2.teacher, ask=lambda prompt: next(answers, ""))
    shell.say = said.append
    for line in ["help", "stance", "where right", "open right", "relax right", "goto right b7", "rest",
                 "goto left e2", "square right z9 grasp", "jog right 9 5", "nonsense", "path right show",
                 "limp left", "hold left", "rest", "test", "save"]:
        shell.run(line)
    text = "\n".join(str(x) for x in said)
    check("! " in text and "unknown command" in text and "saved" in text and not rig2.world.errors,
          "teaching shell: commands work and mistakes are answered, not crashed on")
    check(rig2.hands.at_rest() or all(a.where == "rest" for a in rig2.teacher.hands.arms.values()),
          "teaching shell: arms end at rest")
    _test_wizard(tmp)


class Pupil:
    """The person who follows the guided tour of chess_teach.py with the pretend robot."""

    def __init__(self, world, bus, squares):
        self.world, self.bus = world, bus
        self.squares = {a: list(v) for a, v in squares.items()}
        self.arm = None
        self.way = 0
        self.asked = []
        self.first_limp = True

    def ask(self, prompt):
        self.asked.append(prompt)
        w = self.world
        if "STEP 1" in prompt:
            self.kept, w.pieces = w.pieces, {}
            return ""
        if "Is the grid aligned" in prompt:
            w.pieces = self.kept
            return "y"
        if "Teach the" in prompt and "arm?" in prompt:
            self.arm = "right" if "RIGHT" in prompt else "left"
            self.way = 0
            return ""
        if prompt.startswith("grip = "):
            g = sim.GRIPPER[self.arm]
            tight = int(w.contact_pulse(self.arm) + (-12 if g["open"] > g["closed"] else 12))
            return "" if ("grip = %d" % tight) in prompt else str(tight)
        if "limp now?" in prompt:
            if self.first_limp:                     # this robot wants the other torque command
                self.first_limp = False
                return "n"
            return "y"
        if "And now?" in prompt:
            return "y"
        if "way-point" in prompt and "press Enter" in prompt:
            path = sim.PATH[self.arm]
            if self.way >= len(path):
                return "done"
            self.bus.hand_moves(self.arm, path[self.way])
            self.way += 1
            return ""
        if "Square to teach" in prompt:
            return self.squares[self.arm].pop(0) if self.squares[self.arm] else ""
        if "put the arm there" in prompt:
            name, kind = prompt.split()[0], prompt.split()[1].rstrip(":")
            z = sim.GRASP_Z if kind == "grasp" else sim.HOVER_Z
            pose = sim.solve(self.arm, w.square_world(chess.parse_square(name), z))
            if pose is not None:
                self.bus.hand_moves(self.arm, pose)
            return ""
        if "above the box" in prompt:
            self.bus.hand_moves(self.arm, sim.bin_pose(self.arm))
            return ""
        if "servos switch on" in prompt:
            self.bus.hand_moves(self.arm, sim.PATH[self.arm][-1])
            return ""
        return ""


def _test_wizard(tmp):
    world = sim.SimWorld(robot_white=False, seed=31)
    bus = sim.SimBus(world)
    cam = SimCam(world)
    world.set_position(chess.Board())
    import ChessPlay as CP
    saved = (CP.CONFIG_YAML, CP.REFERENCE_PNG, CP.DEBUG_DIR, CP.SHOW, CP.ask)
    try:
        CP.CONFIG_YAML = os.path.join(tmp, "wizard_config.yaml")
        CP.REFERENCE_PNG = os.path.join(tmp, "wizard_reference.png")
        CP.DEBUG_DIR = os.path.join(tmp, "wizard_debug")
        CP.SHOW = False
        pupil = Pupil(world, bus, {"right": "a8 c8 e8 a6 c6 e6 b7 d7".split(),
                                   "left": "h8 f8 d8 h6 f6 d6 g7 e7".split()})
        CP.ask = pupil.ask
        path = os.path.join(tmp, "wizard_arm.yaml")
        teacher = chess_teach.Teacher(chess_arm.default_config(), bus, cam, None, path=path)
        shell = chess_teach.Shell(teacher, ask=pupil.ask)
        out = []
        shell.say = out.append
        shell.run("wizard")
        text = "\n".join(str(x) for x in out)
        cfg = chess_arm.load_config(path)
        ok = (cfg is not None and chess_arm.is_taught(cfg) and cfg["limp_value"] == 0
              and all(len(cfg["arms"][a]["points"]) == 8 and len(cfg["arms"][a]["path"]) == 3
                      and cfg["arms"][a]["bin"] for a in chess_arm.ARM_NAMES))
        check(ok, "the guided tour teaches both arms and saves chess_arm.yaml")
        check(not world.errors and all(a.where == "rest" for a in teacher.hands.arms.values())
              and "! " not in text, "...without knocking anything over, arms back at rest%s"
              % ("" if not world.errors else "  [%s]" % "; ".join(world.errors[:3])))
        if ok:
            hands = chess_arm.Hands(cfg, bus)
            hands.stance()
            hands.set_board(teacher.cal.oriented_corners())
            b = chess.Board(chess.STARTING_FEN.replace(" w ", " b "))
            world.errors.clear()
            plan, why = hands.plan(b, b.parse_san("Nc6"))
            if plan is not None:
                hands.execute(plan)
            after = b.copy(stack=False)
            after.push_san("Nc6")
            check(plan is not None and world.matches(after) and not world.errors,
                  "...and the arms taught that way play a move (%s)" % (plan or why))
        n = teacher.taught_squares()
        teacher.head(teacher.cfg["head"]["tilt"] - 60)
        check(n == 16 and teacher.taught_squares() == 0 and teacher.cal is None and teacher.cfg["board"] is None,
              "turning the head afterwards voids the taught squares (they were seen through the camera)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, "the guided tour raised %r" % e)
    finally:
        CP.CONFIG_YAML, CP.REFERENCE_PNG, CP.DEBUG_DIR, CP.SHOW, CP.ask = saved


# ---------------------------------------------------------------- controller board
class PretendBoard:
    """What sits at the other end of /dev/ttyAMA0: answers the way the robot's controller board does."""

    def __init__(self):
        import queue
        self.crc = None                     # the SDK's checksum, set once the SDK is loaded
        self.rx = queue.Queue()             # bytes on their way to the SDK
        self.servo = {i: 500 for i in range(1, 19)}
        self.pwm = {}
        self.torque = []                    # (sub command, servo id) as received
        self.bad_packets = 0
        self.silent = {99}                  # servos that do not exist
        self.mute = False                   # says nothing at all, not even the battery voltage
        self.unplugged = False              # the port cannot be opened again
        self.alive = True

    def start(self, crc):
        import threading
        self.crc = crc
        threading.Thread(target=self._battery, daemon=True).start()

    def _battery(self):
        import time
        import struct
        while self.alive:
            if not self.mute:
                self.send(0, bytes([0x04]) + struct.pack("<H", 11750))
            time.sleep(0.2)

    def send(self, func, payload):
        body = bytes([func, len(payload)]) + bytes(payload)
        for b in bytes([0xAA, 0x55]) + body + bytes([self.crc(body)]):
            self.rx.put(b)

    def receive(self, buf):
        import struct
        buf = bytes(buf)
        if len(buf) < 5 or buf[0] != 0xAA or buf[1] != 0x55 or buf[3] != len(buf) - 5 \
                or self.crc(buf[2:-1]) != buf[-1]:
            self.bad_packets += 1
            return
        func, data = buf[2], buf[4:-1]
        if func == 5:
            sub = data[0]
            if sub == 0x01:
                for k in range(data[3]):
                    sid, pos = struct.unpack_from("<BH", data, 4 + 3 * k)
                    self.servo[sid] = pos
            elif sub in (0x0B, 0x0C):
                self.torque.append((sub, data[1]))
            elif sub in (0x05, 0x07, 0x09) and data[1] not in self.silent:
                sid = data[1]
                if sub == 0x05:
                    self.send(5, struct.pack("<BBbh", sid, sub, 0, self.servo.get(sid, 500)))
                elif sub == 0x07:
                    self.send(5, struct.pack("<BBbH", sid, sub, 0, 11700))
                else:
                    self.send(5, struct.pack("<BBbB", sid, sub, 0, 36))
        elif func == 4 and data[0] == 0x01:
            for k in range(data[3]):
                sid, pos = struct.unpack_from("<BH", data, 4 + 3 * k)
                self.pwm[sid] = pos


class PretendPort:
    """Just enough of serial.Serial for the SDK."""

    def __init__(self, board):
        self.board = board
        self.rts = self.dtr = False
        self.is_open = False
        self.closed = 0                 # how often the SDK closed the port

    def setPort(self, device):
        self.device = device

    def open(self):
        if self.board.unplugged:
            import serial
            raise serial.SerialException("could not open port")
        self.is_open = True

    def close(self):
        if self.is_open:
            self.closed += 1
        self.is_open = False

    def read(self, size=1):
        import queue
        import serial
        if not self.is_open:
            raise serial.PortNotOpenError()
        try:
            return bytes([self.board.rx.get(timeout=1.0)])
        except queue.Empty:
            return b""

    def write(self, buf):
        import serial
        if not self.is_open:
            raise serial.PortNotOpenError()
        self.board.receive(buf)


def test_bus():
    print("controller board (the robot's SDK on a pretend serial line)")
    try:
        import serial
    except ImportError:
        print("  info  pyserial is not installed here; skipped (it is on the robot)")
        return
    import time
    sdk = os.path.join(os.path.dirname(HERE), "HiwonderSDK")
    if not os.path.isdir(os.path.join(sdk, "hiwonder")):
        print("  info  HiwonderSDK not found next to Functions; skipped")
        return
    sys.path.insert(0, sdk)
    real_serial = serial.Serial
    real_sleep = chess_arm._sleep
    board = None
    try:
        import importlib
        board = PretendBoard()
        serial.Serial = lambda *a, **k: PretendPort(board)
        for name in [m for m in sys.modules if m == "hiwonder" or m.startswith("hiwonder.")]:
            del sys.modules[name]
        rrc = importlib.import_module("hiwonder.ros_robot_controller_sdk")
        board.start(rrc.checksum_crc8)
        chess_arm._sleep = time.sleep                    # real time: the SDK runs threads
        bus = chess_arm.HiwonderBus(limp_value=1)

        bus.move({16: 300, 15: 250, 14: 410}, 200)
        time.sleep(0.1)
        check((board.servo[16], board.servo[15], board.servo[14]) == (300, 250, 410) and board.bad_packets == 0,
              "one command moves several servos together")
        check(bus.read(16) == 300 and bus.read(14) == 410, "servo positions are read back")
        t = time.time()
        dead = bus.read(99)
        check(dead is None and time.time() - t < 3.0 and bus.read_failures == 1,
              "a servo that does not answer times out (%.1f s) instead of hanging" % (time.time() - t))
        check(bus.read(15) == 250, "...and the next read is not confused by it")
        bus.torque(16, False)
        bus.torque(16, True)
        time.sleep(0.1)
        check(board.torque[-2:] == [(0x0B, 16), (0x0C, 16)], "limp / hold send the two torque commands")
        bus.limp_value = 0
        bus.torque(16, False)
        time.sleep(0.1)
        check(board.torque[-1] == (0x0C, 16), "limp_value in the configuration swaps them")
        bus.limp_value = 1
        bus.head(980, 1540, 100)
        time.sleep(0.1)
        check(board.pwm == {1: 980, 2: 1540}, "the head is pointed with the two PWM servos")
        mv = bus.battery_mv()
        check(mv == 11750, "battery voltage is read (%s mV)" % mv)
        check(bus.servo_voltage_mv(16) == 11700 and bus.servo_temperature(16) == 36,
              "servo voltage and temperature are read")
        if os.path.isfile(os.path.join(chess_arm.ACTION_DIR, "stand_slow.d6a")):
            bus.run_action("stand_slow")
            time.sleep(0.1)
            stand = (board.servo[6], board.servo[7], board.servo[8], board.servo[14], board.servo[15], board.servo[16])
            check(stand == (575, 800, 725, 425, 200, 275),
                  "action group stand_slow leaves the arms in the rest pose of the default configuration")
        try:
            bus.run_action("no_such_action")
            refused = False
        except chess_arm.ArmError:
            refused = True
        check(refused, "a missing action group is an error, not a silent nothing")

        cfg = chess_arm.default_config()
        cfg["settle"] = 0.05
        cfg["speed"] = 1500
        arm = chess_arm.Arm("right", cfg, bus)
        arm.sync()
        check(arm.where == "rest" and list(arm.pose) == [275, 200, 425], "the arm finds itself at rest after standing")
        arm.move([400, 300, 350])
        check([board.servo[i] for i in arm.joints] == [400, 300, 350], "an arm move arrives and is checked")
        board.silent.add(15)
        arm.move([410, 300, 350])                        # a servo stops answering: carry on without the check
        board.silent.discard(15)
        check(board.servo[16] == 410, "a servo that stops answering does not stop the game")
        arm.limp()
        pose = arm.hold()
        time.sleep(0.1)
        check(pose == [410, 300, 350] and board.torque[-1][0] == 0x0C and not arm.is_limp,
              "limp and hold: the arm is held where it was left")

        # The SDK closes the port for a second when it hears nothing, or when a read fails
        # because the tonypi service reads the same port. Commands sent meanwhile have to wait.
        port = bus.board.port
        before = port.closed
        board.mute = True
        t = time.time()
        while port.is_open and time.time() - t < 5:
            time.sleep(0.01)
        check(port.closed == before + 1 and not port.is_open, "a silent line makes the SDK close the port")
        try:
            bus.move({16: 333}, 200)
            bus.head(1000, 1500, 100)
            arm.set_gripper(510, wait=0.05)
            survived = True
        except Exception as e:
            survived = "%s: %s" % (type(e).__name__, e)
        board.mute = False
        time.sleep(0.1)
        check(survived is True and board.servo[16] == 333 and board.servo[17] == 510 and board.pwm[1] == 1000,
              "commands sent while the port is closed wait for it and arrive%s"
              % ("" if survived is True else "  [%s]" % survived))
        check(bus.read(16) == 333, "...and the servos are read again afterwards")

        board.unplugged = True
        board.mute = True
        t = time.time()
        while port.is_open and time.time() - t < 5:
            time.sleep(0.01)
        bus.PORT_WAIT = 1.5
        t = time.time()
        try:
            bus.move({16: 300}, 200)
            said = None
        except chess_arm.ArmError as e:
            said = str(e)
        except Exception as e:
            said = "%s: %s" % (type(e).__name__, e)
        check(said is not None and "systemctl stop tonypi" in said and time.time() - t < 4,
              "a port that stays closed is reported in plain words%s" % ("" if said else "  [no error]"))
        board.unplugged = False
        board.mute = False
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, "the controller board test raised %r" % e)
    finally:
        serial.Serial = real_serial
        chess_arm._sleep = real_sleep
        if board is not None:
            board.alive = False


# ---------------------------------------------------------------- whole games
class Table:
    """
    The player at the pretend table: sets up the board, makes moves, makes
    the robot's moves when asked to, and answers the prompts of ChessPlay.
    """

    def __init__(self, CP, rig, robot_white, plies, seed=0, sloppy=False, type_some=False):
        self.CP, self.rig, self.world = CP, rig, rig.world
        self.robot_white = robot_white
        self.plies = plies
        self.rng = np.random.default_rng(seed)
        self.sloppy = sloppy                      # the player forgets to make the robot's move once
        self.type_some = type_some
        self.game = None
        self.spoken = []
        self.prompts = []
        self.robot_move = None                    # (board before, move) the robot announced last
        self.my_move = None
        self.made = 0
        self.by_arm = 0
        self.by_player = 0
        self.fixes = 0
        self.confirms = 0
        self.forgot = False
        self.misread = []                         # moves the robot took for something else without asking
        self.pull_plug = False                    # stop with Ctrl-C instead of resigning
        self.my_at = 0
        self.watch_turn, self.watch_frames = -1, 0

    # what the board should show right now
    def target(self):
        b = self.game.board.copy(stack=False) if self.game else chess.Board()
        if self.robot_move is not None and self.game is not None and self.game.board.turn == self.game.robot_color:
            if self.robot_move[1] in b.legal_moves:
                b.push(self.robot_move[1])
        return b

    def choose(self, board):
        legal = list(board.legal_moves)
        caps = [m for m in legal if board.is_capture(m)]
        castle = [m for m in legal if board.is_castling(m)]
        if castle and self.rng.random() < 0.7:
            return castle[0]
        if caps and self.rng.random() < 0.5:
            return caps[int(self.rng.integers(len(caps)))]
        quiet = [m for m in legal if not m.promotion or m.promotion == chess.QUEEN]
        return quiet[int(self.rng.integers(len(quiet)))]

    def say(self, keys, text=None):
        self.spoken.append(text or "")

    # -- playing without the Enter key (--auto): the player acts between two looks of the robot
    def typed(self, timeout):
        g = self.game
        if g is None or g.board.turn == g.robot_color or self.made < self.plies:
            return None
        return "q" if len(g.board.move_stack) > self.my_at else None    # the last move has been read

    def between_frames(self):
        g = self.game
        if g is None or g.board.turn == g.robot_color:
            return
        turn = len(g.board.move_stack)
        if turn != self.watch_turn:
            self.watch_turn, self.watch_frames = turn, 0
            if self.my_move is not None and self.my_at < turn:
                if g.board.move_stack[self.my_at] != self.my_move:
                    self.misread.append("%s read as %s" % (self.my_move.uci(), g.board.move_stack[self.my_at].uci()))
                self.my_move = None
        self.watch_frames += 1
        cam = self.rig.cam
        if self.watch_frames == 6 and self.made < self.plies:
            cam.hand = True                        # reaching for a piece
        elif self.watch_frames == 7 and cam.hand:
            self.made += 1
            self.my_at = turn
            self.my_move = self.choose(g.board)
            self.world.human_move(g.board, self.my_move)
        elif self.watch_frames == 8:
            cam.hand = False

    def announce(self, board, move):
        self.spoken.append(speak.move_fragments(board, move)[1])
        self.robot_move = (board.copy(stack=False), move)

    def ask(self, prompt):
        self.prompts.append(prompt)
        w = self.world
        if "STEP 1" in prompt:
            w.clear()
            return ""
        if "STEP 2" in prompt:
            w.set_position(chess.Board())
            return ""
        if self.my_move is not None and self.game is not None and "Did you play" not in prompt:
            # the robot has made up its mind about the player's last move
            stack = self.game.board.move_stack
            if self.my_at < len(stack):
                if stack[self.my_at] != self.my_move:
                    self.misread.append("%s read as %s" % (self.my_move.uci(), stack[self.my_at].uci()))
                self.my_move = None
        if "Your move" in prompt:
            board = self.game.board
            if self.made >= self.plies:
                if self.pull_plug:
                    raise KeyboardInterrupt
                return "q"
            self.my_at = len(board.move_stack)
            self.made += 1
            self.my_move = self.choose(board)
            w.human_move(board, self.my_move)
            if self.type_some and self.made % 4 == 0:
                return board.san(self.my_move)
            return ""
        if "Did you play" in prompt:
            self.confirms += 1
            san = prompt.split("Did you play ")[1].split("?")[0]
            return "" if san == self.game.board.san(self.my_move) else self.my_move.uci()
        if "made my move" in prompt:
            self.by_player += 1
            if self.sloppy and not self.forgot:
                self.forgot = True
                return ""                          # pressed Enter without touching the board
            w.human_move(*self.robot_move)
            return ""
        if "Put it right" in prompt:
            self.fixes += 1
            if self.fixes > 6:
                raise RuntimeError("the robot keeps asking to put the board right")
            if self.fixes > 3:
                return "ok"                        # the board IS right; tell the robot so
            w.fix(self.target())
            return ""
        if "I will let go" in prompt:
            w.held.clear()                         # the player has taken hold of the piece
            return ""
        if "stands on" in prompt:                  # promotion: swap the pawn for the new piece
            w.fix(self.target())
            return ""
        if "board shows this position" in prompt:
            w.set_position(self.game.board)
            return ""
        return ""


def run_game(label, robot_white, plies, seed, engine="builtin", sloppy=False, type_some=False, auto=False,
             jolt=None, resume_after=None):
    import ChessPlay as CP
    tmp = tempfile.mkdtemp(prefix="tonypi_chess_")
    saved = dict(CONFIG_YAML=CP.CONFIG_YAML, REFERENCE_PNG=CP.REFERENCE_PNG, DEBUG_DIR=CP.DEBUG_DIR,
                 GAMES_DIR=CP.GAMES_DIR, SHOW=CP.SHOW, Cam=CP.Cam, ask=CP.ask, typed=CP.typed, Game=CP.Game)
    saved_arm = dict(HiwonderBus=chess_arm.HiwonderBus, load_config=chess_arm.load_config,
                     service_running=chess_arm.service_running)
    saved_speak = (speak.say, speak.announce_move)
    try:
        CP.CONFIG_YAML = os.path.join(tmp, "chess_config.yaml")
        CP.REFERENCE_PNG = os.path.join(tmp, "chess_reference.png")
        CP.DEBUG_DIR = os.path.join(tmp, "chess_debug")
        CP.GAMES_DIR = os.path.join(tmp, "chess_games")
        CP.SHOW = False
        rig = Rig(robot_white=robot_white, tmp=tmp, seed=seed)
        rig.teacher.save()
        world = rig.world
        world.errors.clear()
        table = Table(CP, rig, robot_white, plies, seed, sloppy, type_some)

        class Game(saved["Game"]):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                table.game = self

            def robots_turn(self):
                n = len(world.events)
                go_on = super().robots_turn()
                if any("picked up" in e for e in world.events[n:]):
                    table.by_arm += 1
                if jolt is not None and len(self.board.move_stack) == jolt[0]:
                    world.centre = world.centre + np.array(jolt[1])      # somebody bumps the table
                return go_on

        CP.Game = Game
        CP.Cam = lambda *a, **k: rig.cam
        CP.ask = table.ask
        CP.typed = table.typed
        rig.cam.on_frame = table.between_frames if auto else None
        speak.say, speak.announce_move = table.say, table.announce
        chess_arm.HiwonderBus = lambda *a, **k: rig.bus
        chess_arm.load_config = lambda path=None: chess_arm.__dict__["_selftest_cfg"]
        chess_arm._selftest_cfg = saved_arm["load_config"](rig.path)
        chess_arm.service_running = lambda name="tonypi": False

        def args(**k):
            d = dict(white=robot_white, time=0.15, skill=3, recalibrate=False, no_robot=False, camera=None,
                     no_camera=False, engine=engine, stockfish=None, no_arms=False, check_arms=False,
                     auto=auto, resume=False, tilt=None)
            d.update(k)
            return argparse.Namespace(**d)

        if resume_after:
            table.plies = resume_after
            table.pull_plug = True
        CP.play(args(check_arms=True))
        game = table.game
        played = len(game.board.move_stack)
        ok = check(not world.errors, "%s: nothing knocked over, dropped or misplaced%s"
                   % (label, "" if not world.errors else "  [%s]" % "; ".join(world.errors[:4])))
        check(not table.misread, "%s: every move of the player was read correctly or asked about%s"
              % (label, "" if not table.misread else "  [%s]" % "; ".join(table.misread)))
        check(world.matches(table.target()) or world.matches(game.board),
              "%s: after %d moves the pieces stand where the game says" % (label, played))
        # Stockfish does not play the same game twice: how many of its moves are within reach is luck
        check(table.by_arm >= (3 if engine == "builtin" else 0),
              "%s: the arms made %d of the robot's moves, the player %d" % (label, table.by_arm, table.by_player))
        check(table.fixes <= (1 if sloppy else 0) + (1 if jolt else 0),
              "%s: the player was asked to put the board right %d time(s)" % (label, table.fixes))
        if sloppy:
            check(table.fixes >= 1, "%s: a move the player forgot to make for the robot is noticed" % label)
        check(table.confirms <= max(3, played // 5), "%s: %d moves had to be confirmed by the player"
              % (label, table.confirms))
        if auto:
            check(not any("Your move" in p for p in table.prompts) and table.made >= plies,
                  "%s: %d moves were noticed without the Enter key" % (label, table.made))
        check(rig.bus.actions and rig.bus.actions[0] == "stand_slow" and rig.bus.head_pose is not None,
              "%s: the robot took its stance and looked at the board" % label)
        pgns = [p for p in glob.glob(os.path.join(CP.GAMES_DIR, "*.pgn")) if not p.endswith(CP.UNFINISHED)]
        check(len(pgns) == 1 and played > 0, "%s: game saved as PGN" % label)
        check(all(a.where == "rest" for a in game.robot.hands.arms.values()), "%s: arms at rest at the end" % label)

        if resume_after:
            check(os.path.isfile(os.path.join(CP.GAMES_DIR, CP.UNFINISHED)), "%s: unfinished game kept" % label)
            before = game.board.copy()
            table.plies = plies
            table.made = 0
            table.robot_move = table.my_move = None
            table.pull_plug = False
            world.set_position(chess.Board())          # the pieces were put back in the box meanwhile
            world.errors.clear()
            CP.play(args(resume=True))
            g2 = table.game
            same = g2.board.move_stack[:len(before.move_stack)] == before.move_stack
            check(same and len(g2.board.move_stack) > len(before.move_stack) and not world.errors,
                  "%s: resumed after move %d and played on to move %d"
                  % (label, len(before.move_stack), len(g2.board.move_stack)))
            check(not table.misread and table.fixes == 0 and world.matches(g2.board),
                  "%s: the resumed game was read correctly%s"
                  % (label, "" if not table.misread else "  [%s]" % "; ".join(table.misread)))
        if not ok and KEEP:
            shutil.copytree(CP.DEBUG_DIR, os.path.join(KEEP, label.replace(" ", "_")), dirs_exist_ok=True)
        return table
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, "%s: ChessPlay.play raised %r" % (label, e))
    finally:
        for k, v in saved.items():
            setattr(CP, k, v)
        for k, v in saved_arm.items():
            setattr(chess_arm, k, v)
        speak.say, speak.announce_move = saved_speak
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_arms():
    """The old way of playing must still work: no robot, the player moves everything."""
    print("game without arms (ChessPlay.play --no-robot)")
    import ChessPlay as CP
    tmp = tempfile.mkdtemp(prefix="tonypi_chess_")
    saved = dict(CONFIG_YAML=CP.CONFIG_YAML, REFERENCE_PNG=CP.REFERENCE_PNG, DEBUG_DIR=CP.DEBUG_DIR,
                 GAMES_DIR=CP.GAMES_DIR, SHOW=CP.SHOW, Cam=CP.Cam, ask=CP.ask, Game=CP.Game)
    saved_speak = (speak.say, speak.announce_move)
    try:
        for k in ("CONFIG_YAML", "REFERENCE_PNG", "DEBUG_DIR", "GAMES_DIR"):
            setattr(CP, k, os.path.join(tmp, os.path.basename(saved[k])))
        CP.SHOW = False
        world = sim.SimWorld(robot_white=False, seed=11)

        class Stub:
            pass
        rig = Stub()
        rig.world = world
        table = Table(CP, rig, False, 6, seed=11)

        class Game(saved["Game"]):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                table.game = self

        CP.Game = Game
        cam = SimCam(world)
        CP.Cam = lambda *a, **k: cam
        CP.ask = table.ask
        speak.say, speak.announce_move = table.say, table.announce
        CP.play(argparse.Namespace(white=False, time=0.15, skill=3, recalibrate=True, no_robot=True, camera=None,
                                   no_camera=False, engine="builtin", stockfish=None, no_arms=False,
                                   check_arms=False, auto=False, resume=False, tilt=None))
        n = len(table.game.board.move_stack)
        check(n == 12 and world.matches(table.game.board) and not table.misread,
              "6 moves each were played and read (%d)" % n)
        check(table.by_player == 6 and table.fixes == 0, "the player made the robot's moves and none was doubted")
        check(os.path.isfile(CP.CONFIG_YAML), "calibration saved")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, "ChessPlay.play raised %r" % e)
    finally:
        for k, v in saved.items():
            setattr(CP, k, v)
        speak.say, speak.announce_move = saved_speak
        shutil.rmtree(tmp, ignore_errors=True)


def test_games():
    print("games with the arms (ChessPlay.play on the pretend robot)")
    run_game("robot black", False, plies=14, seed=21)
    run_game("robot white", True, plies=14, seed=22)
    run_game("forgetful player, typed moves", False, plies=10, seed=23, sloppy=True, type_some=True)
    run_game("table bumped", False, plies=10, seed=24, jolt=(6, (0.22, -0.15)))
    run_game("interrupted and resumed", False, plies=6, seed=25, resume_after=4)
    run_game("hands-free (--auto)", False, plies=8, seed=27, auto=True)
    if chess_engine.find_stockfish():
        run_game("against Stockfish", False, plies=10, seed=26, engine="stockfish")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the full games")
    ap.add_argument("--keep", action="store_true", help="keep pictures of failures in ./selftest_debug")
    ap.add_argument("--only", default=None, help="vision, engine, speech, arms, board, noarms or games")
    opts = ap.parse_args()
    if opts.keep:
        KEEP = os.path.join(os.getcwd(), "selftest_debug")
    tests = {"vision": test_vision, "engine": test_engine, "speech": test_speech, "arms": test_arms,
             "board": test_bus, "noarms": test_no_arms, "games": test_games}
    # nothing the tests write may land among the robot's own files
    import ChessPlay
    scratch = tempfile.mkdtemp(prefix="tonypi_selftest_")
    ChessPlay.SHOW = False
    for attr in ("CONFIG_YAML", "REFERENCE_PNG", "DEBUG_DIR", "GAMES_DIR"):
        setattr(ChessPlay, attr, os.path.join(scratch, os.path.basename(getattr(ChessPlay, attr))))
    chess_arm.ARM_YAML = os.path.join(scratch, "chess_arm.yaml")
    try:
        for name, fn in tests.items():
            if opts.only and name != opts.only:
                continue
            if opts.quick and name == "games":
                continue
            fn()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print()
    if FAILURES:
        print("%d check(s) FAILED" % len(FAILURES))
        for f in FAILURES:
            print("   " + f)
        sys.exit(1)
    print("ALL OK")
