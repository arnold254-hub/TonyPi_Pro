#!/usr/bin/python3
# coding=utf8
"""
Board vision for TonyPi chess. Pure OpenCV + numpy, no robot hardware.

The approach is deliberately independent of what the pieces look like:
  * The board is located once (auto-detected or by clicking its 4 corners)
    and warped to a flat 8x8 image.
  * An EMPTY board is captured as a reference. A square is "occupied" when
    it differs enough from the reference.
  * The STARTING position is captured to learn two thresholds under the
    current lighting: how much difference means "occupied", and how bright
    a piece must be to count as white. It also works out which way round
    the board is, so the pieces' colour tells us where white sits.
  * During play, which piece stands where is tracked by the game logic
    (python-chess); the camera only needs to see occupied / empty and
    light / dark. The opponent's move is the legal move that best explains
    what the board looks like now AND what changed since the last look.

Three things make this hold up on the real robot rather than only on a
drawing of a board:
  * Pieces are tall and the camera looks at them from an angle, so a piece
    also covers part of the squares behind it. The camera pose is recovered
    from the board outline, the outline of every piece is projected, and a
    square that is hidden behind an expected piece is not held against a
    candidate move ("shadow" tables).
  * The light changes between the day the empty board was photographed
    and the game. Every frame is brought back to the reference's brightness
    and contrast before it is compared (photometric_fit).
  * The board or the robot gets nudged. register() measures how far the
    squares have moved from where the calibration expects them and
    corrects the corners, or reports that the board is out of place.

All images handed to this module must already be undistorted (ChessPlay's
Cam does that with the robot's lens calibration).
"""
import os
import cv2
import numpy as np
import chess

WARP_SIZE = 400                 # warped board is WARP_SIZE x WARP_SIZE px
SQ = WARP_SIZE // 8
INNER = 0.30                    # fraction of a square trimmed from each side when sampling

PIECE_HEIGHT = 1.6              # height of the king, in squares
PIECE_RADIUS = 0.30             # radius of a piece, in squares
OUTLINE_MARGIN = 0.12           # squares added around a piece's outline: it never stands dead centre
MIN_VISIBLE = 0.25              # judge a square only if this much of its sample box can be seen
# height of each piece relative to the king
PIECE_REL_HEIGHT = {chess.PAWN: 0.55, chess.ROOK: 0.62, chess.KNIGHT: 0.70,
                    chess.BISHOP: 0.80, chess.QUEEN: 0.90, chess.KING: 1.0}
CHANGE_FACTOR = 0.6             # change threshold as a fraction of the occupancy threshold

MAX_PROBLEMS = 2                # accept a move when at most this many squares disagree
MIN_LEAD = 2                    # ...and it beats the runner-up and "no move" by at least this
FIT_LEAD = 0.15                 # a lead of one is enough if the changed pixels fit this much better (squares)

REG_PATCH = 0.40                # half-size of a registration patch, in squares
REG_SEARCH = 0.45               # largest shift register() can measure, in squares
REG_MIN_SCORE = 0.55            # weakest patch correlation that still counts
REG_MIN_POINTS = 4
REG_APPLY_MIN = 0.06            # track() leaves the corners alone for movements smaller than this (squares)
REG_APPLY_MAX = 1.00            # ...and refuses to follow the board further than this in one go
ALIGN_SLACK = 1.5               # px two looks at the board may be out of line with each other
QUIET_AREA = 0.05               # changed area (in squares) below which the picture counts as unchanged


# ---------------------------------------------------------------- geometry
def order_corners(pts):
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left (clockwise on screen)."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(ang)]                          # clockwise, because image y points down
    first = int(np.argmin(pts.sum(axis=1)))             # the corner nearest the image's top-left
    return np.roll(pts, -first, axis=0).astype(np.float32)


def rotate_corners(corners, k):
    """Rotate the corner order so the warped image turns by 90*k degrees."""
    c = [np.array(p, np.float32) for p in np.array(corners, np.float32).reshape(4, 2)]
    k %= 4
    return np.array(c[k:] + c[:k], dtype=np.float32)


def find_board_corners(img):
    """Auto-detect an EMPTY chessboard. Returns 4 outer corners (TL,TR,BR,BL) or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    pts = None
    if hasattr(cv2, "findChessboardCornersSB"):          # robust detector, OpenCV >= 4.0
        try:
            found, pts = cv2.findChessboardCornersSB(
                gray, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        except cv2.error:
            found, pts = False, None
        if not found:
            pts = None
    if pts is None:                                        # classic detector as a backup
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, pts = cv2.findChessboardCorners(cv2.GaussianBlur(gray, (3, 3), 0), (7, 7), flags=flags)
        if not found:
            return None
        pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    inner = pts.reshape(49, 2).astype(np.float32)
    # Fit the grid to all 49 inner corners; far steadier than extrapolating from four of them.
    grid = np.float32([[c + 1, r + 1] for r in range(7) for c in range(7)])
    H, _ = cv2.findHomography(grid, inner, 0)
    if H is None:
        g = inner.reshape(7, 7, 2)
        outer = [2 * g[0, 0] - g[1, 1], 2 * g[0, 6] - g[1, 5], 2 * g[6, 6] - g[5, 5], 2 * g[6, 0] - g[5, 1]]
    else:
        outer = cv2.perspectiveTransform(np.float32([[[0, 0], [8, 0], [8, 8], [0, 8]]]), H)[0]
    return order_corners(outer)


# Corners go to the centres of the outermost pixels, so that turning the warped image with
# np.rot90 is exactly the same as warping with the corners taken in a rotated order.
_EDGE = WARP_SIZE - 1
_WARP_DST = np.array([[0, 0], [_EDGE, 0], [_EDGE, _EDGE], [0, _EDGE]], np.float32)
_BOARD_DST = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)


def warp_matrix(corners):
    """3x3 matrix taking image pixels to warped-board pixels."""
    return cv2.getPerspectiveTransform(np.array(corners, np.float32).reshape(4, 2), _WARP_DST)


def warp_board(img, corners):
    return cv2.warpPerspective(img, warp_matrix(corners), (WARP_SIZE, WARP_SIZE))


def board_matrix(corners):
    """3x3 matrix taking board units (0..8 across, 0..8 down, in the order of `corners`) to image pixels."""
    return cv2.getPerspectiveTransform(_BOARD_DST, np.array(corners, np.float32).reshape(4, 2))


def transform_points(M, pts):
    pts = np.array(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.array(M, np.float64)).reshape(-1, 2)


def square_centre(square):
    """Centre of a chess square in board units of the oriented board (a8 top-left, h1 bottom-right)."""
    return chess.square_file(square) + 0.5, 7 - chess.square_rank(square) + 0.5


def square_rect(square):
    """Pixel rect (x0,y0,x1,y1) of a chess square in the warped image (a8 top-left, h1 bottom-right)."""
    f, r = chess.square_file(square), chess.square_rank(square)
    x0, y0 = f * SQ, (7 - r) * SQ
    m = int(SQ * INNER)
    return x0 + m, y0 + m, x0 + SQ - m, y0 + SQ - m


_RECTS = [square_rect(_s) for _s in chess.SQUARES]


def _box_means(values):
    """Mean of a WARP_SIZE x WARP_SIZE array inside every square's sample box."""
    out = np.zeros(64, np.float32)
    for sq, (x0, y0, x1, y1) in enumerate(_RECTS):
        out[sq] = values[y0:y1, x0:x1].mean()
    return out


def _gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


# ---------------------------------------------------------------- camera model
def estimate_intrinsics(G, image_size):
    """
    Guess the camera matrix from one board homography (board units -> image),
    assuming square pixels and the principal point in the middle of the image.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    g = T @ np.array(G, np.float64)
    g1, g2 = g[:, 0], g[:, 1]
    guesses = []
    d1 = g1[2] * g2[2]
    if abs(d1) > 1e-12:
        guesses.append((abs(d1), -(g1[0] * g2[0] + g1[1] * g2[1]) / d1))
    d2 = g2[2] ** 2 - g1[2] ** 2
    if abs(d2) > 1e-12:
        guesses.append((abs(d2), (g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2) / d2))
    f = None
    for _, f2 in sorted(guesses, reverse=True):          # best conditioned first
        if f2 > 0 and 0.3 * w < np.sqrt(f2) < 5.0 * w:
            f = float(np.sqrt(f2))
            break
    if f is None:
        f = 0.9 * w                                       # typical webcam
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)


class CameraModel:
    """Pinhole camera placed relative to the board, recovered from the board outline."""

    def __init__(self, G, K):
        self.K = np.array(K, np.float64)
        M = np.linalg.inv(self.K) @ np.array(G, np.float64)
        lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
        if M[2, 2] * lam < 0:                              # the board is in front of the camera
            lam = -lam
        r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R)                        # nearest true rotation
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        self.r1, self.r2, self.t = R[:, 0], R[:, 1], t
        n = R[:, 2]
        self.n = -n if float(n @ t) > 0 else n            # "up": from the board towards the camera
        self.height = float(-(self.n @ self.t))            # camera height above the board, squares

    def project(self, X, Y, z=0.0):
        """Image position of board point (X, Y) raised z squares above the board."""
        X, Y, z = np.broadcast_arrays(np.asarray(X, np.float64), np.asarray(Y, np.float64),
                                      np.asarray(z, np.float64))
        p = (X[..., None] * self.r1 + Y[..., None] * self.r2 + z[..., None] * self.n + self.t)
        q = p @ self.K.T
        depth = np.where(np.abs(q[..., 2]) < 1e-9, 1e-9, q[..., 2])
        return q[..., :2] / depth[..., None]


_CIRCLE = np.linspace(0, 2 * np.pi, 16, endpoint=False)


def piece_outline(model, square, height, radius=PIECE_RADIUS):
    """Image outline (convex polygon) of a piece of the given height standing on `square`."""
    X, Y = square_centre(square)
    xs, ys = X + radius * np.cos(_CIRCLE), Y + radius * np.sin(_CIRCLE)
    pts = np.vstack([model.project(xs, ys, 0.0), model.project(xs, ys, height)]).astype(np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)


class Shadows:
    """
    Which pixels of the other squares a piece hides from the camera.
    cover[(piece type, square)] lists (other square, mask) pairs; the mask
    has the shape of a sample box and is True where the piece is in the way.
    """

    def __init__(self, model=None, to_warp=None, piece_height=PIECE_HEIGHT, radius=PIECE_RADIUS):
        self.cover = {(p, s): [] for p in PIECE_REL_HEIGHT for s in chess.SQUARES}
        self.ready = model is not None
        self._poly = {}                 # (piece type, square, wide) -> outline in the warped image
        self._blob = {}                 # same key -> (x0, y0, x1, y1, mask of that window)
        self.depth = [0.0] * 64         # how far each square is from the camera, along the table
        if model is None:
            return
        foot = np.linalg.solve(np.column_stack([model.r1, model.r2, model.n]), -model.t)[:2]
        self.depth = [float(np.hypot(*(np.array(square_centre(s)) - foot))) for s in chess.SQUARES]
        for ptype, rel in PIECE_REL_HEIGHT.items():
            for s in chess.SQUARES:
                for wide in (False, True):
                    outline = piece_outline(model, s, piece_height * rel * (1.08 if wide else 1.0),
                                            radius + (OUTLINE_MARGIN if wide else 0.0))
                    poly = np.clip(transform_points(to_warp, outline), -4 * WARP_SIZE, 5 * WARP_SIZE)
                    self._poly[(ptype, s, wide)] = np.round(poly).astype(np.int32)
                poly = self._poly[(ptype, s, True)]
                mask = np.zeros((WARP_SIZE, WARP_SIZE), np.uint8)
                cv2.fillConvexPoly(mask, poly, 1)
                bx0, by0, bw, bh = cv2.boundingRect(poly)
                for t, (x0, y0, x1, y1) in enumerate(_RECTS):
                    if t == s or x1 <= bx0 or y1 <= by0 or x0 >= bx0 + bw or y0 >= by0 + bh:
                        continue
                    box = mask[y0:y1, x0:x1]
                    if box.any():
                        self.cover[(ptype, s)].append((t, box.astype(bool)))

    def blob(self, ptype, s, wide=False):
        """Outline of a piece as (x0, y0, x1, y1, mask); the window is clipped to the warped image."""
        key = (ptype, s, wide)
        if key not in self._blob:
            poly = self._poly[key]
            x, y, w, h = cv2.boundingRect(poly)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(WARP_SIZE, x + w), min(WARP_SIZE, y + h)
            if x1 <= x0 or y1 <= y0:
                self._blob[key] = (0, 0, 0, 0, np.zeros((0, 0), bool))
            else:
                mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillConvexPoly(mask, poly - np.array([x0, y0], np.int32), 1)
                self._blob[key] = (x0, y0, x1, y1, mask.astype(bool))
        return self._blob[key]

    def hidden(self, pieces):
        """pieces: (square, piece type) pairs. Returns {square: mask of its hidden pixels}."""
        out = {}
        for s, ptype in pieces:
            for t, m in self.cover[(ptype, s)]:
                out[t] = (out[t] | m) if t in out else m
        return out


def build_shadows(model, to_warp, piece_height=PIECE_HEIGHT, radius=PIECE_RADIUS):
    return Shadows(model, to_warp, piece_height, radius)


def _visible_mean(img, t, hidden, min_visible=MIN_VISIBLE, most=False):
    """
    Mean of img over the part of square t's sample box that is not hidden;
    None if too little shows. With most=True the median is taken instead:
    the value that at least half of the visible pixels reach. That is the
    test for "something is here that should not be": a neighbour that
    stands a little off its square, or leans a little further than
    expected, touches the rim of the box but never half of it.
    """
    x0, y0, x1, y1 = _RECTS[t]
    box = img[y0:y1, x0:x1]
    m = hidden.get(t)
    if m is not None:
        vis = ~m
        if vis.sum() < min_visible * m.size:
            return None
        box = box[vis]
    return float(np.median(box) if most else box.mean())


def _piece_brightness(gray, absdiff, t, hidden):
    """Grey level of the piece on square t, from the pixels that show it; None if it cannot be seen."""
    x0, y0, x1, y1 = _RECTS[t]
    d = absdiff[y0:y1, x0:x1]
    sel = d > max(15.0, float(d.mean()))
    m = hidden.get(t)
    if m is not None:
        sel = sel & ~m
    if sel.sum() < 20:
        return None
    return float(gray[y0:y1, x0:x1][sel].mean())


# ---------------------------------------------------------------- brightness / contrast
def photometric_fit(gray, ref_gray):
    """
    Find a, b with gray ~ a * ref_gray + b on the squares that look the same
    as in the reference (the empty ones). Squares holding a piece do not fit
    the line and are ignored. Returns (a, b); (1, 0) if nothing sensible fits.
    """
    x = _box_means(ref_gray).astype(np.float64)
    y = _box_means(gray).astype(np.float64)
    i, j = np.triu_indices(64, 1)
    keep = np.abs(x[i] - x[j]) > 25                       # a light and a dark square
    if not keep.any():
        return 1.0, 0.0
    i, j = i[keep], j[keep]
    a = (y[i] - y[j]) / (x[i] - x[j])
    b = y[i] - a * x[i]
    sane = (a > 0.3) & (a < 3.0)
    if not sane.any():
        return 1.0, 0.0
    a, b = a[sane], b[sane]
    resid = np.abs(y[None, :] - a[:, None] * x[None, :] - b[:, None])
    votes = (resid < 8.0).sum(axis=1)
    best = int(np.argmax(votes))
    if votes[best] < 12:
        return 1.0, 0.0
    inl = resid[best] < 8.0
    A = np.column_stack([x[inl], np.ones(int(inl.sum()))])
    sol, *_ = np.linalg.lstsq(A, y[inl], rcond=None)
    a_fit, b_fit = float(sol[0]), float(sol[1])
    if not 0.3 < a_fit < 3.0:
        return 1.0, 0.0
    return a_fit, b_fit


def normalise(gray, ref_gray):
    """Bring `gray` to the brightness and contrast of the reference."""
    a, b = photometric_fit(gray, ref_gray)
    return np.clip((gray - b) / a, 0, 255).astype(np.float32)


# ---------------------------------------------------------------- measurements
def measure(warped, reference):
    """
    For all 64 squares return (diff, brightness):
      diff       mean abs difference to the empty reference (0-255)
      brightness mean grey level of the pixels that differ (the piece), 0-255
    Both are measured after matching the frame's lighting to the reference.
    """
    r = _gray(reference)
    g = normalise(_gray(warped), r)
    return _measure_gray(g, r)


def _measure_gray(g, r):
    absdiff = np.abs(g - r)
    diff = np.zeros(64, np.float32)
    bright = np.zeros(64, np.float32)
    for sq, (x0, y0, x1, y1) in enumerate(_RECTS):
        d = absdiff[y0:y1, x0:x1]
        diff[sq] = d.mean()
        mask = d > max(15, d.mean())
        px = g[y0:y1, x0:x1][mask]
        bright[sq] = px.mean() if px.size > 20 else g[y0:y1, x0:x1].mean()
    return diff, bright


START_OCC = np.zeros(64, bool)
for _sq in chess.SQUARES:
    START_OCC[_sq] = chess.square_rank(_sq) in (0, 1, 6, 7)
START_WHITE = np.zeros(64, bool)
for _sq in chess.SQUARES:
    START_WHITE[_sq] = chess.square_rank(_sq) in (0, 1)


class Observation:
    """One look at the board."""

    def __init__(self, occ, white, diff, bright, warped, gray, absdiff):
        self.occ, self.white, self.diff, self.bright = occ, white, diff, bright
        self.warped = warped            # BGR, as the camera saw it
        self.gray = gray                # float32, lighting matched to the reference
        self.absdiff = absdiff          # float32, difference to the empty reference
        self._edges = None

    def edges(self):
        """How fast the grey level changes from one pixel to the next (grey levels per px)."""
        if self._edges is None:
            g = cv2.GaussianBlur(self.gray, (3, 3), 0)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            self._edges = cv2.dilate(np.abs(gx) + np.abs(gy), np.ones((3, 3), np.uint8))
        return self._edges

    def __iter__(self):                 # occ, white, warped = observe(...)
        return iter((self.occ, self.white, self.warped))


class Calibration:
    """Everything needed to read the board: corners, orientation, thresholds, reference."""

    def __init__(self, corners=None, rotation=0, occ_threshold=None, white_threshold=None, reference=None,
                 image_size=None, K=None, piece_height=PIECE_HEIGHT, piece_radius=PIECE_RADIUS):
        self.corners = None if corners is None else np.array(corners, np.float32).reshape(4, 2)
        self.rotation = int(rotation)
        self.occ_threshold = occ_threshold
        self.white_threshold = white_threshold
        self.reference = reference          # warped empty board (BGR), already rotated
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self.K = None if K is None else np.array(K, np.float64).reshape(3, 3)
        self.piece_height = float(piece_height)
        self.piece_radius = float(piece_radius)
        self._shadow_key = None
        self._shadows = None
        self._ref_gray = None

    @property
    def ready(self):
        return (self.corners is not None and self.reference is not None
                and self.occ_threshold is not None and self.white_threshold is not None)

    @property
    def change_threshold(self):
        return CHANGE_FACTOR * float(self.occ_threshold)

    def oriented_corners(self):
        return rotate_corners(self.corners, self.rotation)

    def set_oriented_corners(self, corners):
        self.corners = rotate_corners(corners, -self.rotation)

    def warp(self, img):
        return warp_board(img, self.oriented_corners())

    def ref_gray(self):
        if self._ref_gray is None or self._ref_gray[0] is not self.reference:
            self._ref_gray = (self.reference, _gray(self.reference))
        return self._ref_gray[1]

    def square_to_image(self, square):
        """Image position of the centre of a chess square."""
        return transform_points(board_matrix(self.oriented_corners()), [square_centre(square)])[0]

    def camera_model(self, corners=None):
        corners = self.oriented_corners() if corners is None else corners
        G = board_matrix(corners)
        K = self.K
        if K is None:
            size = self.image_size
            if size is None:
                c = np.array(corners).reshape(4, 2)
                size = (2 * float(c[:, 0].mean()), 2 * float(c[:, 1].mean()))
            K = estimate_intrinsics(G, size)
        return CameraModel(G, K)

    def shadows(self, corners=None):
        """Shadow tables for the oriented board (cached until the corners move)."""
        corners = self.oriented_corners() if corners is None else np.array(corners, np.float32)
        key = (np.round(corners, 1).tobytes(), self.piece_height, self.piece_radius,
               None if self.K is None else self.K.tobytes())
        if key != self._shadow_key:
            try:
                self._shadows = build_shadows(self.camera_model(corners), warp_matrix(corners),
                                              self.piece_height, self.piece_radius)
            except Exception:                               # degenerate geometry: assume no shadows
                self._shadows = Shadows()
            self._shadow_key = key
        return self._shadows

    # -- persistence
    def save(self, yaml_path, reference_path):
        import yaml
        data = {
            "corners": [[float(x), float(y)] for x, y in self.corners],
            "rotation": self.rotation,
            "occ_threshold": None if self.occ_threshold is None else float(self.occ_threshold),
            "white_threshold": None if self.white_threshold is None else float(self.white_threshold),
            "reference": reference_path,
            "image_size": None if self.image_size is None else [int(v) for v in self.image_size],
            "K": None if self.K is None else [[float(v) for v in row] for row in self.K],
            "piece_height": float(self.piece_height),
            "piece_radius": float(self.piece_radius),
        }
        if self.reference is not None:
            root, ext = os.path.splitext(reference_path)
            tmp = root + ".tmp" + (ext or ".png")
            if not cv2.imwrite(tmp, self.reference):
                raise IOError("could not write %s" % tmp)
            os.replace(tmp, reference_path)
        tmp = yaml_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp)
        os.replace(tmp, yaml_path)

    @classmethod
    def load(cls, yaml_path):
        import yaml
        if not os.path.isfile(yaml_path):
            return None
        try:
            with open(yaml_path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except Exception:
            return None
        ref_path = data.get("reference") or ""
        if ref_path and not os.path.isfile(ref_path):       # the folder was copied somewhere else
            ref_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), os.path.basename(ref_path))
        ref = cv2.imread(ref_path) if ref_path else None
        if data.get("corners") is None or ref is None or ref.shape[:2] != (WARP_SIZE, WARP_SIZE):
            return None
        return cls(data["corners"], data.get("rotation", 0), data.get("occ_threshold"),
                   data.get("white_threshold"), ref, data.get("image_size"), data.get("K"),
                   data.get("piece_height") or PIECE_HEIGHT, data.get("piece_radius") or PIECE_RADIUS)


def set_reference(cal, empty_img):
    """Store the warped empty board. Must be called before learning thresholds."""
    cal.reference = warp_board(empty_img, rotate_corners(cal.corners, 0))
    cal.rotation = 0
    if cal.image_size is None:
        cal.image_size = (empty_img.shape[1], empty_img.shape[0])


def _pieces(piece_map, colour=None):
    return [(s, p.piece_type) for s, p in piece_map.items() if colour is None or p.color == colour]


_START_MAP = chess.Board().piece_map()


def learn_from_start(cal, start_img):
    """
    Given the image of the starting position, work out the board orientation
    and the occupancy / colour thresholds. Returns a quality report dict.
    """
    # cal.reference is stored already rotated by cal.rotation (so a reloaded
    # calibration can be used directly); undo that to get the rotation-0 view.
    raw_ref = np.rot90(cal.reference, cal.rotation) if cal.rotation else cal.reference
    best = None
    for k in range(4):
        corners = rotate_corners(cal.corners, k)
        warped = warp_board(start_img, corners)
        ref = np.ascontiguousarray(np.rot90(raw_ref, -k)) if k else raw_ref   # rotate the reference the same way
        rg = _gray(ref)
        g = normalise(_gray(warped), rg)
        absdiff = np.abs(g - rg)
        diff, bright = _measure_gray(g, rg)
        shadows = cal.shadows(corners)
        hidden = shadows.hidden(_pieces(_START_MAP))
        # empty squares: only the part that no piece leans over
        emp_d = [_visible_mean(absdiff, t, hidden, most=True) for t in chess.SQUARES if not START_OCC[t]]
        n_hidden = sum(1 for v in emp_d if v is None)
        emp_d = np.array([v for v in emp_d if v is not None], np.float32)
        if len(emp_d) < 8:
            emp_d = diff[~START_OCC]
        for t in chess.SQUARES:
            if START_OCC[t]:
                others = shadows.hidden(_pieces(_START_MAP, not _START_MAP[t].color))
                b = _piece_brightness(g, absdiff, t, others)
                if b is not None:
                    bright[t] = b
        occ_d = diff[START_OCC]
        separation = occ_d.min() - emp_d.max()               # >0 means occupancy is cleanly separable
        white_b = bright[START_WHITE].mean()
        black_b = bright[START_OCC & ~START_WHITE].mean()
        score = separation + 0.5 * (white_b - black_b)       # white must be at the bottom
        if best is None or score > best[0]:
            best = (score, k, diff, bright, ref, separation, white_b, black_b, emp_d, n_hidden)
    score, k, diff, bright, ref, separation, white_b, black_b, emp_d, n_hidden = best
    cal.rotation = k
    cal.reference = np.ascontiguousarray(ref)
    occ_d = diff[START_OCC]
    if separation > 0:
        cal.occ_threshold = float((occ_d.min() + emp_d.max()) / 2)
    else:                                                     # overlapping: split the typical values
        cal.occ_threshold = float((np.median(occ_d) + np.median(emp_d)) / 2)
    wb = np.sort(bright[START_WHITE])
    bb = np.sort(bright[START_OCC & ~START_WHITE])
    if wb.min() > bb.max():
        cal.white_threshold = float((wb.min() + bb.max()) / 2)
    else:
        cal.white_threshold = float((np.median(wb) + np.median(bb)) / 2)
    return {
        "rotation": k,
        "occupancy_separation": float(separation),
        "occ_threshold": cal.occ_threshold,
        "white_mean": float(white_b),
        "black_mean": float(black_b),
        "white_threshold": cal.white_threshold,
        "colour_separation": float(wb.min() - bb.max()),
        "hidden_squares": int(n_hidden),
    }


def observe(cal, img):
    """Read the board. The result unpacks as (occupied[64], is_white[64], warped image)."""
    warped = cal.warp(img)
    ref = cal.ref_gray()
    gray = normalise(_gray(warped), ref)
    diff, bright = _measure_gray(gray, ref)
    occ = diff > cal.occ_threshold
    white = bright > cal.white_threshold
    return Observation(occ, white, diff, bright, warped, gray, np.abs(gray - ref))


# ---------------------------------------------------------------- where is the board now?
class Registration:
    """Result of register(): how the board sits compared with the calibration."""

    def __init__(self, ok, reason="", points=0, inliers=0, rms=0.0, shift=0.0, angle=0.0,
                 corners=None, offset=(0.0, 0.0)):
        self.ok = ok
        self.reason = reason
        self.points = points            # patches that could be matched
        self.inliers = inliers          # ...that agree on one movement
        self.rms = rms                  # their scatter, px of the warped image
        self.shift = shift              # largest movement of a board corner, in squares
        self.angle = angle              # rotation, degrees
        self.corners = corners          # oriented corners (image px) where the board is now
        self.offset = offset            # movement of the board centre (right, down) in squares

    def __repr__(self):
        if not self.ok:
            return "Registration(failed: %s)" % self.reason
        return ("Registration(shift %.2f sq, angle %.1f deg, %d/%d patches, rms %.1f px)"
                % (self.shift, self.angle, self.inliers, self.points, self.rms))


def _fit_rigid(src, dst):
    """Least-squares rotation + translation taking src to dst (n x 2 arrays)."""
    if len(src) < 3:
        return np.eye(2), (dst - src).mean(axis=0)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, cd - R @ cs


def _robust_rigid(src, dst, tol=2.5):
    """Rigid fit that ignores patches disagreeing with the majority. Returns R, t, inlier mask, rms."""
    n = len(src)
    # start from the median movement, which a minority of wild matches cannot drag away
    med = np.median(dst - src, axis=0)
    err = np.linalg.norm(dst - src - med, axis=1)
    inl = err <= max(3 * tol, 2.5 * np.median(err))
    if inl.sum() < min(n, 3):
        inl = np.ones(n, bool)
    R, t = np.eye(2), med

    def fit(sel):
        # patches huddled in one spot cannot tell a turn from a shift (a row of them can)
        if np.ptp(src[sel], axis=0).max() < 2.5 * SQ:
            return np.eye(2), (dst[sel] - src[sel]).mean(axis=0)
        return _fit_rigid(src[sel], dst[sel])

    for _ in range(6):
        R, t = fit(inl)
        err = np.linalg.norm(src @ R.T + t - dst, axis=1)
        new = err <= tol
        if new.sum() < min(n, 3) or (new == inl).all():
            break
        inl = new
    err = np.linalg.norm(src[inl] @ R.T + t - dst[inl], axis=1)
    return R, t, inl, float(np.sqrt((err ** 2).mean())) if len(err) else 0.0


def _measure_shifts(cur, ref, expected_occ):
    """Match the pattern around the 49 inner grid corners. Returns (reference points, shifts)."""
    hp = int(round(REG_PATCH * SQ))
    rad = int(round(REG_SEARCH * SQ))
    src, shift = [], []
    for j in range(1, 8):
        for i in range(1, 8):
            if expected_occ is not None:
                around = [chess.square(f, 7 - r) for f in (i - 1, i) for r in (j - 1, j)]
                if any(expected_occ[s] for s in around):
                    continue
            x, y = int(round(i * _EDGE / 8.0)), int(round(j * _EDGE / 8.0))
            x0, y0, x1, y1 = x - hp - rad, y - hp - rad, x + hp + rad, y + hp + rad
            if x0 < 0 or y0 < 0 or x1 > WARP_SIZE or y1 > WARP_SIZE:
                continue
            templ = ref[y - hp:y + hp, x - hp:x + hp]
            if templ.std() < 8:                              # no pattern to lock on to
                continue
            res = cv2.matchTemplate(cur[y0:y1, x0:x1], templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if not np.isfinite(score) or score < REG_MIN_SCORE:
                continue
            px, py = loc
            if px in (0, res.shape[1] - 1) or py in (0, res.shape[0] - 1):
                continue                                      # peak on the rim: the true shift is larger
            dx = _parabola(res[py, px - 1], res[py, px], res[py, px + 1])
            dy = _parabola(res[py - 1, px], res[py, px], res[py + 1, px])
            src.append((x, y))
            shift.append((px + dx - rad, py + dy - rad))
    return np.array(src, np.float64).reshape(-1, 2), np.array(shift, np.float64).reshape(-1, 2)


def _parabola(a, b, c):
    d = a - 2 * b + c
    return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def register(cal, img, expected_occ=None, passes=2):
    """
    Measure where the board is now, compared with where the calibration
    expects it. The checker pattern around the inner grid corners of the
    empty reference is looked for in the current frame; squares expected to
    hold a piece are skipped (pass expected_occ), and patches that disagree
    with the rest (a piece, a hand, a shadow) are thrown out.

    Each pass can measure a shift of up to REG_SEARCH squares; the second
    pass starts from where the first one found the board. Nothing is
    changed; use apply_registration() to adopt the result.
    """
    ref = cal.ref_gray()
    start = cal.oriented_corners()
    corners = start.copy()
    result = None
    for _ in range(max(1, passes)):
        cur = _gray(warp_board(img, corners))
        src, shift = _measure_shifts(cur, ref, expected_occ)
        if len(src) < REG_MIN_POINTS:
            if result is not None:
                break
            return Registration(False, "only %d of the board's grid corners could be found" % len(src),
                                points=len(src))
        R, t, inl, rms = _robust_rigid(src, src + shift)
        if inl.sum() < REG_MIN_POINTS or inl.sum() < 0.5 * len(src):
            if result is not None:
                break
            return Registration(False, "the grid corners do not agree on where the board is",
                                points=len(src), inliers=int(inl.sum()), rms=rms)
        # where the four board corners are now, in the current warp, then in the image
        moved = _WARP_DST.astype(np.float64) @ R.T + t
        corners = transform_points(np.linalg.inv(warp_matrix(corners)), moved).astype(np.float32)
        result = (len(src), int(inl.sum()), rms)
        if np.abs(moved - _WARP_DST).max() < 0.5:
            break
    # express the total movement in the calibration's own warped frame
    now = transform_points(warp_matrix(start), corners)
    delta = (now - _WARP_DST) / SQ
    R, _ = _fit_rigid(_WARP_DST.astype(np.float64), now)
    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    centre = delta.mean(axis=0)
    return Registration(True, points=result[0], inliers=result[1], rms=result[2],
                        shift=float(np.linalg.norm(delta, axis=1).max()), angle=angle,
                        corners=corners, offset=(float(centre[0]), float(centre[1])))


def apply_registration(cal, reg):
    """Move the calibration's corners to where register() found the board."""
    if reg.ok and reg.corners is not None:
        cal.set_oriented_corners(reg.corners)


def track(cal, img, expected_occ=None, limit=REG_APPLY_MAX):
    """
    Follow the board: register() and, when the board has really moved (more
    than REG_APPLY_MIN but no more than `limit` squares), move the corners
    along. Returns (Registration, moved); moved tells whether the corners
    were changed, in which case earlier looks at the board were taken with
    the board in another place.
    """
    reg = register(cal, img, expected_occ)
    if not reg.ok or reg.shift < REG_APPLY_MIN or reg.shift > limit:
        return reg, False
    apply_registration(cal, reg)
    return reg, True


# ---------------------------------------------------------------- reading a move
def expected_state(board):
    occ = np.zeros(64, bool)
    white = np.zeros(64, bool)
    for sq, piece in board.piece_map().items():
        occ[sq] = True
        white[sq] = piece.color == chess.WHITE
    return occ, white


def state_score(occ, white, exp_occ, exp_white):
    """How well an observation matches an expected position (max 128). Ignores shadows."""
    occ_match = (occ == exp_occ)
    colour_match = occ_match & exp_occ & (white == exp_white)
    return int(occ_match.sum() + colour_match.sum())


def position_problems(cal, board, obs):
    """
    Squares that do not look the way `board` says they should.
    Returns a list of (square, text). A square hidden behind a piece that is
    expected to stand in front of it is given the benefit of the doubt.
    """
    pm = board.piece_map()
    shadows = cal.shadows()
    hidden = shadows.hidden(_pieces(pm))
    rivals = {c: shadows.hidden(_pieces(pm, not c)) for c in (chess.WHITE, chess.BLACK)}
    problems = []
    for t in chess.SQUARES:
        piece = pm.get(t)
        name = chess.square_name(t)
        if piece is None:
            seen = _visible_mean(obs.absdiff, t, hidden, most=True)    # the part no piece leans over
            if seen is not None and seen > cal.occ_threshold:
                problems.append((t, "%s should be empty" % name))
        elif not obs.occ[t]:
            problems.append((t, "%s should hold a %s %s" % (
                name, "white" if piece.color else "black", chess.piece_name(piece.piece_type))))
        else:
            # look at the piece where no piece of the other colour is in the way
            b = _piece_brightness(obs.gray, obs.absdiff, t, rivals[piece.color])
            if b is not None and (b > cal.white_threshold) != (piece.color == chess.WHITE):
                problems.append((t, "the piece on %s should be %s" % (name, "white" if piece.color else "black")))
    return problems


def change_map(obs, baseline):
    """
    How much every pixel changed between two looks at the board (grey levels).
    Two frames never line up to the pixel, and along the outline of a piece
    half a pixel is a big difference in grey level; whatever a misalignment
    of ALIGN_SLACK pixels could explain is therefore not counted as change.
    """
    slack = ALIGN_SLACK * np.maximum(obs.edges(), baseline.edges())
    return np.maximum(np.abs(obs.gray - baseline.gray) - slack, 0.0)


def changed_squares(before, after):
    """Squares whose content differs between two positions, with the piece types involved."""
    a, b = before.piece_map(), after.piece_map()
    out = {}
    for s in chess.SQUARES:
        if a.get(s) != b.get(s):
            out[s] = [p.piece_type for p in (a.get(s), b.get(s)) if p is not None]
    return out


def change_problems(cal, before, after, change):
    """
    Squares where what changed on the board does not fit the step from
    `before` to `after`: a square that should have changed but did not, or
    one that changed for no reason. Returns a list of (square, text).

    A piece that moves also changes the look of the squares it leans over,
    so those pixels are left out; and a square that should change is judged
    on the part that the pieces standing still do not hide.
    """
    return _change_fit(cal, before, after, change)[0]


def _change_fit(cal, before, after, change, seen=None):
    """change_problems() plus a number saying how well the change fits (higher is better)."""
    thr = cal.change_threshold
    expect = changed_squares(before, after)
    shadows = cal.shadows()
    a = before.piece_map()
    standing = [(s, p.piece_type) for s, p in a.items() if s not in expect]
    still = shadows.hidden(standing)
    moving = shadows.hidden([(s, ptype) for s, types in expect.items() for ptype in types])
    problems = []
    for t in chess.SQUARES:
        if t in expect:
            # The square a piece left may look just the same if the piece now leans over it
            # from the next square: leave out what the other squares of the move cover, too.
            cover = dict(still)
            for s, types in expect.items():
                if s != t:
                    for ptype in types:
                        for u, m in shadows.cover[(ptype, s)]:
                            if u == t:
                                cover[t] = (cover[t] | m) if t in cover else m
            v = _visible_mean(change, t, cover)
            if v is not None and v < thr:
                problems.append((t, "%s did not change" % chess.square_name(t)))
        else:
            v = _visible_mean(change, t, moving, most=True)
            if v is not None and v > thr:
                problems.append((t, "%s changed" % chess.square_name(t)))
    return problems, _silhouette_fit(shadows, expect, standing, change, thr, seen)


def _silhouette_fit(shadows, expect, standing, change, thr, seen=None):
    """
    Compare, pixel by pixel, where the picture changed with where it should
    have changed: the outlines of the pieces that were lifted or put down,
    as far as they are not behind a piece standing nearer to the camera.
    Returns squares' worth of agreement minus disagreement. This is what
    tells a knight from a pawn arriving on the same square when the squares
    they came from cannot be seen.
    """
    if not shadows.ready:
        return 0.0
    if seen is None:
        seen = change > 2.5 * thr
    hit = miss = 0
    done = []                                           # windows already counted, to avoid counting twice
    for s, types in expect.items():
        for ptype in types:
            x0, y0, x1, y1, m = shadows.blob(ptype, s)
            if m.size == 0:
                continue
            m = m.copy()
            for q, qtype in standing:                   # cut away what nearer pieces hide
                if shadows.depth[q] >= shadows.depth[s]:
                    continue
                a0, b0, a1, b1, mq = shadows.blob(qtype, q, True)
                u0, v0, u1, v1 = max(x0, a0), max(y0, b0), min(x1, a1), min(y1, b1)
                if u1 > u0 and v1 > v0:
                    m[v0 - y0:v1 - y0, u0 - x0:u1 - x0] &= ~mq[v0 - b0:v1 - b0, u0 - a0:u1 - a0]
            for c0, d0, c1, d1, mo in done:             # ...and what another moving piece already covers
                u0, v0, u1, v1 = max(x0, c0), max(y0, d0), min(x1, c1), min(y1, d1)
                if u1 > u0 and v1 > v0:
                    m[v0 - y0:v1 - y0, u0 - x0:u1 - x0] &= ~mo[v0 - d0:v1 - d0, u0 - c0:u1 - c0]
            window = seen[y0:y1, x0:x1]
            hit += int((window & m).sum())
            miss += int((~window & m).sum())
            done.append((x0, y0, x1, y1, m))
    # change outside every (generously drawn) outline of a moving piece is unexplained
    wide = np.zeros(seen.shape, bool)
    for s, types in expect.items():
        for ptype in types:
            x0, y0, x1, y1, m = shadows.blob(ptype, s, True)
            if m.size:
                wide[y0:y1, x0:x1] |= m
    stray = int((seen & ~wide).sum())
    return (hit - 0.3 * miss - stray) / float(SQ * SQ)


class Candidate:
    """A legal move and how badly it fits what the camera saw (0 = perfect)."""

    def __init__(self, move, state, change, fit=0.0):
        self.move = move
        self.state = state              # problems with the resulting position
        self.change = change            # problems with what changed
        self.total = len(state) + len(change)
        self.fit = fit                  # tie-break: how well the amount of change fits

    def __iter__(self):                 # for move, score in ranked
        return iter((self.move, -self.total))

    def __getitem__(self, i):
        return (self.move, -self.total)[i]


def rank_moves(cal, board, obs, baseline=None):
    """
    Rank the legal moves by how well each explains the board.
    Returns (candidates best first, the 'nothing was moved' candidate).
    """
    change = None if baseline is None else change_map(obs, baseline)
    seen = None if change is None else change > 2.5 * cal.change_threshold
    ranked = []
    for m in board.legal_moves:
        # All four promotions look identical to the camera; assume a queen.
        # An underpromotion can always be typed at the prompt.
        if m.promotion and m.promotion != chess.QUEEN:
            continue
        after = board.copy(stack=False)
        after.push(m)
        state = position_problems(cal, after, obs)
        chg, fit = ([], 0.0) if change is None else _change_fit(cal, board, after, change, seen)
        ranked.append(Candidate(m, state, chg, fit))
    ranked.sort(key=lambda c: (c.total, -c.fit))
    state = position_problems(cal, board, obs)
    chg, fit = ([], 0.0) if change is None else _change_fit(cal, board, board, change, seen)
    return ranked, Candidate(None, state, chg, fit)


def unchanged(cal, board, obs, baseline=None):
    """True when the board still shows `board` and nothing has changed since `baseline`."""
    if position_problems(cal, board, obs):
        return False
    if baseline is None:
        return True
    problems, fit = _change_fit(cal, board, board, change_map(obs, baseline))
    return not problems and fit > -QUIET_AREA


def decide(ranked, no_move, max_problems=MAX_PROBLEMS, min_lead=MIN_LEAD):
    """
    Pick the move from rank_moves(). Returns (move or None, confident, text).
    With confident False the move is only the best guess and should be
    confirmed by the player.
    """
    if not ranked:
        return None, False, "there is no legal move"
    best = ranked[0]
    second = ranked[1].total if len(ranked) > 1 else 999
    second_fit = ranked[1].fit if len(ranked) > 1 else -999.0
    # "nothing moved" fits when no square disagrees and hardly a pixel has changed; a move between
    # squares hidden behind other pieces changes no square but still shows above them
    quiet = no_move.total == 0 and no_move.fit > -QUIET_AREA
    if quiet or no_move.total + min_lead <= best.total:
        return None, False, "the board looks unchanged"
    text = "%d problem(s), runner-up %d, no move %d" % (best.total, second, no_move.total)
    lead = min(second, no_move.total) - best.total
    sure = best.total <= max_problems and lead >= min_lead and not best.move.promotion
    if (not sure and best.total == 0 and lead >= 1 and not best.move.promotion
            and best.fit >= max(second_fit, no_move.fit) + FIT_LEAD):
        sure = True         # explains every square, and the changed pixels clearly better than the rest
    if best.total > max_problems + 4:
        return None, False, "nothing fits what I see (%s)" % text
    return best.move, sure, text


def infer_move(board, occ, white):
    """
    Shadow-blind ranking kept for callers that only have occupancy and colour:
    returns a list of (move, score) sorted best first, plus the score of
    'no move happened'. Prefer rank_moves() + decide().
    """
    ranked = []
    for m in board.legal_moves:
        if m.promotion and m.promotion != chess.QUEEN:
            continue
        board.push(m)
        eo, ew = expected_state(board)
        board.pop()
        ranked.append((m, state_score(occ, white, eo, ew)))
    ranked.sort(key=lambda t: -t[1])
    eo, ew = expected_state(board)
    return ranked, state_score(occ, white, eo, ew)


# ---------------------------------------------------------------- drawing helpers
def draw_grid(warped, occ=None, white=None, board=None):
    out = warped.copy()
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = f * SQ, (7 - r) * SQ
        cv2.rectangle(out, (x0, y0), (x0 + SQ - 1, y0 + SQ - 1), (0, 200, 255), 1)
        cv2.putText(out, chess.square_name(sq), (x0 + 2, y0 + 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 200, 255), 1)
        if occ is not None and occ[sq]:
            colour = (255, 255, 255) if (white is not None and white[sq]) else (40, 40, 40)
            cv2.circle(out, (x0 + SQ // 2, y0 + SQ // 2), SQ // 4, colour, 2)
        if board is not None:
            p = board.piece_at(sq)
            if p:
                cv2.putText(out, p.symbol(), (x0 + SQ // 2 - 5, y0 + SQ - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def draw_problems(warped, problems):
    """Mark the squares listed by position_problems() / change_problems()."""
    out = warped.copy()
    for sq, _ in problems:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = f * SQ, (7 - r) * SQ
        cv2.rectangle(out, (x0 + 2, y0 + 2), (x0 + SQ - 3, y0 + SQ - 3), (0, 0, 255), 2)
    return out


def draw_corners(img, corners):
    out = img.copy()
    pts = np.array(corners, np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    for i, (x, y) in enumerate(np.array(corners).reshape(4, 2)):
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(out, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out
