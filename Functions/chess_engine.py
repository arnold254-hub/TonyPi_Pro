#!/usr/bin/python3
# coding=utf8
"""
Chess engine wrapper for TonyPi.

Uses Stockfish through python-chess (UCI) when it is installed on the robot,
otherwise falls back to a built-in alpha-beta search that runs fine on a
Raspberry Pi. Both are driven through the same `best_move(board)` call, and
if Stockfish dies mid-game the wrapper switches to the built-in search so the
game can go on.

Installing Stockfish on the robot:
    sudo apt install stockfish          # installs /usr/games/stockfish
Debian puts it in /usr/games, which is often not on PATH for ssh / service
sessions, so that directory is searched explicitly. A binary anywhere else
can be given with --stockfish PATH or the STOCKFISH_PATH environment variable.

Requires python-chess, which comes with the project as the folder Functions/chess.
"""
import os
import glob
import time
import shutil

import chess

try:
    import chess.engine
    HAVE_ENGINE_API = True
except ImportError:
    HAVE_ENGINE_API = False

# ------------------------------------------------------------ built-in engine
PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# Piece-square tables from white's point of view, index 0 = a1 ... 63 = h8.
_PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0]
_PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50]
_PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20]
_PST_ROOK = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0]
_PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20]
_PST_KING = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30]
PST = {
    chess.PAWN: _PST_PAWN, chess.KNIGHT: _PST_KNIGHT, chess.BISHOP: _PST_BISHOP,
    chess.ROOK: _PST_ROOK, chess.QUEEN: _PST_QUEEN, chess.KING: _PST_KING,
}
MATE_SCORE = 100000


def evaluate(board):
    """Static evaluation from the side to move's point of view (centipawns)."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        table = PST[piece.piece_type]
        pst = table[square] if piece.color == chess.WHITE else table[chess.square_mirror(square)]
        if piece.color == chess.WHITE:
            score += value + pst
        else:
            score -= value + pst
    return score if board.turn == chess.WHITE else -score


def _order_moves(board):
    def key(m):
        s = 0
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square)
            attacker = board.piece_type_at(m.from_square)
            s += 10 * PIECE_VALUE.get(victim, 100) - PIECE_VALUE.get(attacker, 0)
        if m.promotion:
            s += 800
        if board.gives_check(m):
            s += 50
        return -s
    return sorted(board.legal_moves, key=key)


class _Timeout(Exception):
    pass


class BuiltinEngine:
    """Iterative-deepening negamax with alpha-beta and a quiescence search."""

    name = "TonyPi built-in engine"

    def __init__(self, time_limit=2.0, max_depth=5):
        self.time_limit = time_limit
        self.max_depth = max_depth
        self._deadline = 0
        self.nodes = 0

    def _check_time(self):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.time() > self._deadline:
            raise _Timeout()

    def _quiesce(self, board, alpha, beta, depth=0):
        self._check_time()
        stand = evaluate(board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        if depth >= 4:
            return alpha
        for m in _order_moves(board):
            if not board.is_capture(m) and not m.promotion:
                continue
            board.push(m)
            score = -self._quiesce(board, -beta, -alpha, depth + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _negamax(self, board, depth, alpha, beta):
        self._check_time()
        if depth == 0:
            return self._quiesce(board, alpha, beta)
        if board.is_game_over():
            return evaluate(board)
        best = -MATE_SCORE - 1
        for m in _order_moves(board):
            board.push(m)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def best_move(self, board):
        board = board.copy()          # a timeout can interrupt mid-line; never touch the caller's board
        self._deadline = time.time() + self.time_limit
        self.nodes = 0
        moves = _order_moves(board)
        if not moves:
            return None
        best_move = moves[0]
        try:
            for depth in range(1, self.max_depth + 1):
                alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
                depth_best, depth_move = -MATE_SCORE - 1, best_move
                for m in moves:
                    board.push(m)
                    score = -self._negamax(board, depth - 1, -beta, -alpha)
                    board.pop()
                    if score > depth_best:
                        depth_best, depth_move = score, m
                    if depth_best > alpha:
                        alpha = depth_best
                best_move = depth_move
                # search the previous best first at the next depth
                moves.remove(best_move)
                moves.insert(0, best_move)
                if abs(depth_best) >= MATE_SCORE:
                    break
        except _Timeout:
            pass
        return best_move

    def quit(self):
        pass


# ------------------------------------------------------------ Stockfish
STOCKFISH_CANDIDATES = [
    "/usr/games/stockfish",             # Debian / Raspberry Pi OS package
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/stockfish/stockfish",
    os.path.expanduser("~/stockfish/stockfish"),
]


def find_stockfish(path=None):
    """Return a runnable Stockfish binary, or None. Explicit path > env > PATH > known dirs."""
    candidates = [path, os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish")]
    candidates += STOCKFISH_CANDIDATES
    candidates += sorted(glob.glob(os.path.expanduser("~/stockfish*/stockfish*")))
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _builtin_depth(skill):
    return 3 if skill < 5 else (4 if skill < 15 else 5)


class StockfishEngine:
    """Stockfish over UCI. Falls back to BuiltinEngine if the process dies."""

    def __init__(self, path, time_limit=1.0, skill=10, threads=None, hash_mb=64):
        self.path = path
        self.time_limit = time_limit
        self.skill = int(max(0, min(20, skill)))
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.name = self.engine.id.get("name", "Stockfish")
        if threads is None:
            threads = max(1, min(2, (os.cpu_count() or 1) - 1))   # leave a core for vision / servos
        wanted = {"Threads": threads, "Hash": hash_mb, "Skill Level": self.skill}
        for key, value in wanted.items():
            if key in self.engine.options:
                try:
                    self.engine.configure({key: value})
                except chess.engine.EngineError as e:
                    print("chess_engine: could not set %s=%s (%s)" % (key, value, e))
        self.limit = chess.engine.Limit(time=time_limit)
        self.fallback = None
        self.nodes = 0

    def best_move(self, board):
        if self.fallback is not None:
            return self.fallback.best_move(board)
        try:
            result = self.engine.play(board, self.limit)
            return result.move
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, OSError) as e:
            print("chess_engine: Stockfish failed (%s); continuing with the built-in engine" % e)
            self.quit()
            self.fallback = BuiltinEngine(time_limit=self.time_limit, max_depth=_builtin_depth(self.skill))
            self.name = self.fallback.name
            return self.fallback.best_move(board)

    def quit(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def make_engine(time_limit=2.0, skill=10, engine="auto", path=None):
    """
    Return an engine object with best_move(board), quit() and name.
      engine = "auto"       Stockfish if it can be found, else built-in
      engine = "stockfish"  Stockfish only; raises RuntimeError if it is not available
      engine = "builtin"    the pure-Python search
    """
    engine = (engine or "auto").lower()
    if engine not in ("auto", "stockfish", "builtin"):
        raise ValueError("engine must be auto, stockfish or builtin")
    if engine != "builtin":
        found = find_stockfish(path) if HAVE_ENGINE_API else None
        if found:
            try:
                eng = StockfishEngine(found, time_limit=time_limit, skill=skill)
                print("chess_engine: using %s at %s (skill %d, %.1fs per move)"
                      % (eng.name, found, eng.skill, time_limit))
                return eng
            except Exception as e:
                msg = "Stockfish at %s failed to start: %s" % (found, e)
                if engine == "stockfish":
                    raise RuntimeError(msg)
                print("chess_engine: " + msg + "; using built-in engine")
        else:
            msg = "Stockfish not found (install with: sudo apt install stockfish, or pass --stockfish PATH)"
            if engine == "stockfish":
                raise RuntimeError(msg)
            print("chess_engine: " + msg)
    depth = _builtin_depth(skill)
    print("chess_engine: using built-in engine (time %.1fs, depth %d)" % (time_limit, depth))
    return BuiltinEngine(time_limit=time_limit, max_depth=depth)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "auto"
    eng = make_engine(time_limit=2.0, engine=which)
    b = chess.Board()
    t = time.time()
    m = eng.best_move(b)
    print("opening move:", b.san(m), "in %.2fs" % (time.time() - t))
    # mate in one test: white to move, Qh5xf7#
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    m = eng.best_move(b)
    print("mate-in-one found:", b.san(m), "(expected Qxf7#)")
    eng.quit()
