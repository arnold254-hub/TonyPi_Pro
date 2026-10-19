#!/bin/bash
# One-time setup for TonyPi chess (run on the robot as the pi user):
#     cd /home/pi/TonyPi/Functions && bash chess_setup.sh
# Checks that everything the chess program needs has arrived on the robot.
# With internet it also installs Stockfish (recommended) and a text-to-speech
# program (for the sentences that have no recording).
set -u
cd "$(dirname "$0")" || exit 1
# The date tells an old copy of this script from a new one on a photo of the screen.
echo "== TonyPi chess setup (version 2026-09-29, needs no internet) =="

# The robot is usually its own Wi-Fi hotspot and has no internet. Everything
# that is NEEDED comes with the project; the downloads below are extras.
if getent hosts deb.debian.org >/dev/null 2>&1 || getent hosts pypi.org >/dev/null 2>&1; then
    online=1
    echo "internet: connected"
else
    online=0
    echo "internet: not connected (normal while the robot is a hotspot); skipping downloads"
fi

# The files are copied to the robot by hand, so see first that all of them arrived.
lost=""
for f in ChessPlay.py chess_vision.py chess_engine.py chess_speech.py chess_arm.py chess_teach.py \
         chess_sim.py chess_selftest.py chess/__init__.py chess/engine.py chess/pgn.py \
         CameraCalibration/calibration_param.npz ../audio/chess ../ActionGroups/stand_slow.d6a; do
    [ -e "$f" ] || lost="$lost $f"
done
if [ -z "$lost" ]; then
    echo "files: all there"
else
    echo "files: NOT ON THE ROBOT:$lost"
    echo "       Copy them from the PC (WinSCP: drag the folder from the left side to the"
    echo "       right). Functions/chess goes into Functions, audio/chess goes into audio."
fi

where=$(python3 -c "import chess, os; print(chess.__version__, os.path.dirname(chess.__file__))" 2>/dev/null)
if [ -n "$where" ]; then
    echo "python-chess: $where"
elif [ $online -eq 1 ]; then
    echo "python-chess: installing..."
    python3 -m pip install chess 2>/dev/null || python3 -m pip install --break-system-packages chess || sudo python3 -m pip install chess
else
    echo "python-chess: MISSING. The folder Functions/chess was not copied to the robot;"
    echo "              upload it from the PC (it is part of the project)."
fi
if ! python3 -c "import chess" 2>/dev/null; then
    echo "stockfish, speech: cannot be looked at without python-chess"
    echo
    echo "Some dependencies are missing; see above."
    exit 1
fi

found=$(python3 -c "import sys; sys.path.insert(0, '.'); import chess_engine; print(chess_engine.find_stockfish() or '')" 2>/dev/null)
if [ -n "$found" ]; then
    echo "stockfish: already installed ($found)"
elif [ $online -eq 0 ]; then
    echo "stockfish: not installed. The robot will use its built-in engine, which plays a"
    echo "           beginner's game. For a stronger opponent connect the robot to the"
    echo "           internet once and run this script again."
else
    echo "stockfish: installing with apt (needs sudo)..."
    if sudo apt-get install -y stockfish; then
        echo "stockfish: installed at /usr/games/stockfish"
    else
        echo "stockfish: apt install failed. The robot will use the built-in engine."
        echo "           You can also download an ARM build from https://stockfishchess.org/download/"
        echo "           and run ChessPlay.py --stockfish /path/to/stockfish"
    fi
fi

missing=$(python3 -c "import sys; sys.path.insert(0, '.'); import chess_speech; print(' '.join(chess_speech.missing_fragments()))" 2>/dev/null)
if [ -z "$missing" ]; then
    echo "speech: every sentence has its audio"
else
    echo "speech: no audio yet for: $missing"
    if ! command -v espeak-ng >/dev/null 2>&1 && ! command -v espeak >/dev/null 2>&1 && ! command -v pico2wave >/dev/null 2>&1; then
        if [ $online -eq 1 ]; then
            echo "speech: installing espeak-ng with apt (needs sudo)..."
            sudo apt-get install -y espeak-ng || echo "speech: apt install failed."
        fi
    fi
    python3 chess_speech.py --make-missing || echo "speech: the robot says the nearest recorded sentence instead; the full text is on the screen."
fi

echo
echo "== checking imports =="
python3 - <<'PY'
import importlib, sys
ok = True
for mod in ("cv2", "numpy", "yaml", "serial", "chess", "chess.engine", "chess.pgn"):
    try:
        importlib.import_module(mod)
        print("  %-13s ok" % mod)
    except Exception as e:
        print("  %-13s MISSING (%s)" % (mod, e)); ok = False
sys.path.insert(0, ".")
for mod in ("chess_vision", "chess_arm", "chess_speech", "chess_engine"):
    try:
        importlib.import_module(mod)
        print("  %-13s ok" % mod)
    except Exception as e:
        print("  %-13s failed to import: %s" % (mod, e)); ok = False
try:
    import chess_engine
    p = chess_engine.find_stockfish()
    print("  stockfish     %s" % (p if p else "not found -> built-in engine will be used"))
    import chess_arm
    cfg = chess_arm.load_config()
    if cfg is None or not chess_arm.is_taught(cfg):
        print("  arms          not taught yet -> run: python3 chess_teach.py")
    else:
        n = {a: len(cfg["arms"][a]["points"]) for a in chess_arm.ARM_NAMES}
        print("  arms          taught (right %(right)d squares, left %(left)d)" % n)
except Exception as e:
    print("  check failed: %s" % e); ok = False
sys.exit(0 if ok else 1)
PY
status=$?
echo
if [ $status -eq 0 ]; then
    echo "Setup complete."
    echo "  1. Test without the robot moving:   python3 chess_selftest.py"
    echo "  2. Stop the app service:            sudo systemctl stop tonypi"
    echo "  3. Teach the arms (once):           python3 chess_teach.py --web"
    echo "  4. Play:                            python3 ChessPlay.py"
    echo "Read chess_README.md first: it says how to set up the table."
else
    echo "Some dependencies are missing; see above."
fi
exit $status
