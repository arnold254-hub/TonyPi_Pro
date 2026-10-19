# TonyPi plays chess

TonyPi stands in front of a real chessboard, watches it with the camera in
its head, works out the move you made, thinks with **Stockfish** (or a
built-in search when Stockfish is not installed), says its reply, **picks
its piece up and puts it down on the new square**, and then looks again to
check that the board shows what it should.

## Read this first: what has and has not been tried

Everything here has been run against a *simulated* robot on a PC
(`chess_selftest.py`: camera at an angle, upright pieces, lens distortion,
two arms, a table on which pieces can be knocked over). **None of it has
moved the real robot yet.** Expect the first session to be spent on setup,
not on chess. In particular:

| Unknown until you try it on the robot | Where you find out |
|---|---|
| How many squares the arms reach. A TonyPi arm is about 18 cm long with 3 joints; with a normal board it will reach the ranks nearest to it and not the far side. | `test` in `chess_teach.py` prints the reach map |
| Whether the gripper can hold your pieces, and how far to close it | step 2 of the wizard |
| Which of the two torque commands makes this firmware's servos go limp | the wizard asks "is the arm limp?" and tries the other one |
| The head tilt that shows the whole board | step 1 of the wizard |
| Battery thresholds (`battery_warn_mv` 10500, `battery_min_mv` 9800 are guesses for a 3-cell pack) | `chess_arm.yaml` |
| How well the camera reads *your* pieces under *your* light | step 2 of `ChessPlay.py` reports it |

Pieces the arms cannot reach are not a failure: the robot says its move and
asks you to make it, then checks the board. With `--no-arms` it does that
for every move.

## Files

| File | Purpose |
|------|---------|
| `ChessPlay.py` | The program you run to play. |
| `chess_teach.py` | Teaches the arms where the squares are. Run once before the first game. |
| `chess_arm.py` | Arms, grippers, head, stance; turns a chess move into pick-and-place steps. |
| `chess_vision.py` | Finds the board, follows it when it is nudged, reads the position and the move. |
| `chess_engine.py` | Stockfish over UCI, with a pure-Python alpha-beta fallback. |
| `chess_speech.py` | Builds spoken sentences from the wav fragments in `../audio/chess/`. |
| `chess_sim.py` | The pretend robot used by the self-test. Not used in a game. |
| `chess_selftest.py` | Tests everything on any PC. |
| `chess_setup.sh` | One-time check and installer. |
| `chess/` | The python-chess library (rules of chess), bundled so that the robot needs no internet. GPL-3, see `chess/LICENSE.txt`. |

Written at runtime (in `/home/pi/TonyPi/`):

| File | Content |
|---|---|
| `chess_arm.yaml` | everything `chess_teach.py` taught; plain text, can be edited |
| `chess_config.yaml`, `chess_reference.png` | where the board is in the picture, what it looks like empty |
| `chess_debug/*.jpg` | what the camera saw (`camera`, `last_read`, `start_position`, `empty_board`) |
| `chess_debug/chess.log` | everything the program decided and why |
| `chess_games/*.pgn` | every game; `unfinished.pgn` is the game in progress |

## The table

The robot does not walk during a game. It has to stand in **the same place
relative to the board every time**, because the arms go where they were
taught.

* Put the board on something that raises it to about the height of the
  robot's hips (a box, a few books). The hands hang below the board when the
  arms are at rest and the shoulders are not far above the pieces.
* Push the board up close: the near edge a few centimetres in front of the
  robot's toes.
* A small board helps: squares of 2 to 2.5 cm, pieces no taller than about
  1.5 squares, and slim enough that the open gripper goes over them with
  room to spare. Light pieces (plastic, not weighted).
* **Mark the place** of the robot's feet and of the board's corners with
  tape. The camera notices when the board is up to about one square out of
  place and the arms follow it, but the closer to the marks, the better.
* Put a small box on each side of the board for captured pieces.
* Even, diffuse light. No hard shadows across the board, no glare.

## Install (once, on the robot)

Copy the whole `Functions` folder (including `Functions/chess/`) and
`audio/chess/` to the robot. The folder is called `TonyPi_Code` on the PC
and `TonyPi` on the robot; that does not matter, the program finds its files
from where it is. With WinSCP, drag from the left side (PC) to the right
(robot):

| On the PC | Goes into, on the robot |
|---|---|
| `TonyPi_Code\Functions\chess` (folder, the chess rules) | `/home/pi/TonyPi/Functions/` |
| `TonyPi_Code\Functions\chess_*.py`, `ChessPlay.py`, `chess_setup.sh` | `/home/pi/TonyPi/Functions/` |
| `TonyPi_Code\audio\chess` (folder, the voice) | `/home/pi/TonyPi/audio/` |

There are two folders called `chess`: keep them apart. Then:

```bash
cd /home/pi/TonyPi/Functions
bash chess_setup.sh          # checks that everything needed is there
python3 chess_selftest.py    # everything should print ok (about 5 minutes on the robot)
```

**No internet is needed.** The robot is normally its own Wi-Fi hotspot and
cannot download anything; everything required comes with the project. Two
extras are only installed when the robot is online (`chess_setup.sh` says
which are missing):

* **Stockfish**, a strong opponent. Without it the robot plays with its
  built-in engine, which is a beginner.
* **espeak-ng**, to speak the seven new sentences. Without it the robot says
  the nearest recorded sentence and prints the full text.

To get them, connect the robot to the internet once (network cable, or put
it on your home Wi-Fi with the Hiwonder Wi-Fi settings) and run
`bash chess_setup.sh` again.

If `chess_setup.sh` fills the screen with `Temporary failure in name
resolution` and `No matching distribution found for chess`, the robot still
has an old copy of the script: the current one starts with
`== TonyPi chess setup (version 2026-09-29, needs no internet) ==` and
downloads nothing without a connection. Upload the folder again.

## Before every session

```bash
sudo systemctl stop tonypi   # the app service shares the servo port and the camera
```

(`sudo systemctl start tonypi` afterwards brings the phone app back.)

If you forget, `ChessPlay.py` and `chess_teach.py` notice that the service is
running and offer to stop it; press Enter to accept. With the service
running the two programs fight over the serial port to the servos, and the
game used to end with `PortNotOpenError: Attempting to use a port that is
not open`. Commands now wait for the port, but the arms are only reliable
with the service stopped.

To see what the camera sees, add `--web` to either program and open
`http://<address of the robot>:8081/` in a browser.

## Teach the arms (once)

```bash
cd /home/pi/TonyPi/Functions
python3 chess_teach.py --web
teach> wizard
```

The wizard walks through:

1. **Stance and view.** The robot stands (`stand_slow`), looks down. Change
   the head tilt until the whole board is in the picture with a little room
   around it. Then the **empty** board is photographed.
2. **Gripper.** How far to open, how far to close on a pawn, how far to close
   when empty.
3. **Path.** A few way-points that take the hand from hanging at the side
   to above the near edge of the board *without passing through the pieces*:
   usually elbow out to the side first, then forearm forward. You hold the
   arm (it goes limp), put it at each way-point and press Enter.
4. **Squares.** For each square you teach: the open gripper around the
   piece ("grasp"), then straight above it, high enough that a carried piece
   clears the king ("hover"). Start with the corners of what the arm can
   reach, then fill in. Squares in between are estimated from their
   neighbours; in the simulation every other square taught gives about a
   tenth of a square of error, every square taught gives none. **Teach every
   square your own pieces start on if you can spare the time.**
5. **Bin.** Where the hand lets go of a captured piece.
6. Back to rest.

Then try it before trusting it:

```
teach> test                 reach map, and how well in-between squares are estimated
teach> goto right b7        hand above b7 (look: is it centred?)
teach> goto right b7 grasp  ...and down around the piece
teach> take right b7        pick it up
teach> put right b6         put it down
teach> move g8 f6           the way a game does it, choosing the arm itself
teach> rest
teach> save
```

If a square is off, put the arm right by hand (`limp right`, move it,
`square right b7 grasp`) or nudge it (`jog right 2 -10`, then `square ...`).

The taught places are tied to the table, not to square names, so the same
teaching serves when the robot plays the other colour. They are also tied to
**the direction of the head**: the arms find the squares through the camera.
Changing the tilt afterwards means teaching again.

SAFETY: an arm drops when it goes limp, so hold it first. Keep fingers out
of the gripper. `rest` brings both arms down the way they came.

## Play

```bash
cd /home/pi/TonyPi/Functions
python3 ChessPlay.py                       # you are white, robot is black
python3 ChessPlay.py --white               # robot plays white and starts
python3 ChessPlay.py --check-arms          # first show that the arms are still on target
python3 ChessPlay.py --auto                # no Enter key: the robot notices your move
python3 ChessPlay.py --skill 3 --time 1    # easier and faster opponent
python3 ChessPlay.py --no-arms             # the robot only talks; you move its pieces
python3 ChessPlay.py --resume              # carry on with the interrupted game
python3 ChessPlay.py --recalibrate         # photograph the empty board again
```

At the start the robot asks for the **starting position** and learns what
occupied / white / black squares look like in today's light. It should
report 32 of 32 pieces and 16 of 16 white ones. It then prints which squares
its arms reach.

Every turn:

1. Make your move and press Enter (or just take your hand away with `--auto`).
2. The robot reads the move. If it is not sure it asks
   *"Did you play Nf3?"*: Enter for yes, or type the move. This happens
   mostly when a piece on your side of the board moves from behind your
   pawns, where the camera cannot see it.
3. The robot says its move. It beeps and says *keep your hands clear*, and
   moves the piece: captured piece into the box first, then the piece
   itself; castling is king then rook; a piece that has to cross from one
   arm's side to the other's is put down on a free square in the middle and
   picked up by the other arm. For a promotion it moves the pawn and asks you
   to swap it for a queen.
4. It looks at the board. If something is not where it should be it tells
   you which square and waits for you to put it right.

At any prompt you can type:

- a move such as `e2e4`, `Nf3` or `O-O`,
- `u` to undo the last pair of moves (the robot shows the position to restore),
- `ok` when the robot complains about the board and the board is in fact right,
- `q` to resign and quit. Ctrl-C stops at once; the arms go back to rest and
  the game can be resumed.

## What the robot checks by itself

| Check | What happens |
|---|---|
| Board out of place | up to about 1 square: followed, arms included. More: you are asked to put it back. |
| Board further than `max_drift` (0.75 squares) from where the arms were taught | arms stay down, you move the pieces |
| Battery below `battery_min_mv` | arms stay down |
| A servo does not arrive where it was sent (blocked) | the move stops, the arm backs out the way it came, you finish the move |
| Gripper closed on nothing (if `feel` found that the gripper can tell) | same |
| Piece left in the hand after a failure | you are asked to take it; the hand opens when you press Enter |
| Three arm failures in a row | the arms are switched off for the rest of the game |
| Board does not show the position after a move | you are told which squares and asked to fix them |
| Servos do not answer at all | moves are made without the arrival check, with a warning |

## Engine

Stockfish is used automatically when it can be found: on PATH,
`/usr/games/stockfish` (where `apt install stockfish` puts it), a few other
common directories, or the path given by `--stockfish` / the
`STOCKFISH_PATH` environment variable. It runs with 2 threads and a 64 MB
hash to leave CPU for the camera, and the `--skill` value (0-20) maps to
Stockfish's *Skill Level*. If Stockfish crashes mid-game the program
switches to the built-in engine and the game continues.

## When something does not work

| Symptom | Try |
|---|---|
| "Automatic detection failed" in STEP 1 | whole board in the picture? (`--web`) Glare? Otherwise type the corner coordinates read from `camera.jpg`. |
| Fewer than 32 pieces detected | more light, less shadow; dark pieces on dark squares need contrast |
| The robot often asks "Did you play ...?" | normal for pieces behind your pawns; a steeper view (robot higher, head tilted further down) helps |
| The robot reads moves wrongly without asking | send `chess_debug/chess.log` and `last_read.jpg`; raise `MIN_LEAD` in `chess_vision.py` |
| "... should be empty" although it is | a tall piece next to it; type `ok`. If it keeps happening raise `PIECE_HEIGHT` (king height in squares) in `chess_vision.py` |
| Hand not centred over the square | board or robot off its marks; otherwise teach that square again |
| Hand knocks pieces on the way in | the path's last way-point is too low, or `hover` was taught too low |
| Piece slips out of the gripper | `grip right <number>` a little tighter; rubber on the jaws |
| Gripper servo gets hot | `grip` too tight, or `relax` presses the jaws together: set it a little more open |
| "servo N does not answer" | `sudo systemctl stop tonypi` |
| Arms move too fast / too slowly | `speed`, `slow_speed` (pulses per second) in `chess_arm.yaml` |
| Tall pieces are gripped too low | `grasp_lift` in `chess_arm.yaml`: 0 = at the taught height, 1 = at hover height, per piece letter |

Run with `--no-robot --camera 0` on a laptop with a webcam to try the
vision without the robot.
