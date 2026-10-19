#!/usr/bin/python3
# coding=utf8
"""
Spoken output for TonyPi chess.

Sentences are built by joining short pre-recorded wav fragments from
/home/pi/TonyPi/audio/chess (all 16 kHz mono 16-bit) into one temporary
wav, which is then played through the robot speaker with Hiwonder's
`speech.play_audio`, or `aplay` if that helper is not importable.

Fragments that have no recording are made with a text-to-speech program
(espeak-ng, espeak or pico2wave) when one is installed:
    python3 chess_speech.py --make-missing
writes them into the audio folder once (chess_setup.sh does this). Without
a recording and without text-to-speech the robot falls back to the nearest
sentence it does have; the full text is always printed on the console.
"""
import os
import wave
import shutil
import tempfile
import subprocess

import numpy as np
import chess

try:
    from speech import speech
    HAVE_SPEECH = True
except ImportError:
    HAVE_SPEECH = False

FRAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "chess")
FRAG_RATE = 16000

# every fragment the robot can say, with the words in it
FRAGMENT_TEXT = {
    "lets_play": "Let's play chess.",
    "i_play": "I play",
    "takes": "takes",
    "check": "check",
    "checkmate": "checkmate",
    "castles_kingside": "I castle king side",
    "castles_queenside": "I castle queen side",
    "en_passant": "en passant",
    "promotes_to_queen": "and promotes to a queen",
    "please_move_my_piece": "Please move my piece for me.",
    "your_move": "Your move.",
    "looking": "Let me look.",
    "did_not_understand": "I could not read your move. Please type it.",
    "good_game": "Good game.",
    "i_win": "Checkmate. I win. Good game.",
    "you_win": "You win. Well played.",
    "draw": "The game is a draw.",
    "pawn": "pawn", "knight": "knight", "bishop": "bishop", "rook": "rook", "queen": "queen", "king": "king",
    # added with the arms
    "keep_clear": "Keep your hands clear. I am moving my piece.",
    "please_fix": "Something is not right on the board. Please put it right for me.",
    "board_moved": "The board has moved. Please put it back where it was.",
    "take_piece": "Please take the piece out of my hand.",
    "swap_for_queen": "Please swap my pawn for a queen.",
    "is_this_your_move": "Is this your move?",
    "battery_low": "My battery is low.",
}
for _c in "abcdefgh12345678":
    FRAGMENT_TEXT[_c] = _c
# what to say instead when a fragment has no recording and cannot be synthesised
FALLBACK = {
    "keep_clear": [],
    "please_fix": ["please_move_my_piece"],
    "board_moved": ["did_not_understand"],
    "take_piece": ["please_move_my_piece"],
    "swap_for_queen": ["promotes_to_queen", "please_move_my_piece"],
    "is_this_your_move": ["your_move"],
    "battery_low": [],
}
PIECE_WORD = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
              chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
GAP_MS = 120
TRIM_LEVEL = 300            # samples below this count as silence when trimming
TRIM_KEEP_MS = 60           # silence kept on each side of a fragment
_missing_reported = set()
_no_player_reported = False


def _frag_path(key):
    return os.path.join(FRAG_DIR, "%s.wav" % key)


def _trim(frames, params):
    """Strip leading/trailing silence from 16-bit PCM frames, keeping a short margin."""
    if params.sampwidth != 2:
        return frames
    data = np.frombuffer(frames, np.int16)
    loud = np.where(np.abs(data) > TRIM_LEVEL)[0]
    if loud.size == 0:
        return frames
    keep = int(params.framerate * params.nchannels * TRIM_KEEP_MS / 1000)
    start = max(0, loud[0] - keep)
    end = min(data.size, loud[-1] + keep)
    return data[start:end].tobytes()


def tts_program():
    """The text-to-speech program installed on this machine, or None."""
    for name in ("espeak-ng", "espeak", "pico2wave"):
        if shutil.which(name):
            return name
    return None


def _synthesise(text, out_path):
    """Speak `text` into a 16 kHz mono 16-bit wav. Returns True on success."""
    prog = tts_program()
    if prog is None:
        return False
    fd, raw = tempfile.mkstemp(suffix=".wav", prefix="tonypi_tts_")
    os.close(fd)
    try:
        if prog == "pico2wave":
            cmd = [prog, "-l", "en-GB", "-w", raw, text]
        else:
            cmd = [prog, "-v", "en", "-s", "145", "-w", raw, text]
        subprocess.run(cmd, timeout=30, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(raw, "rb") as f:
            if f.getsampwidth() != 2:
                return False
            rate, channels = f.getframerate(), f.getnchannels()
            data = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float64)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        if data.size == 0:
            return False
        if rate != FRAG_RATE:
            n = int(round(data.size * FRAG_RATE / float(rate)))
            data = np.interp(np.linspace(0, data.size - 1, n), np.arange(data.size), data)
        tmp = out_path + ".tmp"
        with wave.open(tmp, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(FRAG_RATE)
            out.writeframes(np.clip(data, -32768, 32767).astype(np.int16).tobytes())
        os.replace(tmp, out_path)
        return True
    except Exception as e:
        print("chess_speech: text-to-speech failed: %s" % e)
        return False
    finally:
        try:
            os.remove(raw)
        except OSError:
            pass


def missing_fragments():
    return sorted(k for k in FRAGMENT_TEXT if not os.path.isfile(_frag_path(k)))


def make_missing():
    """Synthesise every fragment that has no recording. Returns (made, still missing)."""
    made, failed = [], []
    for key in missing_fragments():
        try:
            os.makedirs(FRAG_DIR, exist_ok=True)
            ok = _synthesise(FRAGMENT_TEXT[key], _frag_path(key))
        except OSError:
            ok = False
        (made if ok else failed).append(key)
    return made, failed


def _resolve(keys):
    """Replace fragments that cannot be played by their fallback."""
    out = []
    for key in keys:
        if os.path.isfile(_frag_path(key)):
            out.append(key)
        elif key in FRAGMENT_TEXT and tts_program() and _synthesise(FRAGMENT_TEXT[key], _frag_path(key)):
            out.append(key)
        else:
            out += [k for k in FALLBACK.get(key, [key]) if k != key or key not in FALLBACK]
    return out


def _join(keys, out_path):
    params = None
    with wave.open(out_path, "wb") as out:
        for key in keys:
            path = _frag_path(key)
            if not os.path.isfile(path):
                if key not in _missing_reported:
                    print("chess_speech: missing fragment %s" % path)
                    _missing_reported.add(key)
                continue
            with wave.open(path, "rb") as f:
                if params is None:
                    params = f.getparams()
                    out.setparams(params)
                    gap = b"\x00" * int(params.framerate * params.nchannels * params.sampwidth * GAP_MS / 1000)
                out.writeframes(_trim(f.readframes(f.getnframes()), params))
                out.writeframes(gap)
    return params is not None


def say(keys, text=None):
    """Speak the fragments in order. `text` is printed so the console mirrors the speaker."""
    if isinstance(keys, str):
        keys = [keys]
    global _no_player_reported
    if text:
        print("Robot: " + text)
    if not HAVE_SPEECH and shutil.which("aplay") is None:
        if not _no_player_reported:
            print("chess_speech: no speaker helper or aplay on this machine; sentences are printed only")
            _no_player_reported = True
        return
    try:
        keys = _resolve(keys)
    except Exception as e:                       # never let the speaker stop the game
        print("chess_speech: %s" % e)
    if not keys:
        return
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="tonypi_chess_")
    os.close(fd)
    try:
        if not _join(keys, tmp):
            return
        if HAVE_SPEECH:
            speech.play_audio(tmp)
        else:
            subprocess.run(["aplay", "-q", tmp], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("chess_speech: playback failed: %s" % e)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def set_volume(level):
    if HAVE_SPEECH:
        try:
            speech.set_volume(level)
        except Exception:
            pass


def move_fragments(board, move):
    """Fragments describing `move` from `board` (board is BEFORE the move). Returns (keys, text)."""
    keys, words = [], []
    if board.is_castling(move):
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            keys.append("castles_kingside"); words.append("I castle king side")
        else:
            keys.append("castles_queenside"); words.append("I castle queen side")
    else:
        piece = board.piece_at(move.from_square)
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        keys += ["i_play", PIECE_WORD[piece.piece_type], frm[0], frm[1]]
        words.append("I play %s %s" % (PIECE_WORD[piece.piece_type], frm))
        if board.is_capture(move):
            keys.append("takes"); words.append("takes")
        keys += [to[0], to[1]]
        words.append(to)
        if board.is_en_passant(move):
            keys.append("en_passant"); words.append("en passant")
        if move.promotion:
            keys.append("promotes_to_queen"); words.append("and promotes to a queen")
    after = board.copy()
    after.push(move)
    if after.is_checkmate():
        keys.append("checkmate"); words.append("checkmate")
    elif after.is_check():
        keys.append("check"); words.append("check")
    return keys, " ".join(words)


def announce_move(board, move):
    keys, text = move_fragments(board, move)
    say(keys, text)


if __name__ == "__main__":
    import sys
    if "--make-missing" in sys.argv:
        if tts_program() is None:
            print("No text-to-speech program found (sudo apt install espeak-ng). Missing: %s"
                  % (", ".join(missing_fragments()) or "nothing"))
            sys.exit(0 if not missing_fragments() else 1)
        made, failed = make_missing()
        print("made: %s" % (", ".join(made) or "nothing"))
        if failed:
            print("could not make: %s" % ", ".join(failed))
        sys.exit(1 if failed else 0)
    b = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nxe5", "Nxe5", "O-O"]:
        m = b.parse_san(san)
        print(san, "->", move_fragments(b, m))
        b.push(m)
