Stockfish 19 for the TonyPi robot (official static builds, no apt needed).

1. On the robot terminal run:   uname -m
     aarch64  -> use the 64bit folder
     armv7l   -> use the 32bit folder
2. In WinSCP copy the file  stockfish  from that folder to  /home/pi/stockfish/stockfish
   (create the folder stockfish in /home/pi if it does not exist).
3. On the robot terminal run:
     chmod +x /home/pi/stockfish/stockfish
     /home/pi/stockfish/stockfish bench 2>&1 | tail -3
   You should see "Nodes/second". Then ChessPlay.py finds it automatically.
