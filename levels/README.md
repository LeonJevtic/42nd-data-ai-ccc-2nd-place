# Levels 1-3

The contest had four levels on the same game. Levels 1-3 introduced the rules step by step; Level 4 was the live
tournament that the rest of this repository is about. These are the small programs we used for the first two levels.
Neither needs any package beyond Python 3.

| Level | Task | Our approach |
|---|---|---|
| 1, "In a Pinch" | Kernel and Ping only. Find a Cyan turn (two actions) after which Magenta cannot capture the Cyan Kernel with its two actions. | [`level1.py`](level1.py): brute force. Tries every Cyan turn and every Magenta reply. Among the safe turns it prefers one that keeps our Ping and takes theirs, and it takes the enemy Kernel directly if it can. |
| 2, "New Friends" | Still Kernel and Ping, but with the full rules: bodyguards, the firewall and the bare-Kernel endgame. Find the best Cyan turn for a given round. | [`level2.py`](level2.py): alpha-beta search over whole turns (default depth 3: us, them, us) with an evaluation of material, Kernel safety and centralisation. |
| 3 | The first model level: upload an ONNX network, games played with the referee search at 60 simulations; each side has a Kernel, a Ping and an Exploit. | Not included. Level 4 builds on the same idea at full scale. |

## Run

```bash
python levels/level1.py levels/examples/level1-board.txt
python levels/level2.py levels/examples/level2-board.txt      # optional 2nd argument: search depth in turns
```

Board format: 8 lines of 8 characters, `K`/`P` for the Cyan Kernel/Ping, `k`/`p` for Magenta's, `.` for empty
(`#` for squares already removed by the firewall). For Level 2 the first line is the current round number.
Squares are printed chess-style: files a-h from left to right, rank 8 at the top.
