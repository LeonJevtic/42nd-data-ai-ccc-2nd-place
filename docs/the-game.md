# The game: Colony: One

Level 4 of the 42nd Cloudflight Coding Contest (Data/AI track, 23 Oct 2026) was a live bot tournament. The game,
*Colony: One*, is a two-player strategy game on an 8x8 board. This page summarises the rules in our own words, as far
as you need them to follow the rest of the docs. The official rules and the engine came with the organizer's starter kit,
which is not part of this repository.

## Basics

- Two teams, **Cyan** and **Magenta**. Cyan moves first.
- Each team starts with six pieces: **Ping, Daemon, Exploit, Trace, Ice, Kernel**.
- A **turn is two actions** by the same team. A round is Cyan's turn followed by Magenta's.
- There is no passing, no draw and no repetition rule. Every game ends with a winner.
- You lose when your **Kernel** is destroyed.

## Pieces (short version)

| Piece | Moves | Captures |
|---|---|---|
| Ping | up to 2 squares orthogonally | diagonally, up to 2 squares |
| Daemon | slides up to 3 squares orthogonally | shoots an adjacent enemy; can also capture by moving 2-3 squares |
| Exploit | an L-shaped jump that ignores pieces in between | by landing on the target |
| Trace | 1 square orthogonally | a diagonal shot that reaches 2 or 3 squares away |
| Ice | up to 2 squares in any direction | by landing on the target |
| Kernel | 1 square in any direction | by landing on the target |

## The rules that decide most games

- **Bodyguards.** If a Kernel is attacked while a friendly non-Kernel piece stands next to it, that neighbour is
  sacrificed instead and the attacker stays where it was. An unguarded Kernel that is hit is gone, and so is the game.
  With two actions per turn, many attacks are "remove the guard, then hit the Kernel".
- **Firewall.** After rounds 8, 16, 24 and 26 the outermost open ring of the board is locked
  (8x8 → 6x6 → 4x4 → 2x2 → nothing). Everything standing on a locked square is destroyed. A Kernel that is still
  on the ring when it locks loses the game on the spot. The last lock after round 26 ends every game, so a game has at
  most 26 rounds (104 actions).
- **Bare Kernels.** If only the two Kernels are left at the end of a turn: Kernels within two squares of each other →
  the side that just moved loses; otherwise the Kernel closer to the centre wins; on a tie Magenta wins.
- Magenta wins every tie-break, which compensates for Cyan moving first.

## The tournament

- Each team uploads **one ONNX file** (at most 5 MiB). It must take the position as a `(batch, 16, 8, 8)` tensor and
  return `policy` logits over 4096 actions (from-square x to-square) and a `value` in `[-1, 1]`.
- The server does not run your code. It plays every game with the **same referee MCTS** for both sides, using each
  team's network to guide its own search: PUCT with c = 1.5 and **100 simulations per action**.
- To vary the games, the referee samples the first 8 actions of a game from the visit counts (temperature 1.0 for
  actions 1-4, 0.5 for actions 5-8) and plays greedily afterwards.
- Ratings use **Glicko-2**. You can upload again at any time; the rating is kept, but the new model starts a fresh
  placement phase and its rating swings more at first. So uploading only made sense for clear improvements.
- The tournament ran live during the 4-hour contest window, and the leaderboard at the end decided the placings.

Because the search is fixed and identical for everyone, the whole competition is about the quality of the network:
how good its move priors are (the search only looks at 100 positions per action) and how well its value head judges a
position.
