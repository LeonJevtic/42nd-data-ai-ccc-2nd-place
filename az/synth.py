"""Synthetic exactly-labelled positions for the kill head, focused on firewall deaths.

Self-play rarely reaches 'Magenta to move in a shrink round with the Cyan Kernel on the ring
about to lock', so the kill head never learned it. This builds such positions directly
(plus near misses and ordinary late positions) and labels them with wins_this_turn.

python -m az.synth 120000   ->  matches/synth_kill.npz
"""
import sys
from multiprocessing import Pool

import numpy as np

from colony_one import ColonyOneEnv, constants, mcts
from colony_one.pieces import PieceType

from az.gpu_selfplay import _place_late
from az.tactics import wins_this_turn


def _ring(r, c):
    return min(r, c, 7 - r, 7 - c)


def make(seed):
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv()
    rnd = int(rng.choice(constants.SHRINK_ROUNDS)) if rng.random() < 0.7 else int(rng.integers(5, 27))
    team = constants.MAGENTA if rng.random() < 0.7 else constants.CYAN
    _place_late(env, rng, rnd, team)
    b = env.board
    nxt = b.next_open_ring()
    if nxt >= 0 and rng.random() < 0.5:
        # move the Cyan Kernel onto (or right next to) the ring that locks next
        target_ring = nxt if rng.random() < 0.7 else min(nxt + 1, 3)
        squares = [(r, c) for r in range(8) for c in range(8)
                   if _ring(r, c) == target_ring and b.grid[r][c] is None and not b.locked[r][c]]
        kp = next(((r, c) for r in range(8) for c in range(8)
                   if b.grid[r][c] is not None and b.grid[r][c].team == constants.CYAN
                   and b.grid[r][c].piece_type is PieceType.KERNEL), None)
        if squares and kp:
            r, c = squares[int(rng.integers(len(squares)))]
            b.grid[r][c], b.grid[kp[0]][kp[1]] = b.grid[kp[0]][kp[1]], None
    env.round_number, env.current_team, env.actions_left = rnd, team, int(rng.choice((1, 2)))
    env.action = mcts.move_number(team, env.actions_left, rnd)
    label = wins_this_turn(b, team, env.actions_left, rnd)
    return np.rint(env.observation() * 8).astype(np.uint8), label


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    with Pool(4) as pool:
        res = pool.map(make, range(10_000_000, 10_000_000 + n), chunksize=500)
    obs = np.stack([o for o, _ in res])
    lab = np.array([k for _, k in res], np.int8)
    np.savez("matches/synth_kill.npz", aux_obs=obs, aux_kill=lab)
    print(f"{n} positions, {int(lab.sum())} positive ({lab.mean():.1%})")
