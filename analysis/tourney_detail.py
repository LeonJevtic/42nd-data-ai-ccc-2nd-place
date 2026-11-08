"""Per-position outcomes of tourney_match: how many positions are decided by the position itself."""
import sys
from multiprocessing import Pool
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from az import arena

R = os.path.join(ROOT, "runs", "r1", "")
PATH = os.path.join(ROOT, "matches", "tourney_positions.npz")

if __name__ == "__main__":
    n, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    pairs = [p.split(":") for p in sys.argv[4:]]
    rnd = np.load(PATH)["round"]
    cand = np.flatnonzero((rnd >= lo) & (rnd <= hi))
    pick = np.random.default_rng(4242).choice(cand, n, replace=False)
    with Pool(6) as pool:
        for a, b in pairs:
            tasks = []
            for k, i in enumerate(pick):
                tasks.append((R + a + ".onnx", R + b + ".onnx", PATH, int(i), 4242 + k, 100))
                tasks.append((R + b + ".onnx", R + a + ".onnx", PATH, int(i), 4242 + k, 100))
            res = pool.map(arena.tourney_game, tasks, chunksize=1)
            g1, g2 = np.array(res[0::2]), np.array(res[1::2])   # mover wins with a as mover / b as mover
            mover_both = int(np.sum(g1 & g2)); other_both = int(np.sum(~g1 & ~g2))
            a_both = int(np.sum(g1 & ~g2)); b_both = int(np.sum(~g1 & g2))
            print(f"{a} vs {b} rounds {lo}-{hi}, {n} positions: decided-by-position {mover_both + other_both} "
                  f"(mover {mover_both}, other {other_both}); {a} wins both {a_both}, {b} wins both {b_both} "
                  f"-> score {a}: {(int(g1.sum()) + int((~g2).sum())) / (2 * n):.3f}", flush=True)
