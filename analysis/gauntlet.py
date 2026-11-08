import sys
from multiprocessing import Pool
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from az import arena

R = os.path.join(ROOT, "runs", "r1", "")
CANDS = sys.argv[1].split(",")
POOL = sys.argv[2].split(",")
GAMES = int(sys.argv[3])

if __name__ == "__main__":
    tot = {c: [0, 0] for c in CANDS}
    with Pool(8) as p:
        for i, opp in enumerate(POOL):
            for c in CANDS:
                r = arena.match(R + c + ".onnx", R + opp + ".onnx", games=GAMES, sims=100, pool=p, seed=7000 + 1000 * i)
                tot[c][0] += r["a_wins"]; tot[c][1] += r["games"]
                print(f"{c} vs {opp}: {r['a_wins']}/{r['games']} (cyan {r['a_as_cyan']}, magenta {r['a_as_magenta']})", flush=True)
    for c, (w, n) in tot.items():
        print(f"TOTAL {c}: {w}/{n} = {w / n:.1%}", flush=True)
