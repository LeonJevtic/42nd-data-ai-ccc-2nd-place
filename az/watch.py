"""Benchmark new generations against the uploaded model and flag clear improvements.

python -m az.watch runs/r1 submit/model.onnx

Every `--every` generations: 100 referee games vs the uploaded model (colours alternate)
plus 40 late positions played both ways. Results go to <run>/bench.txt; a generation that
wins at least `--threshold` of the full games is reported as a CANDIDATE line.
"""
import argparse
import os
import re
import shutil
import time
from multiprocessing import Pool

from az import arena


def latest_gen(run):
    gens = [int(m.group(1)) for f in os.listdir(run) if (m := re.fullmatch(r"gen(\d+)\.onnx", f))]
    return max(gens) if gens else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("reference")
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.62)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--last", type=int, default=-1, help="last generation already benchmarked (default: newest)")
    a = p.parse_args()

    # snapshot the reference so later submit/ changes don't move the goalposts mid-run
    ref = os.path.join(a.run, "reference.onnx")
    shutil.copy(a.reference, ref)
    out = os.path.join(a.run, "bench.txt")
    last = a.last if a.last >= 0 else latest_gen(a.run)
    with Pool(a.workers) as pool:
        while True:
            n = latest_gen(a.run)
            if n < last + a.every:
                time.sleep(20)
                continue
            last = n
            cand = os.path.join(a.run, f"gen{n}.onnx")
            snap = os.path.join(a.run, f"bench_gen{n}.onnx")
            shutil.copy(cand, snap)
            full = arena.match(snap, ref, games=a.games, sims=100, pool=pool, seed=n * 101)
            late = arena.late_match(snap, ref, positions=40, sims=100, pool=pool, seed=n * 211)
            rate = full["a_wins"] / full["games"]
            tag = "CANDIDATE" if rate >= a.threshold else "bench"
            line = (f"{time.strftime('%H:%M:%S')} {tag} gen{n} vs reference: full {full['a_wins']}/{full['games']} "
                    f"(cyan {full['a_as_cyan']}, magenta {full['a_as_magenta']}), late {late['a_wins']}/{late['games']}")
            print(line, flush=True)
            with open(out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            try:
                os.remove(snap)
            except OSError:
                pass   # still held open by a worker's session on Windows


if __name__ == "__main__":
    main()
