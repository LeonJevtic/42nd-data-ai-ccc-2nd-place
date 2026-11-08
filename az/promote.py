"""Benchmark new generations and promote clear improvements to the best model, unattended.

python -m az.promote runs/r1 runs/r1/gen44.onnx --last 47

For the newest generation: 100 referee games vs the current best (stage 1); at >= threshold a
second 100 games on another seed (stage 2); then 60 games against each pool model (stage 3),
which must not fall clearly below the best's pool score. With --tourney-positions the first two
stages are replaced by games from real tournament positions (each played with the models swapped;
the edge on the positions they split must be --tourney-z standard errors) plus a start-position
match that only has to reach --min-start: start-position matches between related nets are
dominated by a few deterministic lines. A promoted model passes the
validator, is copied to <out>/model_level4_genN.onnx and model_level4_best.onnx, and becomes
the new reference. Every result goes to <run>/bench.txt, the current state to <out>/BEST.txt.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

from az import arena
from az.net import clip_value_file

POOL_SEED = 7000   # same seeds as the manual pool gauntlets earlier today


def latest_gen(run):
    gens = [int(m.group(1)) for f in os.listdir(run) if (m := re.fullmatch(r"gen(\d+)\.onnx", f))]
    return max(gens) if gens else 0


def pool_score(model, pool_models, games, workers_pool):
    wins = n = 0
    parts = []
    for i, opp in enumerate(pool_models):
        r = arena.match(model, opp, games=games, sims=100, pool=workers_pool, seed=POOL_SEED + 1000 * i)
        wins += r["a_wins"]
        n += r["games"]
        parts.append(f"{os.path.basename(opp)[:-5]} {r['a_wins']}/{r['games']}")
    return wins / n, ", ".join(parts)


def validate(path, root):
    for _ in range(2):   # the validator can reject transiently when the CPU is saturated
        out = subprocess.run([sys.executable, os.path.join(root, "tools", "validate_submission.py"), path],
                             capture_output=True, text=True).stdout
        if "ACCEPTED" in out:
            return True
        time.sleep(30)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("best")
    p.add_argument("--last", type=int, default=-1, help="last generation already benchmarked (default: newest)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.62)
    p.add_argument("--pool", default="gen26,gen30,gen33")
    p.add_argument("--pool-games", type=int, default=60)
    p.add_argument("--pool-margin", type=float, default=0.03, help="allowed drop vs the best's pool score")
    p.add_argument("--best-pool", type=float, default=-1.0, help="known pool score of the best (else measured)")
    p.add_argument("--out", default="models")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tourney-positions", type=int, default=0,
                   help="main test: this many real tournament positions (0 = start-position matches)")
    p.add_argument("--tourney-file", default="matches/tourney_positions.npz")
    p.add_argument("--tourney-rounds", default="3,6")
    p.add_argument("--tourney-z", type=float, default=2.0, help="required edge in standard errors")
    p.add_argument("--min-start", type=float, default=0.45, help="minimum start-position score vs the best")
    a = p.parse_args()
    a.tourney_rounds = tuple(int(x) for x in a.tourney_rounds.split(","))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench = os.path.join(a.run, "bench.txt")
    pool_models = [os.path.join(a.run, g + ".onnx") for g in a.pool.split(",")]

    def note(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(bench, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def set_best(gen_name, src, pool, info):
        shutil.copy(src, ref)
        shutil.copy(src, os.path.join(a.out, "model_level4_best.onnx"))
        with open(os.path.join(a.out, "BEST.txt"), "w", encoding="utf-8") as f:
            f.write(f"best: {gen_name} (since {time.strftime('%H:%M:%S')})\n"
                    f"file: {os.path.join(a.out, 'model_level4_best.onnx')}\n"
                    f"pool score: {pool:.3f}\n{info}\n")

    ref = os.path.join(a.run, "reference.onnx")
    best_name = os.path.basename(a.best)[:-5]
    last = a.last if a.last >= 0 else latest_gen(a.run)
    with Pool(a.workers) as pool:
        best_pool = a.best_pool
        if best_pool < 0:
            best_pool, detail = pool_score(a.best, pool_models, a.pool_games, pool)
            note(f"promote: pool score of {best_name} = {best_pool:.3f} ({detail})")
        set_best(best_name, a.best, best_pool, "start of promote run")
        note(f"--- promote run: best {best_name}, pool {best_pool:.3f}, next gen > {last} ---")
        while True:
            n = latest_gen(a.run)
            if n <= last:
                time.sleep(20)
                continue
            last = n
            snap = os.path.join(a.run, f"bench_gen{n}.onnx")
            shutil.copy(os.path.join(a.run, f"gen{n}.onnx"), snap)
            if a.tourney_positions:
                # real tournament positions: significant edge on the positions the two models split
                t = arena.tourney_match(snap, ref, positions=a.tourney_positions, sims=100, pool=pool,
                                        seed=n * 13, path=a.tourney_file, rounds=a.tourney_rounds)
                st = f"positions {t['a_both']}:{t['b_both']} of {a.tourney_positions}"
                if t["a_both"] - t["b_both"] < a.tourney_z * max(1.0, (t["a_both"] + t["b_both"]) ** 0.5):
                    note(f"bench gen{n} vs {best_name}: {st}")
                    continue
                r1 = arena.match(snap, ref, games=a.games, sims=100, pool=pool, seed=n * 101)
                s1 = f"start {r1['a_wins']}/{r1['games']} (cyan {r1['a_as_cyan']}, magenta {r1['a_as_magenta']})"
                if r1["a_wins"] / r1["games"] < a.min_start:
                    note(f"CANDIDATE gen{n} vs {best_name}: {st}; {s1} -> too exploitable from the start")
                    continue
                s2 = st
            else:
                r1 = arena.match(snap, ref, games=a.games, sims=100, pool=pool, seed=n * 101)
                s1 = f"{r1['a_wins']}/{r1['games']} (cyan {r1['a_as_cyan']}, magenta {r1['a_as_magenta']})"
                if r1["a_wins"] / r1["games"] < a.threshold:
                    note(f"bench gen{n} vs {best_name}: {s1}")
                    continue
                r2 = arena.match(snap, ref, games=a.games, sims=100, pool=pool, seed=n * 101 + 50021)
                s2 = f"{r2['a_wins']}/{r2['games']} (cyan {r2['a_as_cyan']}, magenta {r2['a_as_magenta']})"
                if r2["a_wins"] / r2["games"] < a.threshold:
                    note(f"CANDIDATE gen{n} vs {best_name}: {s1}; second seed {s2} -> not confirmed")
                    continue
            ps, detail = pool_score(snap, pool_models, a.pool_games, pool)
            if ps < best_pool - a.pool_margin:
                note(f"CANDIDATE gen{n} vs {best_name}: {s1}; {s2}; pool {ps:.3f} < {best_pool:.3f} ({detail}) -> rejected")
                continue
            clip_value_file(snap, snap)   # the running trainer may export without the value clip
            if not validate(snap, root):
                note(f"CANDIDATE gen{n}: validator REJECTED -> not promoted")
                continue
            dst = os.path.join(a.out, f"model_level4_gen{n}.onnx")
            shutil.copy(snap, dst)
            info = f"vs {best_name}: {s1}; {s2}; pool {ps:.3f} ({detail})"
            set_best(f"gen{n}", snap, ps, info)
            note(f"PROMOTED gen{n} {info} -> {dst}")
            best_name, best_pool = f"gen{n}", ps


if __name__ == "__main__":
    main()
