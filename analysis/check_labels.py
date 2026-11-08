"""Brute-force check of az.tactics.wins_this_turn against the real env (all action sequences)."""
import random
import sys

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one.env import ColonyOneEnv  # noqa: E402
from az.tactics import wins_this_turn  # noqa: E402


def truth(env):
    team = env.current_team
    for a in env.legal_action_indices():
        e = env.clone()
        e.step(int(a))
        if e.terminated:
            if e.winner == team:
                return True
            continue
        if e.current_team != team:
            continue
        for b in e.legal_action_indices():
            f = e.clone()
            f.step(int(b))
            if f.terminated and f.winner == team:
                return True
    return False


def main(n, seed):
    rng = random.Random(seed)
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    bad = []
    done = 0
    while done < n:
        env = ColonyOneEnv()
        env.reset()
        stop = rng.randint(10, 200)
        k = 0
        while not env.terminated and k < stop:
            env.step(env.sample_legal_action_index())
            k += 1
        if env.terminated:
            continue
        lab = wins_this_turn(env.board, env.current_team, env.actions_left, env.round_number)
        t = truth(env)
        key = ("t" if lab == t else "f") + ("p" if lab else "n")
        stats[key] += 1
        if lab != t and len(bad) < 5:
            bad.append((env.round_number, env.current_team, env.actions_left, lab, t, env.render()))
        done += 1
    print(stats)
    for b in bad:
        print(b[:5])
        print(b[5])


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
