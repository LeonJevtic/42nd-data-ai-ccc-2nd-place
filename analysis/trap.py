"""Replay a lost match up to our turn at protocol action `start`, then let each model play our side
greedily (referee search) for one turn and report whether the opponent can then kill at once.

python trap.py match.json match_id start model1.onnx [model2.onnx ...]
"""
import json
import sys

import numpy as np

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one import mcts  # noqa: E402
from az.replay import parse, replay  # noqa: E402
from az.selfplay import get_eval  # noqa: E402
from az.tactics import can_kill  # noqa: E402


def sq(p):
    return "ABCDEFGH"[p[1]] + str(8 - p[0])


def main():
    path, mid, start, models = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    if path.endswith(".txt"):
        acts, _ = parse(open(path, encoding="utf-8").read())
    else:
        m = next(x for x in json.load(open(path, encoding="utf-8")) if x["match_id"] == mid)
        acts, _ = parse(m["protocol"])
    env0 = next(e for e, a in replay(acts) if e.action == start)
    team = env0.current_team
    enemy = "CYAN" if team == "MAGENTA" else "MAGENTA"
    # how many of our turns are safe at all
    safe = 0
    total = 0
    for a1 in env0.legal_actions():
        e1 = env0.clone()
        e1.step(a1)
        if e1.terminated:
            continue
        for a2 in e1.legal_actions():
            e2 = e1.clone()
            e2.step(a2)
            total += 1
            safe += int(not e2.terminated and not can_kill(e2.board, enemy, 2))
    print(f"match {mid} action {start}: we are {team}; safe turns {safe}/{total}")
    for path_m in models:
        ev = get_eval(path_m)
        env = env0.clone()
        moves = []
        for _ in range(2):
            a = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number, ev,
                                   mcts.MCTSConfig(simulations=100), temperature=0.0,
                                   rng=np.random.default_rng(0))
            moves.append(f"{sq(a.from_pos)}->{sq(a.to_pos)}")
            env.step(a)
            if env.terminated:
                break
        lost = (not env.terminated) and can_kill(env.board, enemy, 2)
        print(f"  {path_m.split('/')[-1]:14s} plays {' '.join(moves):14s} -> enemy can kill next turn: {lost}")


if __name__ == "__main__":
    main()
