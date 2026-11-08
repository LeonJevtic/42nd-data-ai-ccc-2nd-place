"""For each of our post-warmup actions in a lost match: what the live model played, what gen25 would play,
and both root values (Q of the chosen action, side to move's view).

python cmp_games.py <ours_model.onnx> <new_model.onnx> match-1.txt [match-2.txt ...]
"""
import sys

import numpy as np

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one import encoding, mcts  # noqa: E402
from az.replay import parse, replay  # noqa: E402
from az.selfplay import get_eval  # noqa: E402


def sq(p):
    return "ABCDEFGH"[p[1]] + str(8 - p[0])


def search(env, ev):
    root = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev,
                    mcts.MCTSConfig(simulations=100), rng=np.random.default_rng(0))
    idx = mcts.select_action_index(root, 0.0, rng=np.random.default_rng(0))
    a = encoding.decode_action(idx, env.current_team, env.board)
    return a, root.W[idx] / max(root.N[idx], 1)


def main():
    ours_m, new_m, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    ev_o, ev_n = get_eval(ours_m), get_eval(new_m)
    for f in files:
        acts, winner = parse(open(f, encoding="utf-8").read())
        steps = list(replay(acts))
        ours = "CYAN" if winner == "MAGENTA" else "MAGENTA"
        rows, agree, total = [], 0, 0
        for env, action in steps:
            if action is None or env.action <= 8 or env.current_team != ours:
                continue
            a_o, q_o = search(env, ev_o)
            a_n, q_n = search(env, ev_n)
            total += 1
            agree += int(a_o == action)
            mark = "" if a_n == action else "  <-- new model differs"
            rows.append(f"  {env.action:3d} r{env.round_number:<2d} played {sq(action.from_pos)}->{sq(action.to_pos)} "
                        f"Q={q_o:+.2f} | new {sq(a_n.from_pos)}->{sq(a_n.to_pos)} Q={q_n:+.2f}{mark}")
        print(f"== {f}: we are {ours}, winner {winner}, live model reproduces {agree}/{total} of our moves")
        print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
