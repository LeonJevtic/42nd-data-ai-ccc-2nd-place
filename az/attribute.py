"""Attribute downloaded tournament games to our model versions and tally results.

After the 8-action warmup the referee is deterministic, so the model whose search
reproduces all of our post-warmup moves is the one that played.

python -m az.attribute matches/tournament-matches-3.json [...] -- runs/r1/gen4.onnx runs/r1/gen10.onnx
"""
import json
import sys
from collections import defaultdict

import numpy as np

from colony_one import mcts

from az.replay import parse, replay
from az.selfplay import get_eval


def our_moves_match(acts, ours, model):
    ev = get_eval(model)
    agree = total = 0
    for env, action in replay(acts):
        if action is None or env.action <= 8 or env.current_team != ours:
            continue
        a = mcts.search_action(env.board, ours, env.actions_left, env.round_number, ev,
                               mcts.MCTSConfig(simulations=100), temperature=0.0,
                               rng=np.random.default_rng(0))
        agree += int(a == action)
        total += 1
    return agree, total


def end_kind(protocol):
    if "BARE KERNEL" in protocol:
        return "bare-kernel"
    if any("FIREWALL" in line and "xKE" in line for line in protocol.splitlines()):
        return "firewall"
    return "kernel-capture"


if __name__ == "__main__":
    sep = sys.argv.index("--")
    files, models = sys.argv[1:sep], sys.argv[sep + 1:]
    matches = {}
    for f in files:
        for m in json.load(open(f, encoding="utf-8")):
            matches[m["match_id"]] = m
    tally = defaultdict(lambda: {"CYAN": [0, 0], "MAGENTA": [0, 0]})
    rows = []
    for mid in sorted(matches):
        m = matches[mid]
        if "protocol" not in m:   # e.g. "opponent unavailable" games
            continue
        acts, winner = parse(m["protocol"])
        ours = m["you"]
        best, best_score = "unknown/L3", -1.0
        for model in models:
            a, n = our_moves_match(acts, ours, model)
            score = a / n if n else 0.0
            if n and a == n and score > best_score:
                best, best_score = model, score
                break
        won = winner == ours
        t = tally[best][ours]
        t[0] += int(won)
        t[1] += 1
        rows.append((mid, m["played_at"][11:19], ours, "W" if won else "L", best.split("/")[-1],
                     end_kind(m["protocol"]), m["opponent"] or "?"))
    for r in rows:
        print(*r, sep="\t")
    print()
    for model, t in tally.items():
        w = t["CYAN"][0] + t["MAGENTA"][0]
        n = t["CYAN"][1] + t["MAGENTA"][1]
        print(f"{model}: {w}/{n} won  (CYAN {t['CYAN'][0]}/{t['CYAN'][1]}, MAGENTA {t['MAGENTA'][0]}/{t['MAGENTA'][1]})")
