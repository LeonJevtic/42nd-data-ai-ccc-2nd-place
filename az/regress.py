"""Regression test built from lost tournament games.

Finds every position where our real turn either let the opponent destroy our Kernel on
its next turn, or triggered a losing bare-Kernels resolution, keeps those where a safe
turn existed, and checks whether a model's referee search now avoids the mistake.

python -m az.regress runs/r1/gen10.onnx runs/r1/gen13.onnx ...
"""
import json
import sys

import numpy as np

from colony_one import constants, mcts
from colony_one.rules import legal_actions_for_team

from az.replay import parse, replay
from az.tactics import can_kill, firewall_doom

# our colour in the downloaded single-match files (identified by exact move matching)
TXT_GAMES = {"match-4988.txt": "CYAN", "match-5268.txt": "MAGENTA",
             "match-7173.txt": "MAGENTA", "match-7206.txt": "CYAN",
             "match-9569.txt": "MAGENTA", "match-9751.txt": "MAGENTA", "match-9852.txt": "CYAN",
             "match-10228.txt": "MAGENTA", "match-10233.txt": "CYAN",
             "match-11149.txt": "CYAN", "match-11172.txt": "MAGENTA",
             "match-11684.txt": "CYAN", "match-11736.txt": "CYAN", "match-11796.txt": "MAGENTA",
             "match-14732.txt": "CYAN", "match-15141.txt": "MAGENTA", "match-15256.txt": "CYAN",
             "match-19969.txt": "CYAN", "match-20113.txt": "MAGENTA", "match-20166.txt": "CYAN",
             "match-21909.txt": "MAGENTA"}


def games():
    for name, ours in TXT_GAMES.items():
        acts, winner = parse(open(f"matches/{name}", encoding="utf-8").read())
        if winner != ours:
            yield name, acts, ours
    ms = {m["match_id"]: m for f in ("matches/tournament-matches-2.json", "matches/tournament-matches-3.json")
          for m in json.load(open(f, encoding="utf-8")) if "protocol" in m}
    for m in ms.values():
        acts, winner = parse(m["protocol"])
        if winner != m["you"]:
            yield f"json-{m['match_id']}", acts, m["you"]


def _turn_fails(env, team):
    """After `team`'s finished turn (env at the opponent's turn start): did it blunder?"""
    if env.terminated:
        return env.winner != team
    opp = constants.MAGENTA if team == constants.CYAN else constants.CYAN
    return can_kill(env.board, opp, 2) or firewall_doom(env.board, opp, env.round_number)


def _safe_turn_exists(env, team):
    for a1 in legal_actions_for_team(env.board, team):
        e1 = env.clone()
        e1.step(a1)
        if e1.terminated:
            if e1.winner == team:
                return True
            continue
        for a2 in legal_actions_for_team(e1.board, team):
            e2 = e1.clone()
            e2.step(a2)
            if not _turn_fails(e2, team):
                return True
    return False


def cases():
    out = []
    for name, acts, ours in games():
        snaps = list(replay(acts))
        for i, (env, action) in enumerate(snaps):
            if action is None or env.current_team != ours or env.actions_left != 2:
                continue
            j = i + 2 if i + 2 < len(snaps) else len(snaps) - 1
            after = snaps[j][0]
            if after.current_team == ours and not after.terminated:
                continue
            if _turn_fails(after, ours) and _safe_turn_exists(env, ours):
                out.append((f"{name}@{i + 1}", env, ours))
    return out


def check(model, tests):
    from az.selfplay import get_eval
    ev = get_eval(model)
    passed = []
    for label, env, team in tests:
        e = env.clone()
        while not e.terminated and e.current_team == team:
            a = mcts.search_action(e.board, team, e.actions_left, e.round_number, ev,
                                   mcts.MCTSConfig(simulations=100), temperature=0.0,
                                   rng=np.random.default_rng(0))
            e.step(a)
        passed.append((label, not _turn_fails(e, team)))
    return passed


if __name__ == "__main__":
    tests = cases()
    print(f"{len(tests)} avoidable blunder positions:", [t[0] for t in tests])
    for model in sys.argv[1:]:
        res = check(model, tests)
        print(f"{model}: {sum(ok for _, ok in res)}/{len(res)} avoided  "
              + " ".join(f"{l}:{'ok' if ok else 'X'}" for l, ok in res), flush=True)
