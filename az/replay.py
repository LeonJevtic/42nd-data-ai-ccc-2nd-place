"""Replay tournament protocols and work out which side was our model.

After the 8-action warmup the referee plays greedily with no noise, so our model's
moves can be reproduced exactly by running the same search with the same network.

python -m az.replay matches/match-5035.txt [model.onnx ...]
"""
import re
import sys

import numpy as np

from colony_one import ColonyOneEnv, mcts
from colony_one.rules import legal_actions_for_team

LINE = re.compile(r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w\w)\s+([A-H][1-8])\s*->\s*([A-H][1-8])\s+(MOVE|SHOOT)")


def square(s):
    return (8 - int(s[1]), ord(s[0]) - ord("A"))


def parse(text):
    """Returns [(team, from_pos, to_pos)] and the winner named in the protocol."""
    acts, winner = [], None
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            acts.append((m.group(2), square(m.group(4)), square(m.group(5))))
        elif "WINS" in line:
            winner = "CYAN" if "CYAN WINS" in line else "MAGENTA"
    return acts, winner


def replay(acts):
    """Yields (env_before, action) for each protocol action; env is a snapshot clone."""
    env = ColonyOneEnv()
    for team, fr, to in acts:
        assert env.current_team == team, (team, env.current_team)
        match = [a for a in legal_actions_for_team(env.board, team) if a.from_pos == fr and a.to_pos == to]
        assert len(match) == 1, (fr, to)
        yield env.clone(), match[0]
        env.step(match[0])
    yield env, None


def identify(acts, models, sims=100):
    """For each model and colour: how many post-warmup moves the referee search reproduces."""
    from az.selfplay import get_eval
    out = {}
    for path in models:
        ev = get_eval(path)
        agree = {"CYAN": [0, 0], "MAGENTA": [0, 0]}
        for env, action in replay(acts):
            if action is None or env.action <= 8:
                continue
            a = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number,
                                   ev, mcts.MCTSConfig(simulations=sims), temperature=0.0,
                                   rng=np.random.default_rng(0))
            agree[env.current_team][0] += int(a == action)
            agree[env.current_team][1] += 1
        out[path] = agree
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    models = sys.argv[2:] or ["runs/r1/gen4.onnx", "runs/r1/gen1.onnx"]
    acts, winner = parse(open(path, encoding="utf-8").read())
    *_, (final, _) = replay(acts)
    print(path, "winner", winner, "engine", final.winner, "actions", len(acts))
    for m, ag in identify(acts, models).items():
        print("  ", m, {k: f"{a}/{n}" for k, (a, n) in ag.items()})
