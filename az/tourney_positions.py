"""Extract every post-opening position from the downloaded tournament games as self-play starts.

Lost games are repeated so the positions strong opponents steered us into come up more often.

python -m az.tourney_positions   ->  matches/tourney_positions.npz
"""
import glob
import json

import numpy as np

from az.replay import parse, replay
from az.regress import TXT_GAMES

LOSS_WEIGHT = 3


def all_games():
    seen = set()
    for f in sorted(glob.glob("matches/tournament-matches-*.json")):
        for m in json.load(open(f, encoding="utf-8")):
            if "protocol" in m and m["match_id"] not in seen:
                seen.add(m["match_id"])
                acts, winner = parse(m["protocol"])
                yield acts, winner != m["you"]
    for f in sorted(glob.glob("matches/match-*.txt")):
        mid = int(f.split("-")[-1].split(".")[0])
        if mid in seen:
            continue
        acts, winner = parse(open(f, encoding="utf-8").read())
        ours = TXT_GAMES.get(f.split("\\")[-1].split("/")[-1])
        yield acts, ours is not None and winner != ours


EVAL_EVERY = 5   # every 5th game is held out for az.promote's tournament-position matches


def main():
    sets = {"train": ([], [], [], []), "eval": ([], [], [], [])}
    n_games = n_lost = 0
    for g, (acts, lost) in enumerate(all_games()):
        n_games += 1
        n_lost += lost
        held_out = g % EVAL_EVERY == EVAL_EVERY - 1
        obs, team, left, rnd = sets["eval" if held_out else "train"]
        for env, action in replay(acts):
            if action is None or env.action <= 8:
                continue
            # held-out positions are not repeated: the evaluation weights every position once
            for _ in range(LOSS_WEIGHT if lost and not held_out else 1):
                obs.append(np.rint(env.observation() * 8).astype(np.uint8))
                team.append(0 if env.current_team == "CYAN" else 1)
                left.append(env.actions_left)
                rnd.append(env.round_number)
    for name, path in (("train", "matches/tourney_positions.npz"), ("eval", "matches/tourney_eval.npz")):
        obs, team, left, rnd = sets[name]
        np.savez(path, obs=np.stack(obs), team=np.array(team, np.int8),
                 actions_left=np.array(left, np.int8), round=np.array(rnd, np.int16))
        print(f"{name}: {len(obs)} positions -> {path}")
    print(f"{n_games} games ({n_lost} lost)")


if __name__ == "__main__":
    main()
