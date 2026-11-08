"""Match two evaluators under referee settings.

python -m az.arena A B [games] [sims]     (A/B: path.onnx | heuristic | uniform)
"""
import sys
from multiprocessing import Pool

from az.selfplay import arena_game


def late_game(args):
    """Referee-style game from a random late position (endgame / firewall skill test)."""
    import numpy as np
    from colony_one import constants, mcts
    from az.gpu_selfplay import random_late_env
    from az.selfplay import get_eval
    spec_c, spec_m, seed, sims = args
    env = random_late_env(np.random.default_rng(seed))
    evs = {constants.CYAN: get_eval(spec_c), constants.MAGENTA: get_eval(spec_m)}
    rng = np.random.default_rng(seed + 1)
    while not env.terminated:
        team = env.current_team
        a = mcts.search_action(env.board, team, env.actions_left, env.round_number,
                               evs[team], mcts.MCTSConfig(simulations=sims), rng=rng)
        if a is None:
            return constants.MAGENTA if team == constants.CYAN else constants.CYAN, 0
        env.step(a)
    return env.winner, 0


def late_match(a, b, positions=50, sims=100, pool=None, seed=0):
    """Each random late position is played twice with colours swapped: score for `a`."""
    tasks = []
    for i in range(positions):
        tasks.append((a, b, seed + i, sims))
        tasks.append((b, a, seed + i, sims))
    res = pool.map(late_game, tasks, chunksize=1)
    a_c = sum(1 for g, (w, _) in enumerate(res) if g % 2 == 0 and w == "CYAN")
    a_m = sum(1 for g, (w, _) in enumerate(res) if g % 2 == 1 and w == "MAGENTA")
    return {"a_wins": a_c + a_m, "games": len(res), "a_as_cyan": f"{a_c}/{positions}",
            "a_as_magenta": f"{a_m}/{positions}"}


_POSITIONS = {}


def tourney_game(args):
    """Referee-style game from a real tournament position; True if `spec_mover` (side to move) wins."""
    import numpy as np
    from colony_one import ColonyOneEnv, constants, mcts
    from colony_one.board import Board
    from az.selfplay import get_eval
    spec_mover, spec_other, path, i, seed, sims = args
    if path not in _POSITIONS:
        d = np.load(path)
        _POSITIONS[path] = {k: d[k] for k in d.files}
    d = _POSITIONS[path]
    team = constants.CYAN if d["team"][i] == 0 else constants.MAGENTA
    other = constants.MAGENTA if team == constants.CYAN else constants.CYAN
    env = ColonyOneEnv()
    env.board = Board.from_observation(d["obs"][i].astype(np.float32) / 8.0, team)
    env.current_team = team
    env.actions_left = int(d["actions_left"][i])
    env.round_number = int(d["round"][i])
    env.action = mcts.move_number(team, env.actions_left, env.round_number)
    evs = {team: get_eval(spec_mover), other: get_eval(spec_other)}
    rng = np.random.default_rng(seed)
    while not env.terminated:
        t = env.current_team
        a = mcts.search_action(env.board, t, env.actions_left, env.round_number,
                               evs[t], mcts.MCTSConfig(simulations=sims), rng=rng)
        if a is None:
            return t != team
        env.step(a)
    return env.winner == team


def tourney_match(a, b, positions=100, sims=100, pool=None, seed=0,
                  path="matches/tourney_positions.npz", rounds=(3, 8)):
    """Real tournament positions (rounds in `rounds`), each played twice with the models swapped:
    every game starts somewhere else, so a few deterministic lines cannot dominate the score."""
    import numpy as np
    rnd = np.load(path)["round"]
    cand = np.flatnonzero((rnd >= rounds[0]) & (rnd <= rounds[1]))
    pick = np.random.default_rng(seed).choice(cand, positions, replace=False)
    tasks = []
    for k, i in enumerate(pick):
        tasks.append((a, b, path, int(i), seed + k, sims))
        tasks.append((b, a, path, int(i), seed + k, sims))
    res = pool.map(tourney_game, tasks, chunksize=1)
    a_mover = sum(res[0::2])
    a_other = sum(1 - r for r in res[1::2])
    # positions the models split are the informative ones; the rest the position itself decided
    a_both = sum(1 for g1, g2 in zip(res[0::2], res[1::2]) if g1 and not g2)
    b_both = sum(1 for g1, g2 in zip(res[0::2], res[1::2]) if g2 and not g1)
    return {"a_wins": a_mover + a_other, "games": len(res), "a_as_mover": f"{a_mover}/{positions}",
            "a_as_other": f"{a_other}/{positions}", "a_both": a_both, "b_both": b_both}


def match(a, b, games=40, sims=100, workers=22, seed=0, pool=None):
    tasks = []
    for g in range(games):
        if g % 2 == 0:
            tasks.append((a, b, seed + g, sims))
        else:
            tasks.append((b, a, seed + g, sims))
    own = pool is None
    pool = pool or Pool(workers)
    try:
        res = pool.map(arena_game, tasks, chunksize=1)
    finally:
        if own:
            pool.close()
    a_c = sum(1 for g, (w, _) in enumerate(res) if g % 2 == 0 and w == "CYAN")
    a_m = sum(1 for g, (w, _) in enumerate(res) if g % 2 == 1 and w == "MAGENTA")
    n_c = (games + 1) // 2
    n_m = games // 2
    avg_len = sum(l for _, l in res) / len(res)
    return {"a_wins": a_c + a_m, "games": games, "a_as_cyan": f"{a_c}/{n_c}",
            "a_as_magenta": f"{a_m}/{n_m}", "avg_len": round(avg_len, 1)}


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    r = match(a, b, games, sims)
    print(f"{a} vs {b}: {r}")
