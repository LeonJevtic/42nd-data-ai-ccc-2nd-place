"""Self-play and arena games, run in worker processes."""
import numpy as np

from colony_one import ColonyOneEnv, constants, mcts
from colony_one.env import IllegalActionError

K = 128  # max legal actions stored per position

_EVALS = {}


class OnnxEval:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, planes):
        p, v = self.s.run(None, {"observation": np.asarray(planes, dtype=np.float32)})
        return p, v


def _uniform(planes):
    b = planes.shape[0]
    return np.zeros((b, 4096), np.float32), np.zeros(b, np.float32)


def get_eval(spec):
    if spec not in _EVALS:
        if spec == "heuristic":
            from az import heuristic
            _EVALS[spec] = heuristic.evaluate
        elif spec == "uniform":
            _EVALS[spec] = _uniform
        else:
            _EVALS[spec] = OnnxEval(spec)
    return _EVALS[spec]


def selfplay_game(args):
    """One self-play game. Returns compact training arrays, or None if aborted.

    With an opponent spec, `spec` plays one colour (alternating by seed) against the
    opponent and only `spec`'s positions are recorded as training targets.
    """
    spec, seed, sims, temp_moves, noise_eps, q_mix = args[:6]
    opp_spec = args[6] if len(args) > 6 else None
    rng = np.random.default_rng(seed)
    ev = get_eval(spec)
    evs = {constants.CYAN: ev, constants.MAGENTA: ev}
    record_team = None
    if opp_spec:
        record_team = constants.CYAN if seed % 2 == 0 else constants.MAGENTA
        other = constants.MAGENTA if record_team == constants.CYAN else constants.CYAN
        evs[other] = get_eval(opp_spec)
    env = ColonyOneEnv()
    # warmup window covers the whole game so root noise stays on throughout
    cfg = mcts.MCTSConfig(simulations=sims, dirichlet_eps=noise_eps,
                          warmup_temperatures=(1.0,) * 64, warmup_step_actions=4)
    obs_l, idx_l, prob_l, team_l, q_l = [], [], [], [], []
    while not env.terminated:
        team = env.current_team
        if not env.legal_actions():
            return None
        obs = env.observation()
        root = mcts.run(env.board, team, env.actions_left, env.round_number, evs[team], cfg, rng=rng)
        if record_team is None or team == record_team:
            idxs, counts = mcts.visit_distribution(root)
            n = len(idxs)
            assert n <= K, n
            counts = np.asarray(counts, dtype=np.float32)
            total = counts.sum()
            idx_arr = np.full(K, -1, np.int16)
            prob_arr = np.zeros(K, np.float16)
            idx_arr[:n] = idxs
            prob_arr[:n] = counts / total
            q = float(sum(root.W.values()) / max(1, total))
            obs_l.append(np.rint(obs * 8).astype(np.uint8))
            idx_l.append(idx_arr)
            prob_l.append(prob_arr)
            team_l.append(team)
            q_l.append(q)
        temp = 1.0 if env.action <= temp_moves else 0.0
        a = mcts.select_action_index(root, temp, rng)
        try:
            env.step(a)
        except IllegalActionError:
            return None
    if not obs_l:
        return None
    z = np.array([1.0 if t == env.winner else -1.0 for t in team_l], np.float32)
    q = np.array(q_l, np.float32)
    return {
        "obs": np.stack(obs_l), "idx": np.stack(idx_l), "prob": np.stack(prob_l),
        "z": z, "q": q, "v": ((1 - q_mix) * z + q_mix * q).astype(np.float32),
        "winner": env.winner, "len": env.action - 1,
        "opp": opp_spec, "won": record_team is not None and env.winner == record_team,
    }


def arena_game(args):
    """One referee-style game: spec_cyan vs spec_magenta. Returns the winner."""
    spec_c, spec_m, seed, sims = args
    rng = np.random.default_rng(seed)
    evs = {constants.CYAN: get_eval(spec_c), constants.MAGENTA: get_eval(spec_m)}
    env = ColonyOneEnv()
    cfg = mcts.MCTSConfig(simulations=sims)
    while not env.terminated:
        team = env.current_team
        a = mcts.search_action(env.board, team, env.actions_left, env.round_number,
                               evs[team], cfg, rng=rng)
        if a is None:
            # no legal action: count as a loss for the side to move
            return constants.MAGENTA if team == constants.CYAN else constants.CYAN, env.action - 1
        env.step(a)
    return env.winner, env.action - 1
