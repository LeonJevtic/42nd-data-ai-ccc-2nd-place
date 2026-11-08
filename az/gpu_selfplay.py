"""Self-play with many concurrent games per process and batched GPU inference.

The search is colony_one.mcts.run turned into a generator: it yields the planes it
needs evaluated and receives (logits, values), so one process can interleave dozens
of games and send all their leaves to the GPU in a single call. Search behaviour
(PUCT, virtual loss, batch of 8 leaves per game, root noise) is the referee's.
"""
import os

import numpy as np

from colony_one import ColonyOneEnv, constants, encoding, mcts
from colony_one.board import Board
from colony_one.env import IllegalActionError
from colony_one.rules import legal_actions_for_team

from az.selfplay import K, get_eval
from az.tactics import can_kill, wins_this_turn
from colony_one.encoding import action_index as _action_index

_NET = {}

LATE_ROUNDS = np.arange(5, 27)


def _place_late(env, rng, rnd, team):
    from colony_one.board import Board
    from colony_one.pieces import Piece, PieceType
    locked = Board.locked_cells_for_round(rnd)
    b = env.board
    for r in range(8):
        for c in range(8):
            b.grid[r][c] = None
            b.locked[r][c] = locked[r][c]
    free = {(r, c) for r in range(8) for c in range(8) if not locked[r][c]}
    density = rng.uniform(0.1, 0.7)
    kernels = {}
    for t in (constants.CYAN, constants.MAGENTA):   # kernels first so both always fit
        sq = sorted(free)[int(rng.integers(len(free)))]
        free.discard(sq)
        b.grid[sq[0]][sq[1]] = Piece(PieceType.KERNEL, t)
        kernels[t] = sq
    for t in (constants.CYAN, constants.MAGENTA):
        for pt in PieceType:
            if pt is PieceType.KERNEL or not free or rng.random() >= density:
                continue
            kr, kc = kernels[t]
            near = sorted(sq for sq in free if max(abs(sq[0] - kr), abs(sq[1] - kc)) == 1)
            pool = near if near and rng.random() < 0.5 else sorted(free)   # often a bodyguard
            sq = pool[int(rng.integers(len(pool)))]
            free.discard(sq)
            b.grid[sq[0]][sq[1]] = Piece(pt, t)


def random_late_env(rng):
    """A random mid/late-game position: round 5-26 with its locked rings, both Kernels and a
    random (often small) set of other pieces, so endgames and firewall shrinks get trained.
    Positions where the side to move can destroy the enemy Kernel at once are rejected."""
    from az.tactics import can_kill
    env = ColonyOneEnv()
    rnd = int(rng.choice(LATE_ROUNDS))
    team = constants.CYAN if rng.random() < 0.5 else constants.MAGENTA
    for _ in range(30):
        _place_late(env, rng, rnd, team)
        if not can_kill(env.board, team, 2):
            break
    env.round_number = rnd
    env.current_team = team
    env.actions_left = 2
    env.action = mcts.move_number(team, 2, rnd)
    return env


_MAT_W = {"PING": 1.0, "DAEMON": 3.0, "EXPLOIT": 1.7, "TRACE": 2.0, "ICE": 3.3, "KERNEL": 0.0}


def _material(board):
    """Cyan minus Magenta material (Kernels excluded)."""
    s = 0.0
    for row in board.grid:
        for p in row:
            if p is not None:
                s += _MAT_W[p.piece_type.value] * (1.0 if p.team == constants.CYAN else -1.0)
    return s


_TOURNEY = {}


def tourney_env(rng, path):
    """Start from a random position of a real tournament game (see az/tourney_positions.py)."""
    if path not in _TOURNEY:
        d = np.load(path)
        _TOURNEY[path] = {k: d[k] for k in d.files}
    d = _TOURNEY[path]
    i = int(rng.integers(len(d["team"])))
    team = constants.CYAN if d["team"][i] == 0 else constants.MAGENTA
    env = ColonyOneEnv()
    env.board = Board.from_observation(d["obs"][i].astype(np.float32) / 8.0, team)
    env.current_team = team
    env.actions_left = int(d["actions_left"][i])
    env.round_number = int(d["round"][i])
    env.action = mcts.move_number(team, env.actions_left, env.round_number)
    return env


def run_gen(board, team, actions_left, round_number, cfg, rng, aux=None, aux_rate=0.0):
    """Generator twin of mcts.run: yields (B,16,8,8) planes, receives (logits, values).

    With `aux`, a random share of the evaluated leaves is labelled exactly with can_kill
    and appended as (uint8 planes, label): the positions the referee's search really asks about.
    """
    root = mcts.Node(team, actions_left, round_number)
    in_warmup = mcts.move_number(team, actions_left, round_number) <= cfg.warmup_actions
    dirichlet = ((cfg.dirichlet_alpha, cfg.dirichlet_eps)
                 if cfg.add_root_noise and in_warmup else None)
    legal = legal_actions_for_team(board, team)
    planes = encoding.encode_board(board, team, actions_left, round_number)
    logits, values = yield planes[None]
    mcts._set_priors(root, legal, logits[0], float(values[0]), dirichlet, rng)

    batch_size = max(1, cfg.batch_size)
    remaining = cfg.simulations
    while remaining > 0:
        pending = []
        done = 0
        for _ in range(min(batch_size, remaining)):
            path = []
            value, value_sign, leaf = mcts._descend(root, board, +1, cfg.c_puct, cfg.virtual_loss, path)
            if leaf is not None:
                mcts._apply_virtual_loss(leaf.path)
                pending.append(leaf)
                done += 1
            elif value is not None:
                mcts._backup(path, value, value_sign, release_virtual_loss=False)
                done += 1
            else:
                break
        if pending:
            if aux is not None:
                for p in pending:
                    if rng.random() < aux_rate:
                        lb = Board.from_observation(p.planes, p.node.team)
                        aux.append((np.rint(p.planes * 8).astype(np.uint8),
                                    wins_this_turn(lb, p.node.team, p.node.actions_left,
                                                   p.node.round_number)))
            logits_b, values_b = yield np.stack([p.planes for p in pending])
            for leaf, lg, v in zip(pending, logits_b, values_b):
                mcts._set_priors(leaf.node, leaf.legal, lg, float(v))
                mcts._backup(leaf.path, float(v), leaf.sign, release_virtual_loss=True)
        if done == 0:
            break
        remaining -= done
    return root


def _make_explore_env(rng, s):
    """Walk a random number of legal actions from a base position for explore-start games.

    Base is a tourney position (p=0.7 when tourney_file is configured, else initial env).
    Walk length is uniform in [explore_min, explore_max].  If the walk reaches a terminal
    state, the walk is restarted from a fresh base (up to 10 attempts); on persistent
    failure the last generated base position is returned un-walked.
    """
    explore_min = s.get("explore_min", 1)
    explore_max = s.get("explore_max", 6)
    tourney_file = s.get("tourney_file")
    base_env = ColonyOneEnv()
    for _ in range(10):
        if tourney_file and rng.random() < 0.7:
            base_env = tourney_env(rng, tourney_file)
        else:
            base_env = ColonyOneEnv()
        walk = base_env.clone()
        n = int(rng.integers(explore_min, explore_max + 1))
        ok = True
        try:   # an engine error in a random walk must not kill the self-play worker: just retry
            for _ in range(n):
                if walk.terminated:
                    ok = False
                    break
                legal = walk.legal_actions()
                if not legal:
                    ok = False
                    break
                a = legal[int(rng.integers(len(legal)))]
                walk.step(a)
        except Exception:
            ok = False
        if ok and not walk.terminated:
            return walk
    return base_env


class Game:
    """One self-play game driven step by step from outside."""

    def __init__(self, seed, s, opp_spec):
        self.rng = np.random.default_rng(seed)
        self.s = s
        roll = self.rng.random()
        late_frac, tourney_frac = s.get("late_frac", 0.0), s.get("tourney_frac", 0.0)
        explore_frac = s.get("explore_frac", 0.0)
        # explore: separate roll drawn ONLY when explore_frac > 0, so the RNG stream is
        # unchanged (bit-identical) when explore_frac is absent or 0.
        if explore_frac > 0.0 and opp_spec is None:
            self.explore = self.rng.random() < explore_frac
        else:
            self.explore = False
        if self.explore:
            self.late = False
            self.tourney = False
            self.env = _make_explore_env(self.rng, s)
        else:
            self.late = opp_spec is None and roll < late_frac
            self.tourney = (opp_spec is None and not self.late and s.get("tourney_file")
                            and roll < late_frac + tourney_frac)
            if self.late:
                self.env = random_late_env(self.rng)
            elif self.tourney:
                self.env = tourney_env(self.rng, s["tourney_file"])
            else:
                self.env = ColonyOneEnv()
        self.cfg = mcts.MCTSConfig(simulations=s["sims"], dirichlet_eps=s["noise_eps"],
                                   warmup_temperatures=(1.0,) * 64, warmup_step_actions=4)
        self.opp_spec = opp_spec
        self.record_team = None
        self.keys = {constants.CYAN: "net", constants.MAGENTA: "net"}
        if opp_spec:
            self.record_team = constants.CYAN if seed % 2 == 0 else constants.MAGENTA
            other = constants.MAGENTA if self.record_team == constants.CYAN else constants.CYAN
            self.keys[other] = opp_spec
        self.obs, self.idx, self.prob, self.team, self.q, self.kill = [], [], [], [], [], []
        self._rsc_flags = []   # per-recorded-root int8: 1 when rescue rewrote that root's target
        self.aux = []
        self.mat_hist, self.rec_step = [], []   # Cyan-perspective material per action, recorded steps
        self.aborted = False
        self.gen = None
        # rescue-label state (inert when rescue_rate == 0)
        self._rescue_rate = s.get("rescue_rate", 0.0)
        self._turn_snap = None       # env clone at start of current turn
        self._turn_rec = [-1, -1]    # recorded indices for [first, second] action of current turn
        self._turn_mover = None      # team that owns the current turn
        self._turn_action_count = 0  # actions taken in this turn so far
        self._played_a1_idx = -1     # canonical index of the first action played this turn
        self._rsc_blunders = 0
        self._rsc_rescued = 0
        self._rsc_hopeless = 0
        self._rsc_time = 0.0
        self._rsc_errors = 0

    @property
    def key(self):
        return self.keys[self.env.current_team]

    def start(self):
        """Begin the search for the next action; returns planes to evaluate, or None if over."""
        env = self.env
        while not env.terminated:
            if not env.legal_actions():
                self.aborted = True
                return None
            self.gen = run_gen(env.board, env.current_team, env.actions_left, env.round_number,
                               self.cfg, self.rng, aux=self.aux, aux_rate=self.s.get("aux_rate", 0.0))
            return next(self.gen)
        return None

    def feed(self, logits, values):
        """Deliver evaluations; returns the next planes to evaluate, or None when the game ended."""
        try:
            return self.gen.send((logits, values))
        except StopIteration as stop:
            root = stop.value
        self._act(root)
        if self.aborted:
            return None
        return self.start()

    def _act(self, root):
        env = self.env
        team = env.current_team
        al = env.actions_left          # actions_left BEFORE the step
        recording = self.record_team is None or team == self.record_team

        # --- rescue: start of turn (actions_left == 2) ---
        if self._rescue_rate > 0.0 and al == 2 and recording:
            self._turn_snap = env.clone()
            self._turn_mover = team
            self._turn_rec = [-1, -1]
            self._played_a1_idx = -1

        rec_pos = len(self.idx) if recording else -1

        if recording:
            idxs, counts = mcts.visit_distribution(root)
            n = len(idxs)
            assert n <= K, n
            counts = np.asarray(counts, dtype=np.float32)
            total = counts.sum()
            idx_arr = np.full(K, -1, np.int16)
            prob_arr = np.zeros(K, np.float16)
            idx_arr[:n] = idxs
            # the referee samples the first 8 actions from visit counts: a sharper opening
            # prior concentrates those visits and makes the forced randomness less harmful
            target = counts ** self.s.get("open_sharpen", 1.0) if env.action <= 8 else counts
            prob_arr[:n] = target / target.sum()
            self.obs.append(np.rint(env.observation() * 8).astype(np.uint8))
            self.idx.append(idx_arr)
            self.prob.append(prob_arr)
            self.team.append(team)
            self.q.append(float(sum(root.W.values()) / max(1, total)))
            self.kill.append(wins_this_turn(env.board, team, env.actions_left, env.round_number))
            self._rsc_flags.append(0)
            self.rec_step.append(len(self.mat_hist))

        # rescue: record which rec-list slot belongs to action 1 vs 2 of this turn
        if self._rescue_rate > 0.0 and team == self._turn_mover:
            slot = 0 if al == 2 else 1
            self._turn_rec[slot] = rec_pos

        self.mat_hist.append(_material(env.board))
        temp = 1.0 if env.action <= self.s["temp_moves"] else 0.0
        a = mcts.select_action_index(root, temp, self.rng)

        # rescue: the integer returned by select_action_index IS the canonical action index
        if self._rescue_rate > 0.0 and al == 2 and team == self._turn_mover:
            self._played_a1_idx = int(a)

        try:
            env.step(a)
        except IllegalActionError:
            self.aborted = True
            return

        # rescue: check for blunder when the mover's turn just ended
        # (played action 2, i.e., al was 1 before the step) or game ended during this turn
        if self._rescue_rate > 0.0 and self._turn_mover is not None:
            turn_ended = env.terminated or (al == 1 and team == self._turn_mover)
            if turn_ended:
                self._run_rescue(env)

    def _run_rescue(self, env_after):
        """Blunder detection and policy-target rewrite for the current turn."""
        from az.rescue import apply_rescue_full
        try:
            st = apply_rescue_full(
                snap_env=self._turn_snap,
                mover_team=self._turn_mover,
                env_after_turn=env_after,
                played_a1_idx=self._played_a1_idx,
                idx_list=self.idx,
                prob_list=self.prob,
                rec_indices=self._turn_rec,
                rng=self.rng,
                rescue_rate=self._rescue_rate,
            )
        except Exception:   # a labelling bug must never kill the self-play worker
            st = {"blunder": False, "rescued": False, "hopeless": False, "t": 0.0, "pos": -1}
            self._rsc_errors += 1
        self._rsc_blunders += int(st["blunder"])
        self._rsc_rescued += int(st["rescued"])
        self._rsc_hopeless += int(st["hopeless"])
        self._rsc_time += st["t"]
        pos = st.get("pos", -1)
        if pos >= 0 and pos < len(self._rsc_flags):
            self._rsc_flags[pos] = 1
        self._turn_mover = None   # reset so we don't re-trigger

    def _mat_targets(self, horizon=8):
        """Material balance `horizon` actions later, from each recorded mover's view, scaled to ~[-1, 1]."""
        hist = self.mat_hist + [_material(self.env.board)]
        out = []
        for step, team in zip(self.rec_step, self.team):
            future = hist[min(step + horizon, len(hist) - 1)]
            sign = 1.0 if team == constants.CYAN else -1.0
            out.append(np.clip(sign * future / 6.0, -1.0, 1.0))
        return np.array(out, np.float32)

    def result(self):
        if self.aborted or not self.obs:
            return None
        env = self.env
        z = np.array([1.0 if t == env.winner else -1.0 for t in self.team], np.float32)
        q = np.array(self.q, np.float32)
        qm = self.s["q_mix"]
        return {
            "obs": np.stack(self.obs), "idx": np.stack(self.idx), "prob": np.stack(self.prob),
            "z": z, "q": q, "v": ((1 - qm) * z + qm * q).astype(np.float32),
            "kill": np.array(self.kill, np.int8),
            "rsc": np.array(self._rsc_flags, np.int8),
            "mat": self._mat_targets(),
            "aux_obs": (np.stack([o for o, _ in self.aux]) if self.aux
                        else np.zeros((0, 16, 8, 8), np.uint8)),
            "aux_kill": np.array([k for _, k in self.aux], np.int8),
            "winner": env.winner, "len": env.action - 1, "late": self.late or bool(self.tourney),
            "explore": bool(self.explore),
            "opp": self.opp_spec, "won": self.record_team is not None and env.winner == self.record_team,
            "rsc_blunders": self._rsc_blunders, "rsc_rescued": self._rsc_rescued,
            "rsc_hopeless": self._rsc_hopeless, "rsc_time": self._rsc_time,
            "rsc_errors": self._rsc_errors,
        }


class GraphNet:
    """fp16 network replayed through CUDA graphs, one graph per padded batch size.

    The network is small, so a plain forward pass is dominated by kernel launches;
    replaying a captured graph halves the GPU time per call.
    """

    BUCKETS = (16, 32, 64, 128, 256)

    def __init__(self, net, use_graphs=True):
        import torch
        self.torch = torch
        self.net = net.half().to(memory_format=torch.channels_last).eval()
        self.graphs = {}
        self.use_graphs = use_graphs
        # one memory pool for all bucket graphs: outputs are copied off right after each replay
        self.pool = torch.cuda.graph_pool_handle() if use_graphs else None

    def _graph(self, size):
        torch = self.torch
        if size not in self.graphs:
            x = torch.zeros(size, 16, 8, 8, device="cuda", dtype=torch.half).to(
                memory_format=torch.channels_last)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self.net(x)
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, pool=self.pool):
                p, v = self.net(x)
            self.graphs[size] = (g, x, p, v)
        return self.graphs[size]

    def __call__(self, planes):
        torch = self.torch
        n = planes.shape[0]
        if not self.use_graphs:
            with torch.no_grad():
                x = torch.from_numpy(planes).cuda(non_blocking=True).half().to(
                    memory_format=torch.channels_last)
                p, v = self.net(x)
                return p.float().cpu().numpy(), v.float().cpu().numpy()
        top = self.BUCKETS[-1]
        if n > top:
            parts = [self(planes[i:i + top]) for i in range(0, n, top)]
            return np.concatenate([p for p, _ in parts]), np.concatenate([v for _, v in parts])
        size = next(b for b in self.BUCKETS if b >= n)
        g, x, p, v = self._graph(size)
        with torch.no_grad():
            src = torch.from_numpy(planes).cuda(non_blocking=True).half()
            x[:n].copy_(src)
            if n < size:
                x[n:].zero_()
            g.replay()
            return p[:n].float().cpu().numpy(), v[:n].float().cpu().numpy()


def _get_net(path, use_graphs=True):
    import torch
    from az.net import Net
    key = (path, os.path.getmtime(path))
    if key not in _NET:
        while len(_NET) >= 4:
            _NET.pop(next(iter(_NET)))
            torch.cuda.empty_cache()
        ck = torch.load(path, map_location="cuda", weights_only=False)
        net = Net(**ck["cfg"]).cuda().eval()
        net.load_state_dict(ck["net"], strict=False)   # older nets have no kill head
        _NET[key] = GraphNet(net, use_graphs)
    return _NET[key]


def _net_eval(net, planes):
    return net(planes)


def play_batch(args):
    """Play len(jobs) games, `concurrent` at a time. jobs = [(seed, opp_spec_or_None), ...]."""
    ckpt, jobs, concurrent, s = args
    net = _get_net(ckpt)
    queue = list(jobs)
    active = {}   # game -> planes awaiting evaluation
    results = []

    def launch():
        while queue and len(active) < concurrent:
            seed, opp = queue.pop(0)
            g = Game(seed, s, opp)
            planes = g.start()
            if planes is None:
                results.append(g.result())
            else:
                active[g] = planes

    launch()
    while active:
        groups = {}
        for g in active:
            groups.setdefault(g.key, []).append(g)
        evaluated = {}
        for key, games in groups.items():
            stacked = np.concatenate([active[g] for g in games]).astype(np.float32, copy=False)
            if key == "net":
                logits, values = _net_eval(net, stacked)
            elif key.endswith(".pt"):
                # older generation from the league, also on the GPU (no graphs: saves memory)
                logits, values = _net_eval(_get_net(key, use_graphs=False), stacked)
            else:
                logits, values = get_eval(key)(stacked)
            o = 0
            for g in games:
                n = active[g].shape[0]
                evaluated[g] = (logits[o:o + n], values[o:o + n])
                o += n
        for g, (lg, v) in evaluated.items():
            nxt = g.feed(lg, v)
            if nxt is None:
                del active[g]
                results.append(g.result())
            else:
                active[g] = nxt
        launch()
    return [r for r in results if r is not None]
