import sys
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one import mcts, encoding
from az.replay import parse, replay
from az.selfplay import K, get_eval
from az.rescue import apply_rescue_full

acts, _ = parse(open(sys.argv[1], encoding="utf-8").read())
envs = [(e.clone(), a) for e, a in replay(acts)]
snap, a1 = next((e, a) for e, a in envs if e.action == 29)
team = snap.current_team
ev = get_eval(os.path.join(ROOT, "runs", "r1", "gen85.onnx"))
root = mcts.run(snap.board, team, snap.actions_left, snap.round_number, ev, mcts.MCTSConfig(simulations=100), rng=np.random.default_rng(0)) if hasattr(mcts, "run") else None
idxs, counts = mcts.visit_distribution(root)
idx_arr = np.full(K, -1, np.int16); prob_arr = np.zeros(K, np.float16)
idx_arr[:len(idxs)] = idxs; prob_arr[:len(idxs)] = np.array(counts, np.float32) / sum(counts)
legal = sorted(encoding.action_index(a, team) for a in snap.legal_actions())
print("legal", len(legal), "root entries", len(idxs), "root covers legal:", set(legal) == set(int(i) for i in idxs))
after = snap.clone(); after.step(a1)
a2 = next(a for e, a in envs if e.action == 30); after.step(a2)
before = idx_arr.copy()
st = apply_rescue_full(snap, team, after, encoding.action_index(a1, team), [idx_arr], [prob_arr], [0, -1], np.random.default_rng(1), 1.0)
listed = set(int(i) for i in idx_arr if i >= 0)
pos = {int(i) for i, p in zip(idx_arr, prob_arr) if i >= 0 and p > 0}
sq = lambda i: "ABCDEFGH"[(i // 64) % 8] + str(8 - (i // 64) // 8) + "->" + "ABCDEFGH"[(i % 64) % 8] + str(8 - (i % 64) // 8)
print("rescue stats", st)
print("all original entries kept:", set(int(i) for i in before if i >= 0) <= listed, "| listed", len(listed))
print("positive-mass actions:", sorted(pos), "prob sum", float(prob_arr.astype(np.float32).sum()))
assert set(int(i) for i in before if i >= 0) <= listed and len(pos) == 2 and abs(float(prob_arr.astype(np.float32).sum()) - 1) < 1e-2
print("OK")
