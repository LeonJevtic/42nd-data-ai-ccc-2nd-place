"""Prior mass a model puts on the rescue first actions (KE A2->A3, KE A2->B3) at 24862 action 29."""
import sys
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one import encoding
from az.replay import parse, replay
from az.selfplay import get_eval

acts, _ = parse(open(sys.argv[1], encoding="utf-8").read())
env = next(e for e, a in replay(acts) if e.action == 29)
team = env.current_team
legal = env.legal_actions()
sq = lambda p: "ABCDEFGH"[p[1]] + str(8 - p[0])
for model in sys.argv[2:]:
    ev = get_eval(model)
    logits, v = ev(env.observation()[None].astype(np.float32))
    li = np.array([logits[0][encoding.action_index(a, team)] for a in legal])
    p = np.exp(li - li.max()); p /= p.sum()
    names = [f"{sq(a.from_pos)}->{sq(a.to_pos)}" for a in legal]
    resc = sum(pi for pi, n in zip(p, names) if n in ("A2->A3", "A2->B3"))
    top = sorted(zip(p, names), reverse=True)[:3]
    print(f"{model.split('/')[-1]:22s} value {float(v[0]):+.3f}  rescue prior {resc:.4f}  top {[(n, round(float(x), 3)) for x, n in top]}")
