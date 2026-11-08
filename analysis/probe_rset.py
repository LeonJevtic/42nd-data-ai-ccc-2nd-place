"""Prior mass each model puts on the rescue-set targets (legal-masked softmax), per sample and on average."""
import sys
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one import constants, encoding
from colony_one.board import Board
from colony_one.rules import legal_actions_for_team
from az.selfplay import get_eval

d = np.load(os.path.join(ROOT, "matches", "rescue_set.npz"))
obs = d["obs"].astype(np.float32) / 8.0
teams = [constants.MAGENTA if o[15, 0, 0] > 0.5 else constants.CYAN for o in obs]
legal_idx = []
for o, t in zip(obs, teams):
    b = Board.from_observation(o, t)
    legal_idx.append(np.array([encoding.action_index(a, t) for a in legal_actions_for_team(b, t)]))
for model in sys.argv[1:]:
    logits, _ = get_eval(model)(obs)
    mass = []
    for i in range(len(obs)):
        li = logits[i][legal_idx[i]]
        p = np.exp(li - li.max()); p /= p.sum()
        tgt = set(int(x) for x in d["idx"][i] if x >= 0)
        mass.append(sum(pi for pi, a in zip(p, legal_idx[i]) if int(a) in tgt))
    mass = np.array(mass)
    k = int(np.flatnonzero((d["match_id"] == 24862) & (d["action_num"] == 29))[0]) if "match_id" in d.files else -1
    print(f"{model.split('/')[-1]:14s} mean target mass {mass.mean():.3f}  median {np.median(mass):.3f}  >0.5: {(mass > 0.5).sum()}/{len(mass)}"
          + (f"  24862@29: {mass[k]:.3f}" if k >= 0 else ""))
