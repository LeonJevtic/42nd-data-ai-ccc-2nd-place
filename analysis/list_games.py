import json, os, sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from az.replay import parse
from az.attribute import end_kind
old = set()
import glob
for f in glob.glob(os.path.join(ROOT, "matches", "tournament-matches-*.json")):
    if os.path.abspath(f) == os.path.abspath(sys.argv[1]): continue
    for m in json.load(open(f, encoding="utf-8")): old.add(m["match_id"])
d = json.load(open(sys.argv[1], encoding="utf-8"))
for m in d:
    if "protocol" not in m:
        print(m["match_id"], "no protocol", m.get("opponent")); continue
    acts, winner = parse(m["protocol"])
    print(m["match_id"], m["played_at"][11:19], m["you"], "W" if winner == m["you"] else "L", end_kind(m["protocol"]), len(acts), "NEW" if m["match_id"] not in old else "old", m["opponent"], sep="\t")
