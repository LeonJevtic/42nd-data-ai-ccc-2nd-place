"""Task 1: Scan all games for firewall kernel kills."""
import json
import glob
import re
import sys

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)

MATCHES_DIR = os.path.join(ROOT, "matches")

# Pattern: FIREWALL ROUND N: [TEAMxKE (SQ)]
FW_PATTERN = re.compile(r"FIREWALL ROUND (\d+).*?\[(\w+)xKE \(([A-H][1-8])\)\]")

results = []

# Scan txt match files
for path in sorted(glob.glob(MATCHES_DIR + "/match-*.txt")):
    mid = int(re.search(r"match-(\d+)", path).group(1))
    text = open(path, encoding="utf-8").read()
    m = FW_PATTERN.search(text)
    if m:
        round_n = int(m.group(1))
        team = m.group(2)
        sq = m.group(3)
        # Determine winner
        winner = "CYAN" if "CYAN WINS" in text else ("MAGENTA" if "MAGENTA WINS" in text else "?")
        # Source is txt, no opponent info
        results.append({
            "source": "txt",
            "match_id": mid,
            "fw_team": team,
            "fw_square": sq,
            "fw_round": round_n,
            "winner": winner,
            "opponent": "unknown",
            "our_colour": "unknown",
            "our_loss": None,
        })

# Scan JSON tournament files
for jpath in sorted(glob.glob(MATCHES_DIR + "/tournament-matches-*.json")):
    data = json.load(open(jpath, encoding="utf-8"))
    for m in data:
        proto = m.get("protocol", "")
        fw = FW_PATTERN.search(proto)
        if fw:
            round_n = int(fw.group(1))
            team = fw.group(2)
            sq = fw.group(3)
            winner = "CYAN" if "CYAN WINS" in proto else ("MAGENTA" if "MAGENTA WINS" in proto else "?")
            our_colour = m.get("you", "?")
            our_loss = (team == our_colour)
            results.append({
                "source": "json",
                "match_id": m.get("match_id"),
                "fw_team": team,
                "fw_square": sq,
                "fw_round": round_n,
                "winner": winner,
                "opponent": m.get("opponent", "?"),
                "our_colour": our_colour,
                "our_loss": our_loss,
            })

print("=== FIREWALL KERNEL KILLS ===")
for r in sorted(results, key=lambda x: x["match_id"]):
    flag = ""
    if r["our_loss"] is True:
        flag = " <-- OUR KERNEL KILLED"
    elif r["our_loss"] is False:
        flag = " (opponent's kernel killed)"
    print(f"  match {r['match_id']:6d} | round {r['fw_round']:2d} | {r['fw_team']} KE at {r['fw_square']} | winner: {r['winner']} | opp: {r['opponent'][:30]}{flag}")

print(f"\nTotal firewall kills: {len(results)}")
our_losses = [r for r in results if r["our_loss"] is True]
print(f"  Our kernel killed: {len(our_losses)}")
opp_losses = [r for r in results if r["our_loss"] is False]
print(f"  Opponent kernel killed: {len(opp_losses)}")
