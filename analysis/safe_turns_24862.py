"""Quick action-29 minimax only, no MCTS loading."""
import sys, time
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from colony_one import constants
from colony_one.pieces import PieceType
from az.replay import parse, replay
from az.tactics import wins_this_turn, _kernel_pos

MATCH_TXT = os.path.join(ROOT, "analysis", "m24862.txt")
CYAN, MAGENTA = constants.CYAN, constants.MAGENTA

def sq(pos): return "ABCDEFGH"[pos[1]] + str(8 - pos[0])
def action_str(a): return f"{sq(a.from_pos)}->{sq(a.to_pos)}"
def on_outer(pos): return min(pos[0], pos[1], 7-pos[0], 7-pos[1]) == 0

text = open(MATCH_TXT, encoding="utf-8").read()
acts, winner = parse(text)

def get_env_at(n):
    for env, action in replay(acts):
        if env.action == n:
            return env.clone()

# Action 29 board state
env29 = get_env_at(29)
kp = _kernel_pos(env29.board, CYAN)
print(f"Action 29: CYAN kernel at {sq(kp)}, on outer ring: {on_outer(kp)}")
print(f"Round: {env29.round_number}, team: {env29.current_team}")

# Print board
print("\nBoard at action 29:")
print(str(env29))

# Full minimax at action 29
t = time.time()
safe, total = [], 0
for a1 in env29.legal_actions():
    e1 = env29.clone(); e1.step(a1)
    if e1.terminated: continue
    for a2 in e1.legal_actions():
        e2 = e1.clone(); e2.step(a2)
        total += 1
        if e2.terminated:
            if e2.winner == CYAN: safe.append((a1, a2, "WON"))
            continue
        if wins_this_turn(e2.board, MAGENTA, 2, e2.round_number): continue
        ok = True
        for b1 in e2.legal_actions():
            e3 = e2.clone(); e3.step(b1)
            if e3.terminated:
                if e3.winner == MAGENTA: ok = False; break
                continue
            for b2 in e3.legal_actions():
                e4 = e3.clone(); e4.step(b2)
                if e4.terminated and e4.winner == MAGENTA: ok = False; break
            if not ok: break
        if ok: safe.append((a1, a2, "SURV"))
print(f"\nAction 29 minimax: {len(safe)}/{total} safe (took {time.time()-t:.1f}s)")
for a1,a2,r in safe:
    print(f"  SAFE: {action_str(a1)} {action_str(a2)} [{r}]")

# Heuristic at 25
for action_num in [13, 17, 21, 25]:
    env_a = get_env_at(action_num)
    kp_a = _kernel_pos(env_a.board, CYAN)
    s, tot = 0, 0
    for a1 in env_a.legal_actions():
        e1 = env_a.clone(); e1.step(a1)
        if e1.terminated: continue
        for a2 in e1.legal_actions():
            e2 = e1.clone(); e2.step(a2)
            tot += 1
            if e2.terminated: continue
            k2 = _kernel_pos(e2.board, CYAN)
            if k2 is None or on_outer(k2): continue
            if wins_this_turn(e2.board, MAGENTA, 2, e2.round_number): continue
            s += 1
    print(f"Action {action_num}: kernel at {sq(kp_a)}, heuristic safe {s}/{tot}")

# Check KE A2->B2 specifically
print("\n--- KE A2->B2 analysis ---")
env29 = get_env_at(29)
ke = next((a for a in env29.legal_actions()
           if env29.board.grid[a.from_pos[0]][a.from_pos[1]] is not None
           and env29.board.grid[a.from_pos[0]][a.from_pos[1]].piece_type is PieceType.KERNEL
           and a.from_pos == (6,0) and a.to_pos == (6,1)), None)
if ke:
    e1 = env29.clone(); e1.step(ke)
    second_safe = 0
    for a2 in e1.legal_actions():
        e2 = e1.clone(); e2.step(a2)
        if e2.terminated: continue
        if wins_this_turn(e2.board, MAGENTA, 2, e2.round_number): continue
        ok = True
        for b1 in e2.legal_actions():
            e3 = e2.clone(); e3.step(b1)
            if e3.terminated:
                if e3.winner == MAGENTA: ok=False; break
                continue
            for b2 in e3.legal_actions():
                e4=e3.clone(); e4.step(b2)
                if e4.terminated and e4.winner==MAGENTA: ok=False; break
            if not ok: break
        if ok:
            second_safe += 1
            print(f"  KE A2->B2 + {action_str(a2)} = SAFE (minimax)")
        else:
            pass
    print(f"  KE A2->B2: {second_safe} safe second actions out of {len(e1.legal_actions())}")
    if second_safe == 0:
        # Find what kills us
        # pick first a2 and trace why
        a2 = e1.legal_actions()[0]
        e2 = e1.clone(); e2.step(a2)
        print(f"  Example: KE A2->B2 + {action_str(a2)}")
        print(f"  MAGENTA wins_immediately: {wins_this_turn(e2.board, MAGENTA, 2, e2.round_number)}")
        kp2 = _kernel_pos(e2.board, CYAN)
        print(f"  CYAN kernel now at {sq(kp2)}, outer={on_outer(kp2)}")
        # find MAGENTA killer
        for b1 in e2.legal_actions()[:5]:
            e3=e2.clone(); e3.step(b1)
            if e3.terminated and e3.winner==MAGENTA:
                print(f"  MAGENTA {action_str(b1)} wins immediately!")
                break
            for b2 in e3.legal_actions()[:5]:
                e4=e3.clone(); e4.step(b2)
                if e4.terminated and e4.winner==MAGENTA:
                    print(f"  MAGENTA {action_str(b1)} {action_str(b2)} wins -> {e4.winner}")
                    break
else:
    print("  KE A2->B2 not legal. Legal kernel moves:")
    for a in env29.legal_actions():
        p = env29.board.grid[a.from_pos[0]][a.from_pos[1]]
        if p and p.piece_type is PieceType.KERNEL:
            print(f"    {action_str(a)}")
