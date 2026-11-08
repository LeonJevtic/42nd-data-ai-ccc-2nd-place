"""Analyze game logs: classify CYAN losses, cluster by post-action-8 sequence,
find the critical mistake, and test gen34 on the same positions as CYAN."""
import sys
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "analysis", "out")
MODEL_DIR = os.path.join(ROOT, "runs", "r1")
GEN34 = f"{MODEL_DIR}/gen34.onnx"
GEN38 = f"{MODEL_DIR}/gen38.onnx"
GEN39 = f"{MODEL_DIR}/gen39.onnx"


def load_games(path):
    with open(path) as f:
        return json.load(f)


def cyan_loss(game):
    return game["winner"] == "MAGENTA"


def actions_after_8(game):
    """Return the list of (team, fr, to) tuples for actions > 8."""
    return [(a["team"], a["fr"], a["to"]) for a in game["actions"] if a["n"] > 8]


def cyan_actions_after_8(game):
    """CYAN's actions only (post-warmup), as a tuple for clustering."""
    return tuple((a["fr"], a["to"]) for a in game["actions"] if a["n"] > 8 and a["team"] == "CYAN")


def cluster_losses(games, label):
    """Break down losses by end reason and cluster by CYAN post-warmup sequence."""
    losses = [g for g in games if cyan_loss(g)]
    wins = [g for g in games if not cyan_loss(g)]

    print(f"\n=== {label} ===")
    print(f"Total: {len(games)}, CYAN wins: {len(wins)}, CYAN losses: {len(losses)}")

    # End reason breakdown for losses only
    loss_reasons = Counter(g["end_reason"] for g in losses)
    print(f"CYAN loss end reasons: {dict(loss_reasons)}")

    # Game lengths
    loss_lengths = [g["length"] for g in losses]
    if loss_lengths:
        print(f"Loss lengths: min={min(loss_lengths)}, max={max(loss_lengths)}, avg={sum(loss_lengths)/len(loss_lengths):.1f}")

    # Cluster by first N CYAN moves after action 8
    # Use first 4 CYAN moves (8 actions = 4 CYAN moves) after warmup
    cluster_key = lambda g: tuple((a["fr"], a["to"]) for a in g["actions"] if a["n"] > 8 and a["team"] == "CYAN")[:4]
    clusters = defaultdict(list)
    for g in losses:
        clusters[cluster_key(g)].append(g)

    print(f"\nPost-warmup CYAN clusters in losses (by first 4 CYAN moves):")
    for key, cluster_games in sorted(clusters.items(), key=lambda x: -len(x[1])):
        reasons = Counter(g["end_reason"] for g in cluster_games)
        print(f"  {len(cluster_games)}x  {key}  reasons={dict(reasons)}")

    return losses, wins, clusters


def find_critical_turns(losses, label):
    """Find turns where CYAN was in danger - look at moves before the loss."""
    print(f"\n=== Critical turn analysis for {label} losses ===")

    # For each loss, find the turn before the end
    # A kernel capture means the enemy had an opportunity the turn before
    # A firewall means CYAN's kernel was on the outer ring at the critical round

    firewall_losses = [g for g in losses if g["end_reason"].startswith("firewall")]
    kernel_cap_losses = [g for g in losses if g["end_reason"] == "kernel_capture"]
    bare_kernel_losses = [g for g in losses if g["end_reason"] == "bare_kernel"]

    print(f"Firewall losses: {len(firewall_losses)}")
    print(f"Kernel capture losses: {len(kernel_cap_losses)}")
    print(f"Bare kernel losses: {len(bare_kernel_losses)}")

    # For firewall losses: show CYAN's last actions before the firewall
    for g in firewall_losses[:5]:
        round_str = g["end_reason"]  # e.g., "firewall_r8"
        cyan_acts = [(a["n"], a["fr"], a["to"], a["round"]) for a in g["actions"] if a["team"] == "CYAN"]
        print(f"  seed={g['seed']} {round_str}: CYAN last 6 actions: {cyan_acts[-6:]}")

    # For kernel capture losses: show the last few moves
    for g in kernel_cap_losses[:5]:
        acts = [(a["n"], a["team"][:1], a["fr"], a["to"], a["round"]) for a in g["actions"]]
        print(f"  seed={g['seed']} kernel_cap len={g['length']}: last 6 acts: {acts[-6:]}")

    return firewall_losses, kernel_cap_losses


def reconstruct_critical_position(game, action_num):
    """Replay the game up to action_num and return the env state."""
    from colony_one import ColonyOneEnv
    from colony_one.rules import legal_actions_for_team

    env = ColonyOneEnv()
    for act_data in game["actions"]:
        if act_data["n"] > action_num:
            break
        team = act_data["team"]
        fr_str = act_data["fr"]
        to_str = act_data["to"]
        fr_pos = (8 - int(fr_str[1]), ord(fr_str[0]) - ord("A"))
        to_pos = (8 - int(to_str[1]), ord(to_str[0]) - ord("A"))
        legal = [a for a in legal_actions_for_team(env.board, team)
                 if a.from_pos == fr_pos and a.to_pos == to_pos]
        if not legal:
            print(f"  WARNING: action {act_data['n']} not legal, stopping")
            break
        env.step(legal[0])
    return env


def test_gen34_as_cyan(losses, label, n_test=10):
    """Replay critical positions and let gen34 play as CYAN, checking if it's safer."""
    import numpy as np
    from colony_one import ColonyOneEnv, constants, mcts
    from colony_one.rules import legal_actions_for_team
    from colony_one.pieces import PieceType
    from az.selfplay import get_eval
    from az.tactics import wins_this_turn

    print(f"\n=== Testing gen34 as CYAN in critical positions ({label}) ===")

    # Focus on kernel capture losses since those are most common
    test_losses = [g for g in losses if g["end_reason"] == "kernel_capture"][:n_test]
    if not test_losses:
        test_losses = losses[:n_test]

    gen34_ev = get_eval(GEN34)
    rng = np.random.default_rng(42)

    gen34_safe_count = 0
    gen34_total = 0

    for g in test_losses:
        # Find the action number where CYAN last moved before the loss
        # The loss ends at len(actions); the last CYAN action is before the final MAGENTA move
        acts = g["actions"]
        # Find the last turn of CYAN before game ended
        cyan_acts = [a for a in acts if a["team"] == "CYAN"]
        if not cyan_acts:
            continue

        # The "critical position" is after CYAN's last complete turn (before the killing turn)
        # Find what action number CYAN's last turn starts at
        # The game end: if MAGENTA captured CYAN kernel, the critical position is
        # just before MAGENTA's final turn, i.e., after CYAN's last turn

        # Find action number of CYAN's last turn (second action of CYAN's last turn)
        last_cyan_act = cyan_acts[-1]
        critical_action = last_cyan_act["n"]  # action number just after CYAN's last move

        env = reconstruct_critical_position(g, critical_action)
        if env.terminated:
            continue

        # Check: can enemy win right now? (after CYAN's last move)
        enemy = constants.MAGENTA
        can_enemy_win = wins_this_turn(env.board, enemy, env.actions_left if env.current_team == enemy else 2, env.round_number)

        # Now let gen34 play as CYAN from the PREVIOUS critical position
        # Go back to just before CYAN's last turn
        if len(cyan_acts) >= 2:
            # Second-to-last CYAN action
            prev_action_num = cyan_acts[-3]["n"] if len(cyan_acts) >= 3 else cyan_acts[-2]["n"]
        else:
            prev_action_num = 0

        # Reconstruct position just before CYAN's last turn
        if len(cyan_acts) >= 2:
            # Last complete cyan turn starts at cyan_acts[-2]
            start_action = (cyan_acts[-2]["n"] - 1) if len(cyan_acts) >= 2 else 0
        else:
            start_action = 0

        # Find the position after CYAN's second-to-last turn (before CYAN's fatal turn)
        # This is: after all actions up to but not including CYAN's last turn
        # CYAN's last turn = cyan_acts[-2] and cyan_acts[-1] (2 actions per turn)
        if len(cyan_acts) >= 2:
            # CYAN's last turn is acts[-2] and acts[-1]
            # Position BEFORE CYAN's last turn = after acts[-3] (if exists)
            before_cyan_last = cyan_acts[-2]["n"] - 1  # action number before cyan's last turn
        else:
            before_cyan_last = 0

        env_before = reconstruct_critical_position(g, before_cyan_last)
        if env_before.terminated or env_before.current_team != constants.CYAN:
            continue

        # Now let gen34 play from this position greedily (2 actions)
        env_test = env_before.clone() if hasattr(env_before, 'clone') else reconstruct_critical_position(g, before_cyan_last)

        gen34_moves = []
        for _ in range(2):
            if env_test.terminated:
                break
            if env_test.current_team != constants.CYAN:
                break
            a = mcts.search_action(env_test.board, constants.CYAN, env_test.actions_left,
                                   env_test.round_number, gen34_ev, mcts.MCTSConfig(simulations=100),
                                   temperature=0.0, rng=rng)
            if a is None:
                break
            gen34_moves.append(f"{sq(a.from_pos)}->{sq(a.to_pos)}")
            env_test.step(a)

        if env_test.terminated:
            gen34_total += 1
            gen34_safe_count += 0  # already lost
            print(f"  seed={g['seed']}: gen34 immediately lost in replay")
            continue

        # Check if enemy can kill after gen34's moves
        gen34_safe = not wins_this_turn(env_test.board, enemy, 2, env_test.round_number)
        gen34_total += 1
        gen34_safe_count += int(gen34_safe)

        print(f"  seed={g['seed']} before_action={before_cyan_last}: gen34 plays {gen34_moves} -> safe={gen34_safe}")

    print(f"\ngen34 as CYAN: safe in {gen34_safe_count}/{gen34_total} critical positions")
    return gen34_safe_count, gen34_total


def sq(pos):
    return "ABCDEFGH"[pos[1]] + str(8 - pos[0])


if __name__ == "__main__":
    g39 = load_games(os.path.join(OUT_DIR, "gen39_vs_gen34.json"))
    g38 = load_games(os.path.join(OUT_DIR, "gen38_vs_gen34.json"))

    losses_39, wins_39, clusters_39 = cluster_losses(g39, "gen39 (CYAN) vs gen34 (MAGENTA)")
    losses_38, wins_38, clusters_38 = cluster_losses(g38, "gen38 (CYAN) vs gen34 (MAGENTA)")

    fw_39, kc_39 = find_critical_turns(losses_39, "gen39")
    fw_38, kc_38 = find_critical_turns(losses_38, "gen38")

    # Test gen34 in the critical positions (gen39 losses)
    safe34, total34 = test_gen34_as_cyan(losses_39, "gen39 losses")
