"""Record full game logs for gen39/gen38 (CYAN) vs gen34 (MAGENTA)."""
import sys
import json
import os
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "runs", "r1")
GEN34 = f"{MODEL_DIR}/gen34.onnx"
GEN38 = f"{MODEL_DIR}/gen38.onnx"
GEN39 = f"{MODEL_DIR}/gen39.onnx"

SIMS = 100
OUT_DIR = os.path.join(ROOT, "analysis", "out")


def sq(pos):
    return "ABCDEFGH"[pos[1]] + str(8 - pos[0])


def recorded_game(args):
    """Like arena_game but records the full action list and end reason."""
    import numpy as np
    from colony_one import ColonyOneEnv, constants, mcts
    from colony_one.pieces import PieceType

    spec_c, spec_m, seed, sims = args
    rng = np.random.default_rng(seed)

    # lazy import to cache per process
    from az.selfplay import get_eval
    evs = {constants.CYAN: get_eval(spec_c), constants.MAGENTA: get_eval(spec_m)}
    env = ColonyOneEnv()
    cfg = mcts.MCTSConfig(simulations=sims)

    actions_list = []
    end_reason = "unknown"

    while not env.terminated:
        team = env.current_team
        round_num = env.round_number
        actions_left = env.actions_left
        action_num = env.action

        a = mcts.search_action(
            env.board, team, env.actions_left, env.round_number,
            evs[team], cfg, rng=rng
        )
        if a is None:
            winner = constants.MAGENTA if team == constants.CYAN else constants.CYAN
            return {
                "winner": winner, "length": env.action - 1,
                "actions": actions_list, "end_reason": "no_legal_action",
                "seed": seed,
            }

        # Capture detection: piece at to_pos before step
        to_piece = env.board.grid[a.to_pos[0]][a.to_pos[1]]
        # Only-kernels check (needed to distinguish bare_kernel vs firewall)
        only_kernels_before = False
        if actions_left == 1:
            only_kernels_before = env.board.only_kernels_remain()

        actions_list.append({
            "n": action_num + 1,
            "team": team,
            "fr": sq(a.from_pos),
            "to": sq(a.to_pos),
            "round": round_num,
            "al": actions_left,
        })

        env.step(a)

        if env.terminated:
            # Kernel capture: the piece at to_pos was the enemy kernel
            if (to_piece is not None
                    and to_piece.piece_type is PieceType.KERNEL
                    and to_piece.team != team):
                end_reason = "kernel_capture"
            elif actions_left == 1:
                # Last action of the turn: could be bare_kernel or firewall
                if (team == constants.MAGENTA
                        and round_num in constants.SHRINK_ROUNDS
                        and not only_kernels_before):
                    end_reason = f"firewall_r{round_num}"
                else:
                    end_reason = "bare_kernel"
            else:
                # actions_left > 1 means terminated mid-turn = kernel capture
                end_reason = "kernel_capture"

    return {
        "winner": env.winner,
        "length": env.action - 1,
        "actions": actions_list,
        "end_reason": end_reason,
        "seed": seed,
    }


def run_match_cyan(spec_c, spec_m, n_games, base_seed):
    """Run n_games where spec_c is always CYAN.
    Seeds mirror the CYAN half of arena.match: base_seed+0, +2, +4, ..."""
    tasks = [(spec_c, spec_m, base_seed + g * 2, SIMS) for g in range(n_games)]
    with Pool(3) as pool:
        results = pool.map(recorded_game, tasks, chunksize=1)
    return results


if __name__ == "__main__":
    print("=== gen39 (CYAN) vs gen34 (MAGENTA): 50 games, seeds 987654,987656,...987752 ===")
    results_39 = run_match_cyan(GEN39, GEN34, 50, 987654)

    out_39 = os.path.join(OUT_DIR, "gen39_vs_gen34.json")
    with open(out_39, "w") as f:
        json.dump(results_39, f, indent=2)

    wins_39 = sum(1 for r in results_39 if r["winner"] == "CYAN")
    print(f"gen39 CYAN wins: {wins_39}/50  (losses: {50-wins_39}/50)")
    by_reason = {}
    for r in results_39:
        by_reason[r["end_reason"]] = by_reason.get(r["end_reason"], 0) + 1
    print("End reasons:", by_reason)
    print(f"Saved to {out_39}")

    print()
    print("=== gen38 (CYAN) vs gen34 (MAGENTA): 30 games, seeds 987654,987656,...987712 ===")
    results_38 = run_match_cyan(GEN38, GEN34, 30, 987654)

    out_38 = os.path.join(OUT_DIR, "gen38_vs_gen34.json")
    with open(out_38, "w") as f:
        json.dump(results_38, f, indent=2)

    wins_38 = sum(1 for r in results_38 if r["winner"] == "CYAN")
    print(f"gen38 CYAN wins: {wins_38}/30  (losses: {30-wins_38}/30)")
    by_reason_38 = {}
    for r in results_38:
        by_reason_38[r["end_reason"]] = by_reason_38.get(r["end_reason"], 0) + 1
    print("End reasons:", by_reason_38)
    print(f"Saved to {out_38}")
