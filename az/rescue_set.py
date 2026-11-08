"""Build the rescue sample set from tournament training games.

For every training game (skip held-out games exactly like tourney_positions.main),
replay it; for every turn that starts after protocol action 8 of EITHER side,
if the turn was a blunder and a safe turn exists, emit one sample exactly as
apply_rescue_full would rewrite it:

  - played a1 has no safe continuation  ->  obs at turn start,  target = safe first actions
  - played a1 is fine (has a safe a2)   ->  obs after a1,       target = safe second actions

Targets are uniform over the safe actions (no visit counts).
Sparse format: idx int16 K slots -1 pad / prob float16.
obs = np.rint(env.observation() * 8).astype(np.uint8) (same as data_*.npz).

python -m az.rescue_set  ->  matches/rescue_set.npz
"""
import glob
import json
import sys

import numpy as np

from az.replay import parse, replay
from az.regress import TXT_GAMES
from az.rescue import _is_blunder, _enumerate_safe_turns
from az.selfplay import K
from az.tourney_positions import EVAL_EVERY
from colony_one.encoding import action_index

OPENING_ACTIONS = 8   # turns starting before this many actions played are skipped


# ---------------------------------------------------------------------------
# Game iterator (mirrors all_games() but also yields match_id and game index)
# ---------------------------------------------------------------------------

def _games_with_id():
    """Yield (match_id, g, acts, lost) for every downloaded game in the same
    order and with the same g index as tourney_positions.all_games()."""
    seen = set()
    g = 0
    for f in sorted(glob.glob("matches/tournament-matches-*.json")):
        for m in json.load(open(f, encoding="utf-8")):
            if "protocol" in m and m["match_id"] not in seen:
                seen.add(m["match_id"])
                acts, winner = parse(m["protocol"])
                yield m["match_id"], g, acts, winner != m["you"]
                g += 1
    for f in sorted(glob.glob("matches/match-*.txt")):
        mid = int(f.split("-")[-1].split(".")[0])
        if mid in seen:
            continue
        acts, winner = parse(open(f, encoding="utf-8").read())
        fname = f.replace("\\", "/").split("/")[-1]
        ours = TXT_GAMES.get(fname)
        yield mid, g, acts, ours is not None and winner != ours
        g += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(safe_indices, legal_indices):
    """Uniform sparse policy target over safe_indices, K slots, -1 padding. The other legal
    actions are listed with probability 0: the policy loss normalises over the listed actions,
    so leaving them out would give no gradient against the blunder."""
    safe_set = set(safe_indices)
    support = list(safe_indices) + [i for i in legal_indices if i not in safe_set]
    n = min(len(safe_indices), K)
    m = min(len(support), K)
    idx_arr = np.full(K, -1, dtype=np.int16)
    prob_arr = np.zeros(K, dtype=np.float16)
    idx_arr[:m] = support[:m]
    prob_arr[:n] = np.float16(1.0 / n)
    s = float(prob_arr[:n].sum())
    if s > 0:
        prob_arr[:n] = (prob_arr[:n] / s).astype(np.float16)
    return idx_arr, prob_arr


def _process_blunder(snap, mover, env_after, env_mid, start_act, a1_idx,
                     match_id, obs_buf, idx_buf, prob_buf, mid_buf, act_buf):
    """Check one turn for a blunder and emit a sample if rescuable.

    snap      : clone of env at turn start (actions_left==2)
    mover     : team whose turn this is
    env_after : env right after the turn's last action (or after a1 if game ended early)
    env_mid   : env right after a1 (before a2); None when game ended after a1
    start_act : env.action at turn start (number of protocol actions already played)
    a1_idx    : canonical action index of the played first action

    Returns (is_blunder, is_hopeless, is_rescued).
    """
    if not _is_blunder(env_after, mover):
        return False, False, False

    turn_results = _enumerate_safe_turns(snap, mover)
    safe_turns = [(a1t, a2t) for a1t, a2t, s in turn_results if s]
    if not safe_turns:
        return True, True, False

    safe_a1_set = set()
    safe_a2_for_a1 = []
    for a1t, a2t, s in turn_results:
        if not s:
            continue
        ai1 = action_index(a1t, mover)
        safe_a1_set.add(ai1)
        if ai1 == a1_idx and a2t is not None:
            safe_a2_for_a1.append(action_index(a2t, mover))

    if a1_idx not in safe_a1_set:
        # Rewrite first root: observation at turn start, target = safe first actions
        obs = np.rint(snap.observation() * 8).astype(np.uint8)
        t_idx, t_prob = _make_target(sorted(safe_a1_set),
                                     sorted(action_index(a, mover) for a in snap.legal_actions()))
        obs_buf.append(obs)
        idx_buf.append(t_idx)
        prob_buf.append(t_prob)
        mid_buf.append(match_id)
        act_buf.append(start_act)
        return True, False, True
    else:
        # Played a1 was fine; rewrite second root: obs after a1, target = safe a2 given a1
        safe_a2 = sorted(set(safe_a2_for_a1))
        if not safe_a2 or env_mid is None:
            # No safe a2 recorded (game ended after a1 or none found) - skip
            return True, False, False
        obs = np.rint(env_mid.observation() * 8).astype(np.uint8)
        t_idx, t_prob = _make_target(safe_a2, sorted(action_index(a, mover) for a in env_mid.legal_actions()))
        obs_buf.append(obs)
        idx_buf.append(t_idx)
        prob_buf.append(t_prob)
        mid_buf.append(match_id)
        act_buf.append(start_act + 1)
        return True, False, True


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(out_path="matches/rescue_set.npz"):
    """Build rescue_set.npz from all training games.

    Returns the split ('train' or 'eval') that game 24862 is in,
    or None if it is not found.
    """
    obs_buf, idx_buf, prob_buf, mid_buf, act_buf = [], [], [], [], []
    n_games = n_turns = n_blunders = n_rescued = n_hopeless = 0
    game_24862_split = None

    for match_id, g, acts, lost in _games_with_id():
        held_out = g % EVAL_EVERY == EVAL_EVERY - 1
        if match_id == 24862:
            game_24862_split = "eval" if held_out else "train"
        if held_out:
            continue

        n_games += 1
        turn_snap = None       # env clone at turn start (actions_left==2)
        turn_mover = None      # team for this turn
        turn_a1_idx = None     # canonical index of the played a1
        turn_start_act = None  # env.action when turn started

        for env_before, action in replay(acts):
            if action is None:
                # End of game; finalise any pending turn (game ended after a1)
                if (turn_snap is not None
                        and turn_start_act is not None
                        and turn_start_act >= OPENING_ACTIONS):
                    n_turns += 1
                    # env_before is the terminated env (after a1)
                    bl, hp, rs = _process_blunder(
                        turn_snap, turn_mover, env_before, None,
                        turn_start_act, turn_a1_idx, match_id,
                        obs_buf, idx_buf, prob_buf, mid_buf, act_buf)
                    if bl:
                        n_blunders += 1
                    if hp:
                        n_hopeless += 1
                    if rs:
                        n_rescued += 1
                break

            if env_before.actions_left == 2:
                # Start of a new turn (replay() already yields a clone)
                turn_snap = env_before
                turn_mover = env_before.current_team
                turn_a1_idx = action_index(action, turn_mover)
                turn_start_act = env_before.action

            elif env_before.actions_left == 1 and turn_snap is not None:
                # Second action of the current turn
                if turn_start_act >= OPENING_ACTIONS:
                    n_turns += 1
                    env_after_turn = env_before.clone()
                    env_after_turn.step(action)
                    # env_before here is the state after a1 (before a2)
                    bl, hp, rs = _process_blunder(
                        turn_snap, turn_mover, env_after_turn, env_before,
                        turn_start_act, turn_a1_idx, match_id,
                        obs_buf, idx_buf, prob_buf, mid_buf, act_buf)
                    if bl:
                        n_blunders += 1
                    if hp:
                        n_hopeless += 1
                    if rs:
                        n_rescued += 1
                # Reset for next turn
                turn_snap = None
                turn_a1_idx = None

    print(f"Games: {n_games}  Turns checked: {n_turns}  Blunders: {n_blunders}  "
          f"Rescued: {n_rescued}  Hopeless: {n_hopeless}")
    if game_24862_split:
        print(f"Game 24862 is in the '{game_24862_split}' split")
    else:
        print("Game 24862 not found in tournament data")

    if obs_buf:
        np.savez(out_path,
                 obs=np.stack(obs_buf),
                 idx=np.stack(idx_buf),
                 prob=np.stack(prob_buf),
                 match_id=np.array(mid_buf, dtype=np.int32),
                 action_num=np.array(act_buf, dtype=np.int32))
    else:
        # No samples found - save empty arrays so callers don't crash
        dummy_obs = np.zeros((0, 1), dtype=np.uint8)
        np.savez(out_path,
                 obs=dummy_obs,
                 idx=np.zeros((0, K), dtype=np.int16),
                 prob=np.zeros((0, K), dtype=np.float16),
                 match_id=np.zeros(0, dtype=np.int32),
                 action_num=np.zeros(0, dtype=np.int32))

    print(f"{n_rescued} samples  ->  {out_path}")
    return game_24862_split


def main():
    out = "matches/rescue_set.npz"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    build(out)


if __name__ == "__main__":
    main()
