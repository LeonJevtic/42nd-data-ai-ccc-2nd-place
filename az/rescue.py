"""Rescue-label post-processing: rewrite MCTS policy targets on blundered turns.

For a fraction `rescue_rate` of game turns where a blunder is detected
(the mover lost or the enemy can win immediately after the turn), enumerate
all safe 2-action turns from the start-of-turn snapshot and rewrite the
recorded policy target of the blundered root to point at safe actions.
"""
import time

import numpy as np

from colony_one import constants
from colony_one.encoding import action_index
from az.tactics import wins_this_turn
from az.selfplay import K


def _enemy(team):
    return constants.MAGENTA if team == constants.CYAN else constants.CYAN


def _is_blunder(env_after, mover_team):
    """Return True if the turn just played was a blunder.

    Blunder = mover lost outright, OR the enemy can now win this turn.
    Called after the mover's 2nd action (or early termination).
    """
    env = env_after
    if env.terminated:
        return env.winner != mover_team
    enemy = _enemy(mover_team)
    return wins_this_turn(env.board, enemy, 2, env.round_number)


def _enumerate_safe_turns(snap_env, mover_team):
    """Return list of (a1, a2_or_None, is_safe) for all 2-action turns.

    a2_or_None is None when a1 already terminates the game.
    is_safe: mover wins or enemy cannot win this turn.
    """
    results = []
    enemy = _enemy(mover_team)
    for a1 in snap_env.legal_actions():
        e1 = snap_env.clone()
        e1.step(a1)
        if e1.terminated:
            safe = (e1.winner == mover_team)
            results.append((a1, None, safe))
            continue
        if e1.current_team != mover_team:
            # a1 used both actions somehow (shouldn't happen in normal rules)
            safe = not wins_this_turn(e1.board, enemy, 2, e1.round_number)
            results.append((a1, None, safe))
            continue
        for a2 in e1.legal_actions():
            e2 = e1.clone()
            e2.step(a2)
            if e2.terminated:
                safe = (e2.winner == mover_team)
            else:
                safe = not wins_this_turn(e2.board, enemy, 2, e2.round_number)
            results.append((a1, a2, safe))
    return results


def _rewrite_target(idx_arr, prob_arr, safe_actions, visit_counts_by_idx):
    """Rewrite (in-place) the policy target to cover `safe_actions`.

    safe_actions: list of action objects (each has .from_pos/.to_pos).
    Weights proportional to visit counts when available, else uniform.
    Respects the K slot limit.
    """
    # We work with action indices (already canonical, passed in by caller)
    # safe_indices: list of int action indices
    safe_indices = safe_actions  # caller passes canonical indices directly

    # Determine weights: visit count if the action already appears in idx_arr
    weights = np.array([float(visit_counts_by_idx.get(si, 0)) for si in safe_indices],
                       dtype=np.float32)
    if weights.sum() == 0:
        weights = np.ones(len(safe_indices), dtype=np.float32)
    weights /= weights.sum()

    # Build new idx/prob arrays (keep the K slot limit). The other actions already listed (the
    # root's legal actions) stay in with probability 0: the policy loss normalises over the listed
    # actions only, so a target listing just the safe ones could never push the blunder down.
    safe_set = set(int(si) for si in safe_indices)
    support = [int(si) for si in safe_indices] + [int(i) for i in idx_arr if i >= 0 and int(i) not in safe_set]
    new_idx = np.full(K, -1, dtype=np.int16)
    new_prob = np.zeros(K, dtype=np.float16)
    n = min(len(safe_indices), K)
    m = min(len(support), K)
    new_idx[:m] = support[:m]
    new_prob[:n] = weights[:n]
    # renormalise in case we clipped
    s = float(new_prob[:n].sum())
    if s > 0:
        new_prob[:n] = (new_prob[:n] / s).astype(np.float16)

    idx_arr[:] = new_idx
    prob_arr[:] = new_prob


def apply_rescue(snap_env, mover_team, env_after_turn,
                 idx_list, prob_list, rec_indices,
                 rng, rescue_rate):
    """Possibly rewrite policy targets for a blundered turn.

    Parameters
    ----------
    snap_env      : clone of env at start of mover's turn (actions_left==2)
    mover_team    : team that moved
    env_after_turn: env state right after the 2nd action (or early termination)
    idx_list      : Game.idx list (mutable)
    prob_list     : Game.prob list (mutable)
    rec_indices   : [first_rec_pos, second_rec_pos] indices into idx_list for
                    the two roots of this turn (-1 if not recorded / game ended)
    rng           : numpy Generator
    rescue_rate   : float in [0, 1]

    Returns
    -------
    dict with keys: blunder (bool), rescued (bool), hopeless (bool), t (float seconds)
    """
    t0 = time.perf_counter()
    stats = {"blunder": False, "rescued": False, "hopeless": False, "t": 0.0}

    if rescue_rate <= 0.0:
        return stats
    if not _is_blunder(env_after_turn, mover_team):
        stats["t"] = time.perf_counter() - t0
        return stats

    stats["blunder"] = True
    if rng.random() >= rescue_rate:
        stats["t"] = time.perf_counter() - t0
        return stats

    # Enumerate all 2-action turns
    turn_results = _enumerate_safe_turns(snap_env, mover_team)
    safe_turns = [(a1, a2) for a1, a2, safe in turn_results if safe]

    if not safe_turns:
        stats["hopeless"] = True
        stats["t"] = time.perf_counter() - t0
        return stats

    # Build visit count lookup from the FIRST recorded root
    first_pos = rec_indices[0]
    visit_by_idx_first = {}
    if first_pos >= 0 and first_pos < len(idx_list):
        i_arr = idx_list[first_pos]
        p_arr = prob_list[first_pos]
        for i in range(K):
            if i_arr[i] >= 0:
                # prob is a target already, but we want counts-like weights
                visit_by_idx_first[int(i_arr[i])] = float(p_arr[i])

    # Find the played a1 (the action that was actually played = what the first root selected)
    # We identify it as the first-root canonical index whose prob was highest.
    # But actually we don't know the played action here directly. We rely on
    # the caller to pass it; if not available, fall back to checking safe continuations.
    # The played a1 canonical index is needed.  See gpu_selfplay.py for how we pass it.
    # Since this function is called with played_a1_idx, we check it below.
    # This overload signature does NOT include played_a1_idx; see apply_rescue_with_played below.
    stats["t"] = time.perf_counter() - t0
    return stats


def apply_rescue_full(snap_env, mover_team, env_after_turn,
                      played_a1_idx,
                      idx_list, prob_list, rec_indices,
                      rng, rescue_rate):
    """Full rescue logic with played a1 index known.

    played_a1_idx: canonical action index of the actual first action played,
                   or -1 if unknown.
    rec_indices: [first_rec_pos, second_rec_pos], -1 when not recorded.
    """
    t0 = time.perf_counter()
    stats = {"blunder": False, "rescued": False, "hopeless": False, "t": 0.0, "pos": -1}

    if rescue_rate <= 0.0:
        return stats
    if not _is_blunder(env_after_turn, mover_team):
        stats["t"] = time.perf_counter() - t0
        return stats

    stats["blunder"] = True
    if rng.random() >= rescue_rate:
        stats["t"] = time.perf_counter() - t0
        return stats

    # Enumerate all 2-action turns from snap
    turn_results = _enumerate_safe_turns(snap_env, mover_team)
    safe_turns = [(a1, a2) for a1, a2, safe in turn_results if safe]

    if not safe_turns:
        stats["hopeless"] = True
        stats["t"] = time.perf_counter() - t0
        return stats

    # Safe first-actions: first actions that have at least one safe continuation
    safe_a1_set = set()
    safe_a2_for_played_a1 = []
    for a1, a2, safe in turn_results:
        if not safe:
            continue
        ai1 = action_index(a1, mover_team)
        safe_a1_set.add(ai1)
        if played_a1_idx >= 0 and ai1 == played_a1_idx and a2 is not None:
            safe_a2_for_played_a1.append(action_index(a2, mover_team))

    # Decide: was the played a1 already fine (has safe continuations)?
    played_a1_has_safe = (played_a1_idx >= 0 and played_a1_idx in safe_a1_set)

    if not played_a1_has_safe:
        # Rewrite FIRST root
        pos = rec_indices[0]
        if pos < 0 or pos >= len(idx_list):
            stats["t"] = time.perf_counter() - t0
            return stats
        i_arr = idx_list[pos]
        p_arr = prob_list[pos]
        visit_by_idx = {}
        for i in range(K):
            if i_arr[i] >= 0:
                visit_by_idx[int(i_arr[i])] = float(p_arr[i])
        safe_a1_list = sorted(safe_a1_set)
        _rewrite_target(i_arr, p_arr, safe_a1_list, visit_by_idx)
        stats["rescued"] = True
        stats["pos"] = pos
    else:
        # a1 was fine; rewrite SECOND root to safe a2 given the played a1
        pos = rec_indices[1]
        if pos < 0 or pos >= len(idx_list) or not safe_a2_for_played_a1:
            stats["t"] = time.perf_counter() - t0
            return stats
        i_arr = idx_list[pos]
        p_arr = prob_list[pos]
        visit_by_idx = {}
        for i in range(K):
            if i_arr[i] >= 0:
                visit_by_idx[int(i_arr[i])] = float(p_arr[i])
        _rewrite_target(i_arr, p_arr, sorted(set(safe_a2_for_played_a1)), visit_by_idx)
        stats["rescued"] = True
        stats["pos"] = pos

    stats["t"] = time.perf_counter() - t0
    return stats
