"""Exact tactical labels: can the side to move capture the enemy Kernel this turn?"""
from colony_one import constants
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team


def _kernel_pos(board, team):
    for r in range(constants.BOARD_SIZE):
        for c in range(constants.BOARD_SIZE):
            p = board.grid[r][c]
            if p is not None and p.team == team and p.piece_type is PieceType.KERNEL:
                return (r, c)
    return None


def _hits_unguarded_kernel(board, team, enemy, kp):
    """Some legal action of `team` targets the enemy Kernel at kp and it has no bodyguard."""
    if board._pick_guard(kp, enemy) is not None:
        return False
    return any(a.to_pos == kp for a in legal_actions_for_team(board, team))


def firewall_doom(board, team, round_number):
    """True if `team` wins at the end of its current turn because the firewall burns the
    enemy Kernel: only Magenta's turn ends a shrink round, and the Cyan Kernel cannot move
    during it. (Both Kernels burning is a Magenta win as well.)"""
    if team != constants.MAGENTA or round_number not in constants.SHRINK_ROUNDS:
        return False
    ring = board.next_open_ring()
    kp = _kernel_pos(board, constants.CYAN)
    if ring < 0 or kp is None:
        return False
    r, c = kp
    return min(r, c, 7 - r, 7 - c) == ring


def wins_this_turn(board, team, actions_left, round_number):
    """Exact 'the side to move has won by the end of this turn' label."""
    return firewall_doom(board, team, round_number) or can_kill(board, team, actions_left)


def can_kill(board, team, actions_left):
    """True if `team` can destroy the enemy Kernel within its remaining `actions_left` actions."""
    enemy = constants.MAGENTA if team == constants.CYAN else constants.CYAN
    kp = _kernel_pos(board, enemy)
    if kp is None:
        return False
    if _hits_unguarded_kernel(board, team, enemy, kp):
        return True
    if actions_left < 2:
        return False
    # the first action can remove the last guard (or reposition an attacker)
    for a1 in legal_actions_for_team(board, team):
        rec = board.apply_action(a1)
        try:
            if _hits_unguarded_kernel(board, team, enemy, kp):
                return True
        finally:
            board.undo_action(rec)
    return False
