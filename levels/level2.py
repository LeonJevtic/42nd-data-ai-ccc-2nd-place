"""CCC Colony: One - Level 2 "New Friends"

Suggests the best CYAN turn (2 actions) for the current position, using
alpha-beta search over whole turns with the full Level 2 rules
(bodyguards, firewall, kernel endgame).

Input (file or stdin):
  first line: the current round number (1..26), e.g.  7
  then 8 lines of 8 chars:
    K / P = CYAN Kernel / Ping    k / p = MAGENTA Kernel / Ping
    . = empty   (tiles already removed by the firewall may be written as . or #)
Optional argument: search depth in turns (default 3 = you, MAGENTA, you).
Run: python level2.py board.txt [depth]
"""
import sys
import time

N = 8
KING_DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
FIREWALL_ROUNDS = (8, 16, 24, 26)
WIN = 100000


def ring(r, c):
    """0 = outer edge ... 3 = central 2x2."""
    return min(r, c, N - 1 - r, N - 1 - c)


def removed_rings(rnd):
    """Number of rings already gone at the start of round `rnd`."""
    return sum(1 for t in FIREWALL_ROUNDS if t < rnd)


def team(ch):
    if ch in '.#':
        return None
    return 'C' if ch.isupper() else 'M'


def enemy(side):
    return 'M' if side == 'C' else 'C'


def find(board, ch):
    for r in range(N):
        for c in range(N):
            if board[r][c] == ch:
                return (r, c)
    return None


def actions(board, side, gone):
    """All single actions for `side`: (from, to). `gone` = removed rings."""
    def ok(r, c):
        return 0 <= r < N and 0 <= c < N and ring(r, c) >= gone

    res = []
    for r in range(N):
        for c in range(N):
            ch = board[r][c]
            if team(ch) != side:
                continue
            if ch.upper() == 'K':
                for dr, dc in KING_DIRS:
                    nr, nc = r + dr, c + dc
                    if ok(nr, nc) and team(board[nr][nc]) != side:
                        res.append(((r, c), (nr, nc)))
            else:
                for dr, dc in ORTHO:
                    for step in (1, 2):
                        nr, nc = r + dr * step, c + dc * step
                        if not ok(nr, nc) or board[nr][nc] != '.':
                            break
                        res.append(((r, c), (nr, nc)))
                for dr, dc in DIAG:
                    for step in (1, 2):
                        nr, nc = r + dr * step, c + dc * step
                        if not ok(nr, nc):
                            break
                        if board[nr][nc] != '.':
                            if team(board[nr][nc]) == enemy(side):
                                res.append(((r, c), (nr, nc)))
                            break
    return res


def apply(board, act):
    """Apply one action. Returns (new_board, winner_or_None)."""
    (r, c), (nr, nc) = act
    b = [row[:] for row in board]
    target = b[nr][nc]
    if target.upper() == 'K':
        # attacked Kernel: a bodyguard dies instead, attacker stays put
        side = team(target)
        for dr, dc in KING_DIRS:
            gr, gc = nr + dr, nc + dc
            if 0 <= gr < N and 0 <= gc < N and team(b[gr][gc]) == side:
                b[gr][gc] = '.'
                return b, None
        b[nr][nc] = b[r][c]
        b[r][c] = '.'
        return b, team(b[nr][nc])
    b[nr][nc] = b[r][c]
    b[r][c] = '.'
    return b, None


def end_of_turn(board, side, rnd):
    """Firewall + kernel endgame after `side` finished its turn in round `rnd`.
    Returns (board, winner_or_None)."""
    if side == 'M' and rnd in FIREWALL_ROUNDS:
        dying = FIREWALL_ROUNDS.index(rnd)
        board = [row[:] for row in board]
        for r in range(N):
            for c in range(N):
                if ring(r, c) == dying:
                    board[r][c] = '.'
        ck, mk = find(board, 'K'), find(board, 'k')
        if not mk and not ck:
            return board, 'M'
        if not ck:
            return board, 'M'
        if not mk:
            return board, 'C'
    pieces = [(r, c) for r in range(N) for c in range(N) if board[r][c] != '.']
    if len(pieces) == 2:
        ck, mk = find(board, 'K'), find(board, 'k')
        if max(abs(ck[0] - mk[0]), abs(ck[1] - mk[1])) <= 2:
            return board, enemy(side)
        return board, 'C' if ring(*ck) > ring(*mk) else 'M'
    if side == 'M' and rnd >= 26:
        return board, 'M'
    return board, None


def turns(board, side, rnd):
    """All distinct results of a full turn: list of (acts, board, winner)."""
    gone = removed_rings(rnd)
    seen = {}
    for a1 in actions(board, side, gone):
        b1, w = apply(board, a1)
        if w:
            seen.setdefault('win', ((a1,), b1, w))
            continue
        a2s = actions(b1, side, gone)
        if not a2s:
            b, w = end_of_turn(b1, side, rnd)
            key = ''.join(map(''.join, b))
            seen.setdefault(key, ((a1,), b, w))
        for a2 in a2s:
            b2, w = apply(b1, a2)
            if not w:
                b2, w = end_of_turn(b2, side, rnd)
            key = 'win' if w == side else ''.join(map(''.join, b2))
            seen.setdefault(key, ((a1, a2), b2, w))
    return list(seen.values())


def evaluate(board, rnd):
    """Heuristic score from CYAN's point of view."""
    s = 0
    ck, mk = find(board, 'K'), find(board, 'k')
    for r in range(N):
        for c in range(N):
            ch = board[r][c]
            if ch == 'P':
                s += 300 + 10 * ring(r, c)
                if max(abs(r - ck[0]), abs(c - ck[1])) == 1:
                    s += 40  # bodyguard
            elif ch == 'p':
                s -= 300 + 10 * ring(r, c)
                if max(abs(r - mk[0]), abs(c - mk[1])) == 1:
                    s -= 40
    # centrality matters more as the firewall closes in
    w = 15 + 10 * removed_rings(rnd)
    s += w * (ring(*ck) - ring(*mk))
    # pressure: being close to the enemy kernel with our pieces
    s -= 3 * max(abs(ck[0] - mk[0]), abs(ck[1] - mk[1]))
    return s


def search(board, side, rnd, depth, alpha, beta, deadline):
    if depth == 0 or time.time() > deadline:
        return evaluate(board, rnd), None
    options = turns(board, side, rnd)
    if not options:
        return evaluate(board, rnd), None
    nxt_side = enemy(side)
    nxt_rnd = rnd + 1 if side == 'M' else rnd

    def quick(o):
        _, b, w = o
        if w:
            return WIN if w == 'C' else -WIN
        return evaluate(b, nxt_rnd)
    options.sort(key=quick, reverse=(side == 'C'))

    best = None
    for o in options:
        acts, b, w = o
        if w:
            val = (WIN + depth) if w == 'C' else -(WIN + depth)
        else:
            val, _ = search(b, nxt_side, nxt_rnd, depth - 1, alpha, beta, deadline)
        if side == 'C':
            if best is None or val > best[0]:
                best = (val, acts)
            alpha = max(alpha, val)
        else:
            if best is None or val < best[0]:
                best = (val, acts)
            beta = min(beta, val)
        if alpha >= beta:
            break
    return best


def name(pos):
    r, c = pos
    return f"{'ABCDEFGH'[c]}{N - r}"


def main():
    args = [a for a in sys.argv[1:]]
    text = open(args[0]).read() if args else sys.stdin.read()
    depth = int(args[1]) if len(args) > 1 else 3
    lines = [l.replace(' ', '') for l in text.splitlines() if l.strip()]
    rnd = int(lines[0])
    board = [list(l.replace('#', '.')) for l in lines[1:N + 1]]
    assert len(board) == N and all(len(r) == N for r in board), "need 8x8 board"

    start = time.time()
    val, acts = search(board, 'C', rnd, depth, -10 * WIN, 10 * WIN, start + 60)
    b = board
    for i, a in enumerate(acts, 1):
        piece = {'K': 'Kernel', 'P': 'Ping'}[b[a[0][0]][a[0][1]]]
        print(f"Action {i}: {piece} {name(a[0])} -> {name(a[1])}")
        b, _ = apply(b, a)
    verdict = 'WIN found' if val >= WIN else 'LOSS expected' if val <= -WIN else f'eval {val}'
    print(f"({verdict}, {time.time() - start:.1f}s)")
    print('\n'.join(''.join(r) for r in b))


if __name__ == '__main__':
    main()
