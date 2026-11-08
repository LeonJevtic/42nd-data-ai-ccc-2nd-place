"""CCC Colony: One - Level 1 "In a Pinch"

Find two CYAN actions after which MAGENTA cannot capture the CYAN Kernel
with its two actions.

Board input: 8 lines of 8 chars (spaces ignored)
  K / P = CYAN Kernel / Ping      k / p = MAGENTA Kernel / Ping      . = empty
Run: python level1.py  (paste the board, then an empty line / Ctrl+Z Enter)
  or: python level1.py board.txt
"""
import sys

N = 8
KING_DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def inside(r, c):
    return 0 <= r < N and 0 <= c < N


def team(ch):
    if ch == '.':
        return None
    return 'C' if ch.isupper() else 'M'


def actions(board, side):
    """All single actions for `side`: (from, to)."""
    res = []
    for r in range(N):
        for c in range(N):
            ch = board[r][c]
            if team(ch) != side:
                continue
            if ch.upper() == 'K':
                for dr, dc in KING_DIRS:
                    nr, nc = r + dr, c + dc
                    if inside(nr, nc) and team(board[nr][nc]) != side:
                        res.append(((r, c), (nr, nc)))
            else:  # Ping
                for dr, dc in ORTHO:
                    for step in (1, 2):
                        nr, nc = r + dr * step, c + dc * step
                        if not inside(nr, nc) or board[nr][nc] != '.':
                            break
                        res.append(((r, c), (nr, nc)))
                for dr, dc in DIAG:
                    for step in (1, 2):
                        nr, nc = r + dr * step, c + dc * step
                        if not inside(nr, nc):
                            break
                        if board[nr][nc] != '.':
                            if team(board[nr][nc]) not in (None, side):
                                res.append(((r, c), (nr, nc)))
                            break
    return res


def apply(board, act):
    (r, c), (nr, nc) = act
    b = [row[:] for row in board]
    b[nr][nc] = b[r][c]
    b[r][c] = '.'
    return b


def has(board, ch):
    return any(ch in row for row in board)


def kernel_safe(board):
    """True if MAGENTA cannot capture the CYAN Kernel within its two actions."""
    for a1 in actions(board, 'M'):
        b1 = apply(board, a1)
        if not has(b1, 'K'):
            return False
        for a2 in actions(b1, 'M'):
            if not has(apply(b1, a2), 'K'):
                return False
    return True


def solve(board):
    """Returns (score, a1, a2, final_board) of the best CYAN turn, or None."""
    best = None
    for a1 in actions(board, 'C'):
        b1 = apply(board, a1)
        if not has(b1, 'k'):
            # enemy Kernel captured on action 1; any legal second action wins
            a2 = actions(b1, 'C')[0]
            return (99, a1, a2, apply(b1, a2))
        for a2 in actions(b1, 'C'):
            b2 = apply(b1, a2)
            if not has(b2, 'k'):
                return (99, a1, a2, b2)
            if not kernel_safe(b2):
                continue
            # prefer keeping our Ping and capturing the enemy Ping
            score = has(b2, 'P') + (not has(b2, 'p'))
            if best is None or score > best[0]:
                best = (score, a1, a2, b2)
    return best


def name(pos):
    r, c = pos
    return f"{'abcdefgh'[c]}{N - r}"  # chess-style: a-h left->right, 8 at top


def show(board):
    return '\n'.join(''.join(row) for row in board)


def main():
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    rows = [l.replace(' ', '') for l in text.splitlines() if l.strip()]
    board = [list(r) for r in rows[:N]]
    assert len(board) == N and all(len(r) == N for r in board), "need 8x8 board"
    res = solve(board)
    if res is None:
        print("No safe turn found.")
        return
    _, a1, a2, final = res
    for i, (f, t) in enumerate((a1, a2), 1):
        print(f"Action {i}: {board_piece(board, a1, a2, i)} {name(f)} -> {name(t)}"
              f"   (row {f[0]},col {f[1]} -> row {t[0]},col {t[1]})")
    print("\nBoard after your turn:")
    print(show(final))


def board_piece(board, a1, a2, i):
    b = board if i == 1 else apply(board, a1)
    (r, c), _ = a1 if i == 1 else a2
    return {'K': 'Kernel', 'P': 'Ping'}[b[r][c]]


if __name__ == '__main__':
    main()
