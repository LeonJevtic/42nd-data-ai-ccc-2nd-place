"""Hand-made evaluator used to bootstrap self-play before any network exists.

Works on the canonical (B, 16, 8, 8) planes, so it plugs straight into
colony_one.mcts as an evaluator.
"""
import numpy as np

# plane order: Ping, Daemon, Exploit, Trace, Ice, Kernel
PIECE_W = np.array([1.0, 3.0, 1.7, 2.0, 3.3, 0.0], dtype=np.float32)
CAPTURE_W = np.array([1.0, 3.0, 1.7, 2.0, 3.3, 6.0], dtype=np.float32)

_r, _c = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
RING = np.minimum(np.minimum(_r, _c), np.minimum(7 - _r, 7 - _c)).astype(np.int64)  # 0 outer .. 3 centre


def _adjacent(mask):
    """(B,8,8) bool -> (B,8,8) bool of squares adjacent (8-neighbourhood) to mask."""
    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src = mask[:, max(0, -dr):8 - max(0, dr), max(0, -dc):8 - max(0, dc)]
            out[:, max(0, dr):8 - max(0, -dr), max(0, dc):8 - max(0, -dc)] |= src
    return out


def evaluate(planes):
    planes = np.asarray(planes, dtype=np.float32)
    b = planes.shape[0]
    own = planes[:, 0:6]
    enemy = planes[:, 6:12]
    locked = planes[:, 12]
    urgency = planes[:, 14, 0, 0]

    score = ((own - enemy).sum(axis=(2, 3)) * PIECE_W).sum(axis=1)

    # bodyguards: each adjacent friendly non-kernel piece is an extra life for the kernel
    own_k = own[:, 5] > 0.5
    en_k = enemy[:, 5] > 0.5
    own_rest = own[:, :5].sum(axis=1) > 0.5
    en_rest = enemy[:, :5].sum(axis=1) > 0.5
    own_guards = (_adjacent(own_k) & own_rest).sum(axis=(1, 2))
    en_guards = (_adjacent(en_k) & en_rest).sum(axis=(1, 2))
    score += 0.6 * np.minimum(own_guards, 2) - 0.6 * np.minimum(en_guards, 2)

    # firewall: pieces standing on the next ring to lock are in danger as it approaches
    rings_locked = locked[:, 0, 0] + locked[:, 1, 1] + locked[:, 2, 2] + locked[:, 3, 3]
    nxt = rings_locked.astype(np.int64)
    on_next = (RING[None] == nxt[:, None, None])
    danger = np.clip((urgency - 0.5) * 2.0, 0.0, 1.0)  # 0 until halfway, 1 on the shrink round
    own_mat_next = ((own * on_next[:, None]).sum(axis=(2, 3)) * PIECE_W).sum(axis=1)
    en_mat_next = ((enemy * on_next[:, None]).sum(axis=(2, 3)) * PIECE_W).sum(axis=1)
    score -= danger * 0.5 * (own_mat_next - en_mat_next)
    own_k_next = (own[:, 5] * on_next).sum(axis=(1, 2))
    en_k_next = (enemy[:, 5] * on_next).sum(axis=(1, 2))
    score -= (1.0 + 4.0 * danger) * (own_k_next - en_k_next)

    # kernel centralisation matters more as the board shrinks
    own_k_ring = (own[:, 5] * RING[None]).sum(axis=(1, 2))
    en_k_ring = (enemy[:, 5] * RING[None]).sum(axis=(1, 2))
    score += (0.15 + 0.15 * rings_locked) * (own_k_ring - en_k_ring)

    value = np.tanh(0.3 * score).astype(np.float32)

    # policy prior: prefer actions aimed at valuable enemy pieces (captures / shots)
    target_val = (enemy * CAPTURE_W[None, :, None, None]).sum(axis=1).reshape(b, 64)
    logits = np.broadcast_to(target_val[:, None, :], (b, 64, 64)).reshape(b, 4096)
    return np.ascontiguousarray(logits, dtype=np.float32), value
