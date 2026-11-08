# After the contest: from gen34 to gen78

The contest ended a few minutes after we uploaded gen34. We kept the pipeline running for about four more hours
(18:54 to 23:20) to see how far it would go and to fix the weaknesses we had found in our tournament games. The result is
[`models/gen78.onnx`](../models/gen78.onnx).

> **gen78 never played a rated game.** Everything on this page is measured locally, with the referee's search settings.
> The tournament result (2nd place) belongs to gen34.

## Progress

| Step | Evidence |
|---|---|
| gen34 → gen40 | gen40 beats gen34 89% and 84% on two seeds (86.5% over 200 games) |
| gen40 → gen43 → gen44 | gen43 beats gen40 67% (72 + 62); gen44 beats gen43 76% (78 + 74); gen44 beats gen34 84% |
| gen44 → gen50 | 43:17 on 400 real tournament positions |
| gen50 → gen69 → gen71 | gen69: 77% vs gen50, 38:20 on positions; gen71: 70% vs gen69, 38:21 on positions |
| **gen71 → gen78** | **88/100 from the start position (49/50 as Cyan, 39/50 as Magenta), 33:15 on held-out tournament positions** |
| gen79 to gen85 | none of them beats gen78 on held-out positions by the required two standard errors (closest: gen84 at 30:23) |

"a:b on positions" means: positions taken from our real tournament games (rounds 3-6) are played twice, once with each
model on each side; *a* and *b* count the positions that one model won with both colours. The other positions are
decided by the position itself and carry no information. See [evaluation](evaluation.md#tournament-positions).

## What changed

### 1. Value target capped at 0.95, more search in the value target (iteration 35)

Our most common loss pattern was a network that sees every move as lost a few turns before the end, and then stops
fighting. Capping the value target at ±0.95 keeps a gradient in clearly won or lost positions, so the network can
still rank them by how long they survive. At the same time the value target took more of the search result
(`--q-mix` 0.4 → 0.55). The next generations improved quickly: gen34 → gen40 → gen43 → gen44.

### 2. Rescue labels (iteration 46)

In self-play, every turn that lets the opponent win immediately (a Kernel capture or the firewall burning our Kernel) is
checked exactly against **all** two-action alternatives. If a safe turn existed, the recorded policy target of that
turn is rewritten to the safe moves ([`az/rescue.py`](../az/rescue.py)). A fixed **rescue set** of 112 such positions from
our real tournament games is mixed into every batch ([`az/rescue_set.py`](../az/rescue_set.py)).

### 3. A better way to pick the best model (from gen50)

Start-position matches between our own generations turned out to be misleading. After the 8 sampled opening actions
the referee plays greedily, so a match between two fixed networks mostly replays a handful of lines. We saw colour splits
like 48/50 against 10/50 and non-transitive results. The automatic promoter ([`az/promote.py`](../az/promote.py)) therefore
judges each new generation mainly on **positions from held-out tournament games**, with three guard rails: at least 45%
from the start position, a pool of 8 older generations, and the validator.

### 4. More varied self-play (iteration 79 and 85)

25% of self-play games start from real tournament positions (was 10%), 30% from random endgames (was 45%), 15% start
from a real or initial position plus 1-6 random moves, and the first 12 instead of 8 moves are sampled.

## Bugs we found and fixed

- **Validator rejection.** gen50 was rejected because `k + (1 - k) * v` can exceed 1.0 by a float rounding error. Every
  export now ends in a `Clip(-1, 1)` node ([`az/net.py`](../az/net.py), `clip_value`).
- **Rescue labels had no effect for 40 iterations.** The rewritten targets listed only the safe moves. Our policy loss
  normalises over the listed moves only, so the target could never push the probability of the blunder down. Fixed by
  listing all legal moves, with probability 0 for the unsafe ones.
- **Rescue set overweighted.** The rescue-set loss was computed in a separate small forward pass and added with the weight
  of the whole policy loss, about 51 times too much per sample. gen86 and gen87 collapsed (15:39 and 10:55 against gen78
  on positions). Fixed, and the weights were rolled back to gen85.

## What did not help

- Learning rate 5e-5 instead of 1e-4 (A/B test from the same checkpoint: 44% against the control).
- An ensemble of two sibling networks in one ONNX file (47% and 44% against one of them).
- Recalibrating the BatchNorm statistics of gen78.
- Before that, during the contest: export temperature x1.4 or x0.75, weight averaging over 3-4 generations, scaling the
  value output.

## Where gen78 still fails

The clearest example is a game that the tournament winner beat us in twice, move for move: our Kernel sits on the
outer ring, the round-8 firewall is coming, and only 2 of 719 possible turns survive (moving the Kernel inward to a
guarded square). Even with 800 simulations our networks play a capture instead, because the prior on the escape is
tiny: 2.7% for gen34 and 7.6% for gen78. The rescue labels raise it slowly; whether they would have fixed it with more
training is untested.

## Ideas we did not get to

- Exponential moving average (EMA) of the weights for export.
- A wider network: 96 channels still fits the 5 MiB limit, at about twice the iteration time.
- More simulations in self-play.
- "Forced loss" labels and a Kernel-centralisation target for bare-Kernel endgames.
