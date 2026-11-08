# Evaluation: how we decided what to upload, and why we lost games

## Comparing two models

[`az/arena.py`](../az/arena.py) `match(a, b, games, sims=100, seed)` plays two models against each other with the
referee's search settings: 100 simulations per action, the first 8 actions sampled from the visit counts, greedy
afterwards. Colours alternate from game to game and every game gets its own seed. The result includes the score per
colour, which turned out to matter (see below).

```python
from multiprocessing import Pool
from az import arena

if __name__ == "__main__":
    with Pool(6) as pool:
        r = arena.match("models/gen34.onnx", "path/to/other.onnx", games=100, sims=100, pool=pool, seed=343434)
        print(r)   # {'a_wins': .., 'games': 100, 'a_as_cyan': '../50', 'a_as_magenta': '../50', 'avg_len': ..}
```

### Noise

A 100-game match has a standard error of about 5 percentage points. During the contest that was the main trap:

- gen23 scored 64% against the live model, then 54% on a second seed. It was not uploaded.
- gen28 scored 63%, 55% and 67% against gen26 on three seeds.

From then on a candidate had to score at least 62% in the watcher **and** hold up in a second 100-game run with a new
seed before we uploaded it. Every upload restarts part of the Glicko-2 placement, so a wrong upload costs rating.

### Start-position matches can mislead

After the 8 sampled opening actions the referee plays greedily, so a match between two fixed networks mostly replays a
handful of lines. We noticed this properly only after the contest:

- extreme colour splits, e.g. gen40 against gen39: 50/100 overall, but only 6 of 50 as Cyan;
- non-transitive results: gen39 beat gen38 88% but was weaker than gen38 against gen34;
- training noise alone (two runs of the same iteration with different seeds) moved a head-to-head result from 53% to 73%.

### Tournament positions

To measure middlegame strength we took positions from rounds 3-6 of our real tournament games and played each one
twice, with the models swapping sides ([`arena.tourney_match`](../az/arena.py)):

- If the same side wins both games, the position decided the game, not the models. That was the case for 72-85% of the
  positions.
- The rest are informative: `a:b` counts the positions that model A (or B) won with both colours.

Calibration: gen44 against gen20 scored 22:6 (100 positions), gen44 against gen34 44:26 (400 positions). After the contest every
5th tournament game was held out of training, and new generations were judged on those held-out positions only
([`az/tourney_positions.py`](../az/tourney_positions.py), [`az/promote.py`](../az/promote.py)).

### Identifying which model played a game

Because the referee is deterministic after action 8, our model's moves can be reproduced exactly. Replaying a downloaded
tournament game with each of our generations and checking which one agrees with all of our moves after action 8 tells us
which upload played it ([`az/replay.py`](../az/replay.py), [`az/attribute.py`](../az/attribute.py)). That let us analyse
losses per model.

## Things that did not help

All measured against the base model over 100 games ([`results/variant_results.txt`](../results/variant_results.txt)):

| Idea | Result |
|---|---|
| Policy logits x1.4 at export (sharper priors) | 39% |
| Policy logits x0.75 at export (flatter priors) | 52% |
| Weight averaging (SWA) of gens 20-23 | 32% |
| SWA of gens 28-30 / 29-31 | 46% / 50% |
| Value output x1.5, clipped, on gen21 | 63% and 58% (it helped gen21, and was live briefly) |
| The same on gen25 | 42% |

The lesson: tweaks on top of a trained network were noise or worse. The gains came from training (learning rate,
data, labels).

## Why we lost games

We downloaded every game file from the contest dashboard, replayed the losses and looked for the turn where the game was
lost. Most of this analysis was done with [`analysis/trap_fw.py`](../analysis/trap_fw.py) (for a given turn: how many of
our possible turns were safe, and what each model would have played) and
[`analysis/cmp_games.py`](../analysis/cmp_games.py) (move by move: what we played, the search's evaluation, and what a
newer model plays).

Our results by colour were even overall (Cyan 68%, Magenta 69% over 131 games). We beat nearly everyone except the teams
ranked 1, 3 and 7 on the final leaderboard. The losses fall into a few patterns:

**A. Giving up in bad positions (most common).** Several turns before the end every move evaluates to about -1. The
100-simulation search then has nothing to distinguish moves by and falls back to the prior, so it does not find the safe
turns that still exist. Examples from our games: 26 of 111 possible turns were safe at one point, 4 of 47 at another,
6 of 93 at a third, yet a losing turn was played. This motivated the value-target cap and the rescue labels after the
contest ([after the contest](after-the-contest.md)).

**B. Kernel on the firewall ring.** Seven of our tournament losses were firewall Kernel deaths: three at the round-8
lock (always on square A2) and four at round 16. In the clearest case only 2 of 719 possible turns were safe (moving the
Kernel inward to a guarded square). All our networks played a capture instead, even with 800 simulations, because the
prior on the escape was 2.7% (gen34). The tournament winner reached exactly this position against us twice, move for
move: a strong opponent that finds a winning line will repeat it.

**C. Bare-Kernel endgames.** With only the two Kernels left, the more central one wins and ties go to Magenta. Our
Kernel was often not centralised early enough.

**D. An opening trap as Magenta.** One of the top teams, playing Cyan, used the same opening against us in 4 of 5 games,
after which our natural reply lost material to an Exploit raid. Some of our generations fell for it and some did not;
gen30, gen33 and gen34 avoided it by deviating at action 11.

**E. Midgame material slips** against the strongest teams: a piece placed where a two-action Ice, Daemon or Exploit
path could reach it.

What the opponents got wrong in our wins against strong teams was mostly the mirror image: over-extended Exploits,
Exploits walking into our Daemon's range, and Kernels that were centralised too early and became targets.
