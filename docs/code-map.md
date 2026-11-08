# Code map

All modules import the game engine `colony_one` from the organizer's starter kit, which is not part of this repository
(see the main [README](../README.md#setup)). Run everything from the repository root, e.g. `python -m az.train ...`.

## `az/`: the pipeline

| File | What it does | How to run |
|---|---|---|
| [`train.py`](../az/train.py) | Main loop: self-play → replay buffer → training → ONNX export → quick check, forever. All options are command-line flags (see `--help`). Requires a CUDA GPU. | `python -m az.train --run runs/r1 ...` |
| [`net.py`](../az/net.py) | The network (ResNet trunk, bilinear policy head, value, kill and material heads), ONNX export, `clip_value` / `clip_value_file`. | library |
| [`gpu_selfplay.py`](../az/gpu_selfplay.py) | GPU-batched self-play: many concurrent games per process, one batched network call per step, random late-game, tournament and exploration starts, rescue hooks. | used by `train.py` |
| [`selfplay.py`](../az/selfplay.py) | Older CPU self-play (one game per process) plus the ONNX evaluator and the game function the arena uses. | library |
| [`heuristic.py`](../az/heuristic.py) | Hand-written evaluator that bootstraps iteration 0. | library |
| [`tactics.py`](../az/tactics.py) | Exact "can the side to move win this turn" labels: `can_kill`, `firewall_doom`, `wins_this_turn`. | library |
| [`synth.py`](../az/synth.py) | Builds synthetic positions with exact kill labels, focused on firewall deaths. | `python -m az.synth 120000` → `matches/synth_kill.npz` |
| [`arena.py`](../az/arena.py) | Model-vs-model matches under referee settings: `match`, `tourney_match` (tournament positions, both colours), `late_match`. | library |
| [`watch.py`](../az/watch.py) | Watcher: benchmarks every new generation against a reference model and logs candidates (≥ 62%). | `python -m az.watch runs/r1 models/gen34.onnx --every 1 --games 100 --threshold 0.62 --workers 6` |
| [`promote.py`](../az/promote.py) | Post-contest auto-promoter: held-out tournament positions + start-position floor + pool of older generations + validator; keeps `models/model_level4_best.onnx` and `BEST.txt` up to date. | `python -m az.promote runs/r1 <reference.onnx> ...` |
| [`submit.py`](../az/submit.py) | Copies a generation to `submit/model.onnx` and runs the organizer's validator. | `python -m az.submit runs/r1/genN.onnx` |
| [`tourney_positions.py`](../az/tourney_positions.py) | Extracts positions from downloaded tournament games into a training file and a held-out evaluation file (every 5th game). | `python -m az.tourney_positions` |
| [`rescue.py`](../az/rescue.py) | Post-contest: detects one-turn blunders in self-play, enumerates all safe two-action turns and rewrites the policy target. | used by `gpu_selfplay.py` |
| [`rescue_set.py`](../az/rescue_set.py) | Post-contest: builds a fixed set of blunder positions with corrected targets from tournament games. | `python -m az.rescue_set` |
| [`regress.py`](../az/regress.py) | Regression check: avoidable blunders from our lost tournament games; does a model now avoid them? | `python -m az.regress <model.onnx> ...` |
| [`replay.py`](../az/replay.py) | Parses a game protocol and finds which of our models played it (100% move agreement after action 8). | `python -m az.replay matches/match-N.txt genA.onnx genB.onnx` |
| [`attribute.py`](../az/attribute.py) | The same for the dashboard's JSON game exports, plus win/loss per model and how each game ended. | `python -m az.attribute <games.json> -- <models...>` |
| [`ensemble.py`](../az/ensemble.py) | Averages several ONNX models into one file (tried, no gain). | `python -m az.ensemble out.onnx a.onnx b.onnx` |

## `analysis/`: one-off scripts

These were written during the contest to answer specific questions, so several have model names or game numbers built in.
Paths are relative to the repository root. Scripts that read tournament games need your own downloaded game files in
`matches/`.

| Script | Question it answers |
|---|---|
| [`trap.py`](../analysis/trap.py), [`trap_fw.py`](../analysis/trap_fw.py) | At a given turn of a game: how many of our turns were safe, what does each model play, and does that allow an immediate loss? (`trap_fw` also counts firewall deaths.) |
| [`cmp_games.py`](../analysis/cmp_games.py) | For every turn of a lost game: what we played, the search's evaluation, and what a newer model plays. |
| [`check_labels.py`](../analysis/check_labels.py) | Are the kill labels in `az/tactics.py` correct? Compares them with brute force on random positions. |
| [`variants.py`](../analysis/variants.py) | Exports variants of a checkpoint: policy temperature, value scaling, weight averaging (SWA). |
| [`gauntlet.py`](../analysis/gauntlet.py) | Plays candidates against a pool of older generations. |
| [`tourney_detail.py`](../analysis/tourney_detail.py) | Per-position outcomes of a tournament-position match. |
| [`list_games.py`](../analysis/list_games.py) | First look at a new game export: result, colour, how it ended, length, new or already seen. |
| [`firewall_scan.py`](../analysis/firewall_scan.py) | Lists all firewall Kernel deaths in the downloaded games. |
| [`safe_turns_24862.py`](../analysis/safe_turns_24862.py) | Lists the safe turns at the firewall position from game 24862 (included as [`m24862.txt`](../analysis/m24862.txt)). |
| [`probe29.py`](../analysis/probe29.py) | Prior each model puts on the two escape moves in that position. |
| [`probe_rset.py`](../analysis/probe_rset.py) | Average prior each model puts on the corrected targets of the rescue set. |
| [`record_games.py`](../analysis/record_games.py), [`analyze_games.py`](../analysis/analyze_games.py) | Record games between two generations with full move lists and cluster the losses by line. |
| [`ab_train.py`](../analysis/ab_train.py) | Re-train one iteration from a checkpoint with a different learning rate (A/B test). |
| [`bn_recal.py`](../analysis/bn_recal.py) | Recompute BatchNorm statistics of a checkpoint. |
| [`smoke_restart.py`](../analysis/smoke_restart.py), [`test_rewrite2.py`](../analysis/test_rewrite2.py) | Tests run before restarting the trainer with new code. |

## Files the code expects but this repository does not contain

| Path | What it is | How to get it |
|---|---|---|
| `colony_one/`, `tools/`, `play/` | game engine, referee search, validator, match script | organizer's Level 4 starter kit |
| `runs/r1/` | checkpoints, self-play data and every generation (22 GB for the full run) | created by `az.train` |
| `matches/*.json`, `matches/match-*.txt` | tournament games downloaded from the contest dashboard | your own tournament games |
| `matches/tourney_positions.npz`, `matches/tourney_eval.npz` | positions extracted from those games | `python -m az.tourney_positions` |
| `matches/synth_kill.npz` | synthetic kill-labelled positions | `python -m az.synth 120000` |
| `matches/rescue_set.npz` | rescue set | `python -m az.rescue_set` |
