# Models

| File | What it is | Uploaded to the tournament? |
|---|---|---|
| [`gen34.onnx`](gen34.onnx) | **Our final submission.** Live on the contest server from about 18:46 until the end; this is the model that finished **2nd of 382**. | Yes (last upload) |
| [`gen78.onnx`](gen78.onnx) | Best model from training that continued after the contest ended. Much stronger than gen34 in our local tests, but it never played a rated game. | No |

Both files pass the organizer's validator (`tools/validate_submission.py` from the starter kit prints `ACCEPTED`).

## Interface

Both models follow the tournament contract:

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| input `observation` | `(batch, 16, 8, 8)` | float32 | the position from the side to move (16 feature planes, see [docs/how-it-works.md](../docs/how-it-works.md#input)) |
| output `policy` | `(batch, 4096)` | float32 | raw logits over all `from_square * 64 + to_square` actions; illegal actions are masked by the search |
| output `value` | `(batch,)` | float32 | expected result for the side to move, in `[-1, 1]` |

## Details

| | gen34 | gen78 |
|---|---|---|
| Network | 64 channels, 6 residual blocks, policy + value + kill heads | same |
| Stored values (all ONNX initializers, BatchNorm folded into the convolutions) | 502,372 | 502,374 (the 2 extra are the bounds of the `Clip` node) |
| ONNX opset | 17 | 17 |
| File size | 2.0 MB (limit 5 MiB) | 2.0 MB |
| Value output | `k + (1 - k) * v` | same, followed by `Clip(-1, 1)` |
| Training iteration | 33 (exported at the end of iteration 33, ~18:42) | 77 (post-contest) |
| SHA-256 | `71d5c8d73c21741088ff945c2214f8666f4e7beb9ddca754372ffcef90dd8def` | `06964f9ecf627f60bb0de7eb1522ee95cb276bdfb817fb3bdc848119ef69df8e` |

Why gen78 has a `Clip` node: a later generation (gen50) was rejected by the validator because `k + (1 - k) * v` can
overshoot 1.0 by a float rounding error. From then on every export got a final `Clip(-1, 1)`. gen34 predates that fix
and was accepted as it is.

## Try them

With the starter kit installed (see the main [README](../README.md#setup)), the kit's match script plays two models
against each other under the referee search:

```bash
python play/model_vs_model.py --model-a models/gen78.onnx --model-b models/gen34.onnx --games 50
```

Or use our arena, which also reports the result per colour:

```python
from multiprocessing import Pool
from az import arena

if __name__ == "__main__":   # required on Windows, where worker processes re-import this file
    with Pool(6) as pool:
        print(arena.match("models/gen78.onnx", "models/gen34.onnx", games=100, sims=100, pool=pool, seed=1))
```
