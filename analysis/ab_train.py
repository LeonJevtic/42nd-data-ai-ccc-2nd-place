"""Re-run one training iteration of runs/r1 from a saved checkpoint with a different learning rate.

python ab_train.py <ckpt.pt> <it> <lr> <out.onnx> <seed>
Mirrors az/train.py main(): window 6, aux_window 3 + synth file, steps = samples(it) * reuse / batch.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from az.net import Net, export_onnx  # noqa: E402
from az.train import train_steps  # noqa: E402

RUN = os.path.join(ROOT, "runs", "r1")
SYNTH = os.path.join(ROOT, "matches", "synth_kill.npz")


def main():
    ckpt, it, lr, out, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    torch.manual_seed(seed)
    dev = "cuda"
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    assert ck["it"] == it - 1, (ck["it"], it)
    net = Net(64, 6).to(dev)
    net.load_state_dict(ck["net"], strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    opt.load_state_dict(ck["opt"])
    for g in opt.param_groups:
        g["lr"] = lr
    parts = [np.load(os.path.join(RUN, f"data_{i:04d}.npz")) for i in range(it - 5, it + 1)]
    buf = {
        "obs": torch.as_tensor(np.concatenate([q["obs"] for q in parts]), device=dev),
        "idx": torch.as_tensor(np.concatenate([q["idx"] for q in parts]).astype(np.int16), device=dev),
        "prob": torch.as_tensor(np.concatenate([q["prob"] for q in parts]).astype(np.float16), device=dev),
        "v": torch.as_tensor(np.concatenate([q["v"] for q in parts]), device=dev),
        "kill": torch.as_tensor(np.concatenate([q["kill"] for q in parts]), device=dev),
        "mat": torch.as_tensor(np.concatenate([q["mat"] for q in parts]), device=dev),
    }
    aux_parts = [q for q in parts if "aux_obs" in q][-3:] + [np.load(SYNTH)]
    buf["aux_obs"] = torch.as_tensor(np.concatenate([q["aux_obs"] for q in aux_parts]), device=dev)
    buf["aux_kill"] = torch.as_tensor(np.concatenate([q["aux_kill"] for q in aux_parts]), device=dev)
    steps = max(200, int(len(parts[-1]["v"]) * 6.0 / 1024))
    lp, lv, lk, lm, *_ = train_steps(net, opt, buf, steps, 1024, dev)
    export_onnx(net, out)
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it,
                "cfg": {"channels": 64, "blocks": 6}}, out[:-5] + ".pt")
    print(f"it {it} lr {lr} seed {seed}: {steps} steps policy {lp:.3f} value {lv:.3f} kill {lk:.3f} mat {lm:.3f} -> {out}")


if __name__ == "__main__":
    main()
