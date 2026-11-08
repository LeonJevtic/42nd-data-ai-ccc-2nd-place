"""Recompute a net's BatchNorm running statistics on main training positions only (weights unchanged).

python bn_recal.py net_N.pt first_data_it last_data_it out.onnx
"""
import os, sys
import numpy as np, torch
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from az.net import Net, export_onnx

src, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ck = torch.load(src, map_location="cuda", weights_only=False)
net = Net(**ck["cfg"]).cuda()
net.load_state_dict(ck["net"], strict=False)
obs = np.concatenate([np.load(os.path.join(ROOT, "runs", "r1", f"data_{i:04d}.npz"))["obs"] for i in range(lo, hi + 1)])
bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
for m in bns:
    m.reset_running_stats(); m.momentum = None   # cumulative average over all recalibration batches
reflect = lambda o: o.transpose(2, 3).flip(2).flip(3)
g = torch.Generator(device="cpu").manual_seed(0)
net.train()
with torch.no_grad():
    for _ in range(200):
        sel = torch.randint(0, len(obs), (1024,), generator=g)
        x = torch.as_tensor(obs[sel.numpy()], device="cuda").float() / 8.0
        flip = torch.rand(1024, device="cuda") < 0.5
        net(torch.where(flip[:, None, None, None], reflect(x), x))
net.eval()
old = Net(**ck["cfg"]); old.load_state_dict(ck["net"], strict=False)
d = [float((a.running_mean.cpu() - b.running_mean).abs().max()) for a, b in zip(bns, [m for m in old.modules() if isinstance(m, torch.nn.BatchNorm2d)])]
print(f"recalibrated {len(bns)} BN layers on {len(obs)} positions; max |delta running_mean| per layer: {[round(x, 3) for x in d]}")
export_onnx(net, out)
print("->", out)
