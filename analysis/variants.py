"""Export-time model variants: policy temperature and weight averaging (SWA) of consecutive nets.

python variants.py temp <net.pt> <T> <out.onnx>
python variants.py swa <out.onnx> <net_a.pt> <net_b.pt> ...
python variants.py swatemp <T> <out.onnx> <net_a.pt> ...
"""
import sys

import torch
import torch.nn as nn

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from az.net import Net, export_onnx  # noqa: E402


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n = Net(**ck["cfg"])
    n.load_state_dict(ck["net"], strict=False)
    return n.eval(), ck


def average(paths):
    nets = [load(p) for p in paths]
    base, _ = nets[0]
    sd = {}
    for k, v in base.state_dict().items():
        if v.is_floating_point():
            sd[k] = sum(n.state_dict()[k] for n, _ in nets) / len(nets)
        else:
            sd[k] = nets[-1][0].state_dict()[k]
    base.load_state_dict(sd)
    return base.eval()


class Temp(nn.Module):
    def __init__(self, net, t):
        super().__init__()
        self.net, self.t = net, t

    def forward(self, x):
        p, v = self.net(x)
        return p * self.t, v


class VScale(nn.Module):
    def __init__(self, net, s):
        super().__init__()
        self.net, self.s = net, s

    def forward(self, x):
        p, v = self.net(x)
        return p, torch.clamp(v * self.s, -1.0, 1.0)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "vscale":
        net, _ = load(sys.argv[2])
        export_onnx(VScale(net, float(sys.argv[3])), sys.argv[4])
    elif mode == "temp":
        net, _ = load(sys.argv[2])
        export_onnx(Temp(net, float(sys.argv[3])), sys.argv[4])
    elif mode == "swa":
        export_onnx(average(sys.argv[3:]), sys.argv[2])
    elif mode == "swatemp":
        export_onnx(Temp(average(sys.argv[4:]), float(sys.argv[2])), sys.argv[3])
    print("exported", sys.argv)
