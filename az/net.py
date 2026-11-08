"""Policy-value network and ONNX export."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    """ResNet trunk; bilinear from/to policy head; scalar value head."""

    def __init__(self, channels=64, blocks=6, d=32):
        super().__init__()
        self.channels, self.blocks, self.d = channels, blocks, d
        self.stem = nn.Sequential(nn.Conv2d(16, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # policy: per-square from/to embeddings, logit = <from, to> + from bias + to bias + geometry bias
        self.p_from = nn.Conv2d(channels, d + 1, 1)
        self.p_to = nn.Conv2d(channels, d + 1, 1)
        self.geo = nn.Parameter(torch.zeros(4096))
        # value
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.v_fc1 = nn.Linear(4 * 64 + channels, 128)
        self.v_fc2 = nn.Linear(128, 1)
        # kill head: per-square "the side to move destroys the enemy Kernel this turn" score,
        # max-pooled. Starts switched off so older checkpoints keep their value function.
        self.k_conv = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.k_conv.weight)
        nn.init.constant_(self.k_conv.bias, -6.0)
        # training-only head: material balance 8 actions ahead (teaches exchange outcomes)
        self.m_fc = nn.Linear(128, 1)

    def forward(self, x, return_kill=False):
        h = self.trunk(self.stem(x))
        d = self.d
        f = self.p_from(h).reshape(-1, d + 1, 64)            # (B, d+1, 64)
        t = self.p_to(h).reshape(-1, d + 1, 64)
        fe, fb = f[:, :d], f[:, d:]                            # (B,d,64), (B,1,64)
        te, tb = t[:, :d], t[:, d:]
        logits = torch.matmul(fe.transpose(1, 2), te) * (d ** -0.5)   # (B, 64from, 64to)
        logits = logits + fb.transpose(1, 2) + tb                      # + (B,64,1) + (B,1,64)
        policy = logits.reshape(-1, 4096) + self.geo

        v = self.v_conv(h).flatten(1)
        g = h.mean(dim=(2, 3))
        v = F.relu(self.v_fc1(torch.cat([v, g], dim=1)))
        v_raw = torch.tanh(self.v_fc2(v)).reshape(-1)
        k_logit = torch.amax(self.k_conv(h), dim=(2, 3)).reshape(-1)
        k = torch.sigmoid(k_logit)
        # a kill this turn is a certain win; otherwise fall back to the learned value
        value = k + (1 - k) * v_raw
        if return_kill:
            return policy, value, k_logit, torch.tanh(self.m_fc(v)).reshape(-1)
        return policy, value


def export_onnx(net, path):
    import copy
    m = copy.deepcopy(net).cpu().eval()
    dummy = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(
        m, dummy, str(path),
        input_names=["observation"], output_names=["policy", "value"],
        dynamic_axes={"observation": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    # the exporter leaves the policy width symbolic; the referee wants a concrete 4096
    import onnx
    model = onnx.load(str(path))
    dim = model.graph.output[0].type.tensor_type.shape.dim[1]
    dim.ClearField("dim_param")
    dim.dim_value = 4096
    # 'Constant' is not on the referee's allow-list: turn constants into initializers
    from onnx import numpy_helper
    keep = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            arr = numpy_helper.to_array(node.attribute[0].t)
            model.graph.initializer.append(numpy_helper.from_array(arr, node.output[0]))
        else:
            keep.append(node)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    clip_value(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def clip_value(model):
    """Clamp the value output to [-1, 1] in place: k + (1 - k) * v can overshoot 1 by a float
    rounding step, and the referee rejects any value outside the range."""
    import numpy as np
    from onnx import helper, numpy_helper
    g = model.graph
    out = g.output[1].name
    if any(n.op_type == "Clip" and n.output[0] == out for n in g.node):
        return model
    for n in g.node:
        n.output[:] = [out + "_unclipped" if o == out else o for o in n.output]
    g.initializer.extend([numpy_helper.from_array(np.array(-1.0, np.float32), "value_lo"),
                          numpy_helper.from_array(np.array(1.0, np.float32), "value_hi")])
    g.node.append(helper.make_node("Clip", [out + "_unclipped", "value_lo", "value_hi"], [out]))
    return model


def clip_value_file(src, dst):
    import onnx
    model = clip_value(onnx.load(str(src)))
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
