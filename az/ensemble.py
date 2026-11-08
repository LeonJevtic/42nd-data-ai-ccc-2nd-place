"""Average exported nets into one ONNX model: mean policy logits and mean value.

python -m az.ensemble out.onnx runs/r1/genA.onnx runs/r1/genB.onnx
"""
import sys

import numpy as np
import onnx
from onnx import compose, helper, numpy_helper


def ensemble(paths, out):
    base = onnx.load(paths[0])
    nodes, inits = [], []
    for i, p in enumerate(paths):
        g = compose.add_prefix(onnx.load(p), f"m{i}_").graph
        for n in g.node:
            n.input[:] = ["observation" if x == f"m{i}_observation" else x for x in n.input]
            nodes.append(n)
        inits.extend(g.initializer)
    inits.append(numpy_helper.from_array(np.array(1.0 / len(paths), np.float32), "ens_w"))
    for head in ("policy", "value"):
        acc = f"m0_{head}"
        for i in range(1, len(paths)):   # chained Add: only ops the referee already accepts
            nodes.append(helper.make_node("Add", [acc, f"m{i}_{head}"], [f"ens_{head}_{i}"]))
            acc = f"ens_{head}_{i}"
        nodes.append(helper.make_node("Mul", [acc, "ens_w"], [head]))
    graph = helper.make_graph(nodes, "ensemble", list(base.graph.input), list(base.graph.output), inits)
    model = helper.make_model(graph, opset_imports=list(base.opset_import), ir_version=base.ir_version)
    onnx.checker.check_model(model)
    onnx.save(model, out)


if __name__ == "__main__":
    ensemble(sys.argv[2:], sys.argv[1])
