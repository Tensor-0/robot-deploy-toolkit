#!/usr/bin/env python3
"""解剖部署的 onnx：结构 / initializer / 是否把 empirical normalization 烤进去了。"""
import numpy as np
import onnx
from onnx import numpy_helper

P = "/tmp/dm10_39dim_policy.onnx"
m = onnx.load(P)
g = m.graph

print("producer:", m.producer_name, m.producer_version)
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
print("inputs :", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input])
print("outputs:", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output])

print("\n---- initializers ----")
for init in g.initializer:
    a = numpy_helper.to_array(init)
    flat = a.reshape(-1) if a.size else a
    head = np.array2string(flat[:4], precision=5, suppress_small=False) if flat.size else "[]"
    print(f"  {init.name:40s} shape={str(list(a.shape)):12s} dtype={a.dtype}"
          f"  mean={float(flat.mean()):+.5f} std={float(flat.std()):.5f}  head={head}")

print(f"\n---- nodes ({len(g.node)}) ----")
for i, n in enumerate(g.node):
    print(f"  [{i:02d}] {n.op_type:12s} in={list(n.input)}  out={list(n.output)}")
    for attr in n.attribute:
        if attr.type == onnx.AttributeProto.INT:
            print(f"        {attr.name}={attr.i}")
        elif attr.type == onnx.AttributeProto.FLOAT:
            print(f"        {attr.name}={attr.f:.4f}")
        elif attr.type == onnx.AttributeProto.INTS:
            print(f"        {attr.name}={list(attr.ints)}")
        elif attr.type == onnx.AttributeProto.STRING:
            print(f"        {attr.name}={attr.s.decode(errors='replace')!r}")

print("\n---- value_info ----")
for vi in list(g.value_info) + list(g.output):
    dims = [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
    print(f"  {vi.name:30s} {dims}")
