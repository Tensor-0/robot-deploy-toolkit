#!/usr/bin/env python3
"""权重级比对：部署的 onnx 是不是本机某个训练 run 的产物？

字节哈希不同可能只是导出环境差异；真正判据是【权重张量】。
"""
import glob
import hashlib
import sys

import onnx
from onnx import numpy_helper
import numpy as np

TARGET = "/tmp/dm10_39dim_policy.onnx"


def weight_fingerprint(path):
    """把所有 initializer 按 [名字排序] 后拼接，取哈希。"""
    m = onnx.load(path)
    items = []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        items.append((init.name, arr.shape, arr.dtype.str, hashlib.md5(arr.tobytes()).hexdigest()))
    items.sort()
    blob = repr(items).encode()
    return hashlib.md5(blob).hexdigest(), items


def io_signature(path):
    m = onnx.load(path)
    def shapes(vals):
        out = []
        for v in vals:
            dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in v.type.tensor_type.shape.dim]
            out.append((v.name, dims))
        return out
    return shapes(m.graph.input), shapes(m.graph.output)


if __name__ == "__main__":
    fp_t, items_t = weight_fingerprint(TARGET)
    in_t, out_t = io_signature(TARGET)
    n_params = sum(int(np.prod(s)) for _, s, _, _ in items_t)

    print(f"部署的 onnx: {TARGET}")
    print(f"  权重指纹 : {fp_t}")
    print(f"  参数个数 : {n_params}")
    print(f"  输入     : {in_t}")
    print(f"  输出     : {out_t}")
    print(f"  initializer 层数: {len(items_t)}")
    print()

    cands = sorted(glob.glob(
        "/home/zhan/UniLab/logs/**/DM10JoystickFlat/**/policy.onnx", recursive=True))
    print(f"候选 {len(cands)} 个，逐个比对权重指纹：")
    hits = []
    for c in cands:
        try:
            fp, _ = weight_fingerprint(c)
        except Exception as e:
            print(f"  ✗ {c}: {e}")
            continue
        mark = "  ✅ 命中" if fp == fp_t else ""
        if fp == fp_t:
            hits.append(c)
        print(f"  {fp[:12]}  {c.replace('/home/zhan/UniLab/logs/', '')}{mark}")

    print()
    if hits:
        print(f"✅ 找到 {len(hits)} 个权重完全相同的 run：")
        for h in hits:
            print("   ", h)
    else:
        print("❌ 没有任何本地 run 的权重与部署的 onnx 相同 ⇒ 出处仍未定位")
