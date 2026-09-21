#!/usr/bin/env python3
"""在【板子上】用 rclpy 官方反序列化把 bag 导出成 npz。

为什么在板子上跑：本机没有 ROS，手写 CDR 反序列化已经在 name 数组的
变长字符串对齐上错了 5 次。板子上有完整的 rclpy + rosbag2_py，
用官方 API 是唯一可靠的路 —— 别再手写 CDR 了。
"""
import sys
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG = Path.home() / "bags" / "run2"
OUT = Path("/tmp/real_run_export.npz")

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=str(BAG), storage_id="sqlite3"),
    rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                output_serialization_format="cdr"),
)
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
print("话题:", types)

msgs = {k: [] for k in types}
stamps = {k: [] for k in types}
while reader.has_next():
    topic, data, t = reader.read_next()
    msgs[topic].append(deserialize_message(data, get_message(types[topic])))
    stamps[topic].append(t)

for k in msgs:
    print(f"  {k}: {len(msgs[k])} 条")

out = {}
for k in types:
    if not msgs[k]:
        continue
    tag = k.strip("/").replace("/", "_")
    out[f"{tag}__t"] = np.array(stamps[k], dtype=np.int64)
    m0 = msgs[k]
    if hasattr(m0[0], "position"):
        out[f"{tag}__name"] = np.array(list(m0[0].name))
        out[f"{tag}__position"] = np.array([list(m.position) for m in m0])
        out[f"{tag}__velocity"] = np.array([list(m.velocity) for m in m0])
        out[f"{tag}__effort"] = np.array([list(m.effort) for m in m0])
    elif hasattr(m0[0], "orientation"):
        out[f"{tag}__quat"] = np.array([[m.orientation.w, m.orientation.x,
                                         m.orientation.y, m.orientation.z] for m in m0])
        out[f"{tag}__angvel"] = np.array([[m.angular_velocity.x, m.angular_velocity.y,
                                           m.angular_velocity.z] for m in m0])
        out[f"{tag}__linacc"] = np.array([[m.linear_acceleration.x, m.linear_acceleration.y,
                                           m.linear_acceleration.z] for m in m0])

np.savez_compressed(OUT, **out)
print(f"\n✅ 写出 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
for k in sorted(out):
    print(f"   {k:28s} {str(out[k].shape):16s} {out[k].dtype}")
