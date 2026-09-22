#!/usr/bin/env python3
"""在板上重放 bag，用 rclpy 自己反序列化，导出成 .npz 给本地做数值比较。
⚠️ 为什么不在本地手解 CDR：rosbag2 这份数据的字符串长度把 padding 也算进去
   （实测关节名带尾部 NUL）⇒ 逐字段推 offset 会整体错位，我已经错三次。
   板上有 ROS，让 ROS 自己解，别再手写。
用法: dump_bag.py <bag_dir> <out.npz>
"""
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import Imu, JointState, Joy
from std_msgs.msg import String, Float32MultiArray

bag_dir, out_path = sys.argv[1], sys.argv[2]

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
       rosbag2_py.ConverterOptions("cdr", "cdr"))
topics = {t.name: t.type for t in r.get_all_topics_and_types()}
print("topics:", topics)

data = {k: [] for k in ("meta", "mode", "obs", "imu", "js", "joy")}
kinds = {"/node_metadata": "meta", "/act_mode": "mode", "/policy_obs": "obs",
         "/imu": "imu", "/joint_states": "js", "/joy": "joy"}
while r.has_next():
    topic, raw, t = r.read_next()
    k = kinds.get(topic)
    if k is None:
        continue
    ts = t                  # 纳秒
    if k in ("meta", "mode"):
        data[k].append((ts, deserialize_message(raw, String).data))
    elif k == "obs":
        data[k].append((ts, np.array(deserialize_message(raw, Float32MultiArray).data, dtype=np.float32)))
    elif k == "imu":
        m = deserialize_message(raw, Imu)
        data[k].append((ts, np.array([m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z]),
                        np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]),
                        np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])))
    elif k == "js":
        m = deserialize_message(raw, JointState)
        data[k].append((ts, np.array(m.position), np.array(m.velocity), np.array(m.effort)))
    elif k == "joy":
        m = deserialize_message(raw, Joy)
        data[k].append((ts, np.array(m.axes, dtype=np.float64), np.array(m.buttons, dtype=np.int64)))

save = {}
for k in ("obs", "imu", "js", "joy"):
    if not data[k]:
        print("⚠️ %s 没有数据" % k); continue
    save[k + "_t"] = np.array([x[0] for x in data[k]], dtype=np.int64)
    for i in range(1, len(data[k][0])):
        save["%s_%d" % (k, i)] = np.array([x[i] for x in data[k]])
save["meta_json"] = np.array([x[1] for x in data["meta"]], dtype=object)
save["mode_json"] = np.array([x[1] for x in data["mode"]], dtype=object)
save["mode_t"] = np.array([x[0] for x in data["mode"]], dtype=np.int64)
np.savez(out_path, **save)
print("已导出", out_path)
print("计数:", {k: len(v) for k, v in data.items()})
