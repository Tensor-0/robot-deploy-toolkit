#!/usr/bin/env python3
"""⭐ 决定性对照：真机 /action  vs  仿真 act

原理
----
真机: /action 发的就是 act_[i]（缩放+偏移后的关节目标，关节序）
仿真: ref_trace.npz 里的 act 是策略原始输出，target = act*0.25 + default

⇒ 真机 /action 应等于仿真的 target（不是 act！）
⇒ 而且【第一帧】最有价值：此时真机状态 ≈ 仿真的 home 初始态（都静止、都微屈）

判据
----
同 obs ⇒ 同 act。真机静止时 obs 应 ≈ 仿真的 obs[0]
  ⇒ 真机 /action[第一帧] ≈ 仿真 target[0]
  相等 ⇒ 软件链路 100% 正确
  不等 ⇒ 软件有问题，看差在哪一段
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np

REAL = Path("/home/zhan/robot-deploy-toolkit/results/dm10_real_run/real_run.db3")
REF = Path("/home/zhan/robot-deploy-toolkit/results/dm10_ref_run/ref_trace.npz")

SHORT = ["左髋pitch", "左髋roll", "左髋yaw", "左膝", "左踝",
         "右髋pitch", "右髋roll", "右髋yaw", "右膝", "右踝"]


# ---------------------------------------------------------------- CDR 解析
def cdr_jointstate(b: bytes):
    """sensor_msgs/JointState 的 CDR 反序列化（含 4 字节对齐）。"""
    o = 4                                     # skip CDR encapsulation header
    sec, nsec = struct.unpack_from("<iI", b, o); o += 8
    ln = struct.unpack_from("<I", b, o)[0]; o += 4
    frame = b[o:o + ln].decode(errors="replace"); o += ln
    o += (-o) % 4                             # ⭐ CDR 4 字节对齐
    out = {}
    for label in ("name", "position", "velocity", "effort"):
        n = struct.unpack_from("<I", b, o)[0]; o += 4
        if label == "name":
            names = []
            for _ in range(n):
                sl = struct.unpack_from("<I", b, o)[0]; o += 4
                names.append(b[o:o + sl].decode(errors="replace")); o += sl
                o += (-o) % 4
            out["name"] = names
        else:
            v = list(struct.unpack_from(f"<{n}d", b, o)); o += 8 * n
            out[label] = v
    return out


def load_topic(db3, topic_name):
    c = sqlite3.connect(str(db3))
    tid = list(c.execute("SELECT id FROM topics WHERE name=?", (topic_name,)))[0][0]
    rows = list(c.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)))
    c.close()
    return rows


print("=" * 84)
print("真机 /action  vs  仿真 target")
print("=" * 84)

act_rows = load_topic(REAL, "/action")
js_rows = load_topic(REAL, "/joint_states")
print(f"真机 /action      {len(act_rows)} 条")
print(f"真机 /joint_states {len(js_rows)} 条")

t0 = act_rows[0][0]
print(f"起始时间戳 {t0/1e9:.3f}  跨度 {(act_rows[-1][0]-t0)/1e9:.1f} s  "
      f"频率 {len(act_rows)/((act_rows[-1][0]-t0)/1e9):.1f} Hz")

# ---------------------------------------------------------------- 找"策略在跑"的段
# PD 模式下 act 恒为 default；策略模式下会变 ⇒ 用"与 default 的偏差"定位
# ⚠️ FROZEN at the BENT-KNEE frame — do NOT "update" these to the straight-leg
# values. This script analyses the 2026-09 real-robot session, whose policy was
# trained bent-knee. On 2026-09-21 the project moved the sim home and the deploy
# `joint_default_angle` to straight-leg, shifting the action offset and obs zero
# point by a constant; re-pointing these numbers would silently corrupt every
# result below. Rollback: ~/dm10-home-bent-backup-20260921/README.md
DEFAULT = np.array([-0.4, 0, 0, 0.8, -0.4, -0.4, 0, 0, 0.8, -0.4])
acts, times = [], []
for ts, blob in act_rows:
    d = cdr_jointstate(blob)
    acts.append(d["position"])
    times.append(ts)
acts = np.array(acts)
dev = np.abs(acts - DEFAULT).max(axis=1)

print()
print("── 与默认角的最大偏差（>0.02 认为策略在动）──")
moving = dev > 0.02
print(f"  变动帧数 {int(moving.sum())} / {len(acts)}")
if moving.any():
    idx = np.flatnonzero(moving)
    print(f"  第一段变动: 第 {idx[0]} 帧 (t={(times[idx[0]]-t0)/1e9:.2f}s)  "
          f"到 第 {idx[-1]} 帧 (t={(times[idx[-1]]-t0)/1e9:.2f}s)")
    # 找连续段
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 5:
            segs.append((start, a)); start = b
    segs.append((start, idx[-1]))
    print(f"  连续段 {len(segs)} 个:")
    for s, e in segs:
        print(f"    帧 {s}-{e}  ({(e-s+1)*0.02:.1f}s)  峰值偏差 {dev[s:e+1].max():.3f}")

# ---------------------------------------------------------------- 决定性对照
ref = np.load(REF, allow_pickle=True)
ref_act = ref["act"]          # 策略原始输出
ref_tgt = ref["target"]       # = act*scale + default  ← 对应真机 /action
ref_obs = ref["obs"]
scale = float(ref["action_scale"])
defq = ref["joint_default"]

print()
print("=" * 84)
print("★ 逐点对照")
print("=" * 84)
np.set_printoptions(precision=4, suppress=True, linewidth=200)

print()
print("仿真 act[0]（策略原始输出第 1 帧）:")
print("  ", ref_act[0])
print("仿真 target[0] = act*0.25 + default（应等于真机 /action）:")
print("  ", ref_tgt[0])
print("仿真 obs[0]:")
print("   ang_vel ", ref_obs[0, 0:3], " gravity ", ref_obs[0, 3:6])
print("   dof_pos ", ref_obs[0, 6:16])
print("   dof_vel ", ref_obs[0, 16:26])
print("   lastact ", ref_obs[0, 26:36])
print("   command ", ref_obs[0, 36:39])

# 真机策略段的第一帧
if moving.any():
    i0 = int(np.flatnonzero(moving)[0])
    print()
    print(f"真机 /action 策略段第 1 帧 (帧 {i0}, t={(times[i0]-t0)/1e9:.2f}s):")
    print("  ", acts[i0])
    print()
    print("── 逐关节对比 ──")
    print(f"{'关节':<10}{'仿真target[0]':>16}{'真机/action':>16}{'差':>12}")
    for k in range(10):
        dd = acts[i0][k] - ref_tgt[0][k]
        print(f"{SHORT[k]:<10}{ref_tgt[0][k]:>16.4f}{acts[i0][k]:>16.4f}{dd:>12.4f}")
    print()
    print(f"  最大绝对差: {np.abs(acts[i0]-ref_tgt[0]).max():.4f} rad "
          f"({np.abs(acts[i0]-ref_tgt[0]).max()*180/np.pi:.2f}°)")

# ---------------------------------------------------------------- 真机 vs 仿真 轨迹统计
if moving.any():
    print()
    print("=" * 84)
    print("★ 真机策略段的动作幅度 vs 仿真")
    print("=" * 84)
    seg = acts[moving]
    print(f"{'关节':<10}{'真机 幅度':>14}{'仿真 幅度':>14}{'真机 |act|max':>16}{'仿真 |act|max':>16}")
    for k in range(10):
        ra = seg[:, k].max() - seg[:, k].min()
        sa = ref_tgt[:, k].max() - ref_tgt[:, k].min()
        print(f"{SHORT[k]:<10}{ra:>14.4f}{sa:>14.4f}"
              f"{np.abs(seg[:, k] - DEFAULT[k]).max():>16.4f}"
              f"{np.abs(ref_tgt[:, k] - DEFAULT[k]).max():>16.4f}")
