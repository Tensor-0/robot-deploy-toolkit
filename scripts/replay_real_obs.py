#!/usr/bin/env python3
"""从真机 bag 离线重建 obs → 喂同一个 onnx → 和真机 /action 比。

这是最强的定位手段：
  · 若 离线算出的 act == 真机 /action  ⇒ 推理链路 100% 正确，问题在物理侧
  · 若 不等                            ⇒ 软件有问题，而且能看出错在哪一段

obs 定序（已核实 = 训练侧）:
  ang_vel:3 | gravity_b:3 | dof_pos:10 | dof_vel:10 | last_action:10 | cmd_vel:3
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort

REAL = Path("/home/zhan/robot-deploy-toolkit/results/dm10_real_run/real_run.db3")
# 板上那份 onnx 的本地副本。原为 /tmp/dm10_39dim_policy.onnx —— 重启就没了，
# 2026-09-21 移到备份目录。HANDOFF §8.6 已用【权重指纹】证明它 ≡
# logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-09_00-52-12_mujoco/policy.onnx。
# ⚠️ 这是【屈膝参考系】时代的存档，与当前仓库状态（直腿 home）不匹配，
#    只用于复核 2026-09 那次真机分析的结论，不能拿它跑新东西。
ONNX = Path("/home/zhan/dm10-home-bent-backup-20260921/policy_run/policy.onnx")
# ⚠️ FROZEN at the BENT-KNEE frame — do NOT "update" these to the straight-leg
# values. This script replays the 2026-09 real-robot bag, whose policy was trained
# bent-knee. On 2026-09-21 the project moved the sim home and the deploy
# `joint_default_angle` to straight-leg, shifting the action offset and obs zero
# point by a constant; re-pointing these numbers would silently corrupt every
# result below. Rollback: ~/dm10-home-bent-backup-20260921/README.md
DEFAULT = np.array([-0.4, 0, 0, 0.8, -0.4, -0.4, 0, 0, 0.8, -0.4], dtype=np.float64)
SHORT = ["左髋p", "左髋r", "左髋y", "左膝", "左踝", "右髋p", "右髋r", "右髋y", "右膝", "右踝"]


def cdr_jointstate(b):
    """sensor_msgs/JointState 的 CDR 反序列化。

    ⚠️ 这条注释是花了很多轮试错换来的，别再自己推：
      CDR 的对齐规则在【不同消息】上表现不一致（action 和 joint_states 的
      字符串长度不同 8/9 vs 7/8，导致后续 offset 的 4/8 对齐落点不同，
      纯手推极易错）。本函数改用【数值合法性扫描】自动定位数组起点：
        · 先按标准规则快速走一遍
        · 若走出来的值不合理（|q|>10）或有越界，就从当前 offset 起
          在 ±8 字节内搜索「n==目标长度 且 10 个 double 都合理」的位置
    """
    def _read_strings(o):
        o += (-o) % 4
        n = struct.unpack_from("<I", b, o)[0]; o += 4
        names = []
        for _ in range(n):
            sl = struct.unpack_from("<I", b, o)[0]; o += 4
            names.append(b[o:o + sl].decode(errors="replace")); o += sl
            o += (-o) % 4
        return names, o

    o = 4 + 8                                     # CDR hdr + stamp
    ln = struct.unpack_from("<I", b, o)[0]; o += 4 + ln
    names, o = _read_strings(o)

    def find_array(o0, want_n):
        """从 o0 起向后全局搜索 (n==want_n 且 10 个 double 都合理) 的位置。

        ⚠️ 为什么不用推算：CDR 里 name 数组的字符串长度随关节名变化
        （action_1..10 = 8/9 字节，joint_1..10 = 7/8 字节），
        导致后续 offset 的对齐落点不同，纯手推极易错。
        「找一组合法的数值」是自校验的，比推 offset 可靠得多。
        """
        for cand in range(max(4, o0 - 16), min(len(b) - 84, o0 + 64)):
            if b[cand:cand + 4] != struct.pack("<I", want_n):
                continue
            for dstart in (cand + 4, (cand + 4 + 7) // 8 * 8):
                if dstart + 8 * want_n > len(b):
                    continue
                v = np.frombuffer(b[dstart:dstart + 8 * want_n], dtype="<f8")
                if np.all(np.abs(v) < 20):
                    return want_n, dstart, v
        return None

    out = {"name": names}
    for label in ("position", "velocity", "effort"):
        if o >= len(b):
            out[label] = np.zeros(0); continue
        if label != "position":
            o += (-o) % 8
        else:
            o += (-o) % 4
        if o + 4 > len(b):
            out[label] = np.zeros(0); continue
        n = struct.unpack_from("<I", b, o)[0]
        end_ok = (o + 4 + 8 * n) <= len(b) and n <= 64
        vals_ok = False
        if end_ok and n > 0:
            for dstart in (o + 4, (o + 4 + 7) // 8 * 8):
                if dstart + 8 * n <= len(b):
                    v = np.frombuffer(b[dstart:dstart + 8 * n], dtype="<f8")
                    if np.all(np.abs(v) < 20):
                        o, vals_ok = dstart, True
                        out[label] = np.array(v); break
        if not vals_ok:
            got = find_array(o, n if end_ok else 10)
            if got:
                _, o, v = got
                out[label] = np.array(v)
            else:
                out[label] = np.zeros(0)
        o += 8 * len(out[label])
    return out


def cdr_imu(b):
    """sensor_msgs/Imu: stamp(8) frame_id(string) orientation(4f64)
       orientation_covariance(9f64) angular_velocity(3f64)
       angular_velocity_covariance(9f64) linear_acceleration(3f64) ..."""
    o = 4
    o += 8
    ln = struct.unpack_from("<I", b, o)[0]; o += 4 + ln
    o += (-o) % 4
    orient = np.array(struct.unpack_from("<4d", b, o)); o += 32
    o += 72                                     # orientation_covariance 9d
    angvel = np.array(struct.unpack_from("<3d", b, o)); o += 24
    o += 72
    linacc = np.array(struct.unpack_from("<3d", b, o)); o += 24
    return orient, angvel, linacc


def load(db3, topic):
    c = sqlite3.connect(str(db3))
    tid = list(c.execute("SELECT id FROM topics WHERE name=?", (topic,)))[0][0]
    rows = list(c.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)))
    c.close()
    return rows


ja = load(REAL, "/action")
jj = load(REAL, "/joint_states")
ji = load(REAL, "/imu")

print("=" * 88)
print("真机 obs 重建 → onnx → 对比 /action")
print("=" * 88)

# 全部反序列化
A = [cdr_jointstate(b) for _, b in ja]
J = [cdr_jointstate(b) for _, b in jj]
I = [cdr_imu(b) for _, b in ji]
ta = np.array([t for t, _ in ja])
tj = np.array([t for t, _ in jj])
ti = np.array([t for t, _ in ji])

names = A[0]["name"]
print("关节名:", names)
pos_a = np.array([a["position"] for a in A])
pos_j = np.array([j["position"] for j in J])
vel_j = np.array([j["velocity"] for j in J])
ori_i = np.array([i[0] for i in I])
av_i = np.array([i[1] for i in I])
la_i = np.array([i[2] for i in I])

print(f"/action {len(pos_a)}  /joint_states {len(pos_j)}  /imu {len(ori_i)}")
dur = (ta[-1] - ta[0]) / 1e9
print(f"跨度 {dur:.1f}s   /action {len(pos_a)/dur:.1f} Hz")

# ---------------------------------------------------------------- 分成 PD 段 / 策略段
dev = np.abs(pos_a - DEFAULT).max(axis=1)
moving = dev > 0.05
print(f"\n策略在动帧数 {int(moving.sum())}/{len(pos_a)}")

# 连续段
segs, idx = [], np.flatnonzero(moving)
if len(idx):
    s = idx[0]
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if b_ - a_ > 10:
            segs.append((s, a_)); s = b_
    segs.append((s, idx[-1]))
print(f"策略段 {len(segs)} 个:")
for s, e in segs:
    print(f"  帧 {s}-{e}  ({(ta[e]-ta[s])/1e9:.1f}s)  t0={(ta[s]-ta[0])/1e9:.1f}s")

# ---------------------------------------------------------------- 时间对齐到 /joint_states
def nearest(arr_t, t):
    return int(np.argmin(np.abs(arr_t - t)))

sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
onames = sess.get_outputs()[0].name

for si, (s, e) in enumerate(segs):
    print()
    print("=" * 88)
    print(f"策略段 {si+1}: 帧 {s}..{e}   ({(ta[e]-ta[s])/1e9:.1f}s)")
    print("=" * 88)

    # 对准第一帧
    t_start = ta[s]
    js_i = nearest(tj, t_start)
    imu_i = nearest(ti, t_start)

    q = pos_j[js_i]
    qd = vel_j[js_i]
    quat = ori_i[imu_i]          # w,x,y,z
    av = av_i[imu_i]

    # gravity_b = R^T * (0,0,-1)
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    g_b = R.T @ np.array([0.0, 0.0, -1.0])

    last_act = np.zeros(10)      # 刚启动时 = 0
    cmd = np.array([0.5, 0.0, 0.0])   # 摇杆推的，未知；先试 0.4

    print(f"\n真机起始姿态 q  ({tj[js_i]/1e9:.3f}):")
    print("  ", np.array2string(q, precision=4, suppress_small=True))
    print(f"  与默认角之差:")
    print("  ", np.array2string(q - DEFAULT, precision=4, suppress_small=True))
    print(f"\nIMU 四元数 {quat}  ⇒ gravity_b = {np.array2string(g_b, precision=4)}")
    print(f"角速度     = {np.array2string(av, precision=4)}")
    print(f"关节速度   = {np.array2string(qd, precision=4)}")

    for cmd_vx in (0.4, 0.5):
        obs = np.concatenate([av, g_b, q - DEFAULT, qd, last_act, [cmd_vx, 0, 0]])
        out = sess.run([onames], {iname: obs.astype(np.float32)[None, :]})[0][0]
        tgt = out * 0.25 + DEFAULT
        print(f"\n  cmd_vx={cmd_vx} ⇒ 离线算出的 act:")
        print("   ", np.array2string(out, precision=4, suppress_small=True))
        print(f"  ⇒ target = act*0.25+default:")
        print("   ", np.array2string(tgt, precision=4, suppress_small=True))
        if cmd_vx == 0.4:
            print(f"\n  真机 /action 第 {s} 帧:")
            print("   ", np.array2string(pos_a[s], precision=4, suppress_small=True))
            print(f"\n  {'关节':<8}{'离线target':>14}{'真机/action':>14}{'差':>12}")
            for k in range(10):
                print(f"  {SHORT[k]:<8}{tgt[k]:>14.4f}{pos_a[s][k]:>14.4f}"
                      f"{pos_a[s][k]-tgt[k]:>12.4f}")
            print(f"\n  最大差 {np.abs(pos_a[s]-tgt).max():.4f} rad "
                  f"({np.abs(pos_a[s]-tgt).max()*180/np.pi:.1f}°)")
