#!/usr/bin/env python3
"""把仿真参考轨迹变成【真机可对照的判据清单】。

思路：真机上你能录到的是 /action（下发目标角）和 /joint_states（实测角）。
所以判据必须建立在这两个量上，而且要是【一眼能判】的数，不是让你看波形。

输出三样：
  ① 逐关节表：稳态值 / 摆动幅度 / 摆动频率 / 相位（谁是领先腿）
  ② 前 1 秒逐帧表：起转瞬间的行为 —— 真机第一次跑最容易在这暴露
  ③ 一张「该是什么样 vs 蹭地走是什么样」的判据卡
"""
from pathlib import Path

import numpy as np

D = Path("/home/zhan/robot-deploy-toolkit/results/dm10_ref_run")
d = np.load(D / "ref_trace.npz", allow_pickle=True)

dt = float(d["dt"])
q, act, tgt = d["q"], d["act"], d["target"]
names = [str(x) for x in d["joint_names"]]
defq = d["joint_default"]
feet, base = d["feet"], d["base"]
T = len(q)

SHORT = {"leg_l1_joint": "左髋pitch", "leg_l2_joint": "左髋roll", "leg_l3_joint": "左髋yaw",
         "leg_l4_joint": "左膝", "leg_l5_joint": "左踝",
         "leg_r1_joint": "右髋pitch", "leg_r2_joint": "右髋roll", "leg_r3_joint": "右髋yaw",
         "leg_r4_joint": "右膝", "leg_r5_joint": "右踝"}

print("=" * 96)
print(f"仿真参考跑判据清单   run={d['run']}   {T} 步 × {dt}s = {T * dt:.1f}s   指令 vx={float(d['cmd_vx'])}")
print("=" * 96)

# ---------------------------------------------------------------- ① 逐关节
print()
print("① 逐关节：稳态值 → 摆动幅度 → 摆动频率 → 相对相位")
print("-" * 96)
print(f"{'关节':<10}{'默认角':>9}{'稳态均值':>10}{'q 范围':>20}{'摆幅(rad)':>11}{'摆幅(°)':>9}"
      f"{'频率Hz':>9}{'相位°':>8}")
print("-" * 96)

band = (T - int(4.0 / dt))          # 用最后 4 秒（已进入稳态）
rows = {}
for i, nm in enumerate(names):
    qi = q[band:, i]
    mean = float(qi.mean())
    lo, hi = float(qi.min()), float(qi.max())
    amp = hi - lo
    # 过零/峰值计数 → 频率
    c = qi - mean
    peaks = int(np.sum((c[:-2] < c[1:-1]) & (c[1:-1] > c[2:])))
    freq = peaks / (band * dt) if band > 0 else float("nan")
    rows[nm] = dict(mean=mean, lo=lo, hi=hi, amp=amp, freq=freq, sig=c)

# 相位：以左髋 pitch 为基准做互相关
ref = rows["leg_l1_joint"]["sig"]
for nm in names:
    s = rows[nm]["sig"]
    if s.std() < 1e-6:
        rows[nm]["phase"] = float("nan")
        continue
    n = len(s)
    cc = np.correlate(s - s.mean(), ref - ref.mean(), mode="full")[n - 1:]
    lag = int(np.argmax(cc))
    rows[nm]["phase"] = lag * dt * 360.0 * max(rows["leg_l1_joint"]["freq"], 1e-9)

for i, nm in enumerate(names):
    r = rows[nm]
    print(f"{SHORT[nm]:<10}{defq[i]:>+9.3f}{r['mean']:>+10.3f}"
          f"{'[' + format(r['lo'], '+.3f') + ', ' + format(r['hi'], '+.3f') + ']':>20}"
          f"{r['amp']:>11.3f}{r['amp'] * 180 / np.pi:>9.1f}"
          f"{r['freq']:>9.2f}{r['phase']:>8.0f}")

print()
print("解读提示：")
print("  · 摆幅(°) < 5° 的关节 = 基本没在动；行走时【膝】应当明显折叠")
print("  · 相位：左髋pitch 与【右髋pitch】应相差约 180°（左右交替）")
print("  · 相位全同 ⇒ 双腿同步跳，不是走")

# ---------------------------------------------------------------- ② 足端/底盘
print()
print("② 足端与底盘（判断是不是蹭地走的硬指标）")
print("-" * 96)
lb, rb = 1, 2   # feet[:, body, xyz]，body 序 = [base_link, leg_l5_link, leg_r5_link]
for label, bi in (("左脚 leg_l5", lb), ("右脚 leg_r5", rb)):
    z = feet[:, bi, 2]
    zmin = float(np.percentile(z, 1))
    clear = float(z.max() - zmin)
    air = z > (zmin + 0.02)
    trans = int(np.sum((~air[:-1]) & air[1:]))
    freq = trans / (T * dt) if trans else float("nan")
    dxy = np.linalg.norm(np.diff(feet[:, bi, :2], axis=0), axis=1) / dt
    contact = ~air[:-1]
    slip = float(dxy[contact].mean()) if contact.sum() > 2 else float("nan")
    print(f"  {label}: 抬脚高度 {clear * 100:5.1f} cm | 腾空占比 {air.mean() * 100:5.1f}% | "
          f"步频 {freq:4.2f} Hz | 接触滑移 {slip:.3f} m/s")

print()
print(f"  前进位移 {base[-1, 0] - base[0, 0]:+.3f} m  (y {base[-1, 1] - base[0, 1]:+.3f} m)  "
      f"⇒ 均速 {(base[-1, 0] - base[0, 0]) / (T * dt):.3f} m/s   指令 {float(d['cmd_vx']):.2f}")
print(f"  base z 均值 {base[:, 2].mean():.3f}  范围 [{base[:, 2].min():.3f}, {base[:, 2].max():.3f}]")

# ---------------------------------------------------------------- ③ 前 1 秒
print()
print("③ 起转前 1 秒逐帧（真机第一次跑最容易在这暴露问题）")
print("-" * 96)
print(f"{'t(s)':>6}  " + "".join(f"{SHORT[n][:6]:>8}" for n in names))
for k in range(0, int(1.0 / dt) + 1, 5):
    print(f"{k * dt:>6.2f}  " + "".join(f"{q[k, i]:>+8.3f}" for i in range(len(names))))
print()
print("act（策略原始输出，前 1 秒）：")
print(f"{'t(s)':>6}  " + "".join(f"{SHORT[n][:6]:>8}" for n in names))
for k in range(0, int(1.0 / dt) + 1, 5):
    print(f"{k * dt:>6.2f}  " + "".join(f"{act[k, i]:>+8.3f}" for i in range(len(names))))

# ---------------------------------------------------------------- ④ 存一份可机读的判据
import json
out = {
    "run": str(d["run"]), "steps": T, "dt": dt, "cmd_vx": float(d["cmd_vx"]),
    "action_scale": float(d["action_scale"]),
    "joint_default": [float(x) for x in defq],
    "joints": {
        SHORT[nm]: {
            "joint_name": nm,
            "default": float(defq[i]),
            "steady_mean": rows[nm]["mean"],
            "amp_rad": rows[nm]["amp"],
            "amp_deg": rows[nm]["amp"] * 180 / np.pi,
            "freq_hz": rows[nm]["freq"],
            "phase_deg_vs_l1": rows[nm]["phase"],
        } for i, nm in enumerate(names)
    },
    "feet": {
        "left_lift_cm": float(feet[:, lb, 2].max() - np.percentile(feet[:, lb, 2], 1)) * 100,
        "right_lift_cm": float(feet[:, rb, 2].max() - np.percentile(feet[:, rb, 2], 1)) * 100,
    },
    "forward_m": float(base[-1, 0] - base[0, 0]),
    "mean_speed_mps": float((base[-1, 0] - base[0, 0]) / (T * dt)),
    "first_frame_act": [float(x) for x in act[0]],
    "first_frame_target": [float(x) for x in tgt[0]],
    "first_10_act": [[float(x) for x in row] for row in act[:10]],
}
p = D / "reference_criteria.json"
p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print()
print(f"✅ 判据已存: {p}")
