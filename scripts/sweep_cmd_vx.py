#!/usr/bin/env python3
"""扫 cmd_vx：找出哪个 vx 假设最能解释真机 /action。

为什么先做这个（计划 §3.1 第 4 条）：
  "真机幅度 = 仿真 3~8 倍" 这个结论是【在 cmd_vx=0.4 假设下】算出来的。
  如果真机实际 vx 是 0.5，那"幅度大"可能大部分只是【命令不同】的假象，
  而不是真正的异常。
  ⇒ 这一条最便宜，且能【证伪我自己昨晚的结论】。

做法：用真机 obs 序列离线递推，对每个候选 vx 算与真机 /action 的残差，
     取残差最小的 vx 作为真机实际命令的估计，并在此 vx 下重算幅度比。
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

D = Path("/home/zhan/robot-deploy-toolkit/results/dm10_real_run")
R = Path("/home/zhan/robot-deploy-toolkit/results/dm10_ref_run")
E = np.load(D / "real_run_export.npz", allow_pickle=True)
REF = np.load(R / "ref_trace.npz", allow_pickle=True)
# 板上那份 onnx 的本地副本。原为 /tmp/dm10_39dim_policy.onnx —— 重启就没了，
# 2026-09-21 移到备份目录。HANDOFF §8.6 已用【权重指纹】证明它 ≡
# logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-09_00-52-12_mujoco/policy.onnx。
# ⚠️ 这是【屈膝参考系】时代的存档，与当前仓库状态（直腿 home）不匹配，
#    只用于复核 2026-09 那次真机分析的结论，不能拿它跑新东西。
ONNX = "/home/zhan/dm10-home-bent-backup-20260921/policy_run/policy.onnx"

# ⚠️ FROZEN at the BENT-KNEE frame — do NOT "update" these to the straight-leg
# values. This script analyses the 2026-09 real-robot session, whose policy was
# trained bent-knee. On 2026-09-21 the project moved the sim home and the deploy
# `joint_default_angle` to straight-leg, shifting the action offset and obs zero
# point by a constant; re-pointing these numbers would silently corrupt every
# result below. Rollback: ~/dm10-home-bent-backup-20260921/README.md
DEFAULT = np.array([-0.4, 0, 0, 0.8, -0.4, -0.4, 0, 0, 0.8, -0.4])
SHORT = ["左髋p", "左髋r", "左髋y", "左膝", "左踝", "右髋p", "右髋r", "右髋y", "右膝", "右踝"]
SCALE = float(REF["action_scale"])

ta, tj, ti = E["action__t"], E["joint_states__t"], E["imu__t"]
A, J, JV = E["action__position"], E["joint_states__position"], E["joint_states__velocity"]
Q, W = E["imu__quat"], E["imu__angvel"]

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name


def nearest(tarr, t):
    return int(np.argmin(np.abs(tarr - t)))


def gravity_b(quat):
    w, x, y, z = quat
    Rm = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])
    return Rm.T @ np.array([0.0, 0.0, -1.0])


# ---- 找策略段（与 /action 偏离 default 的连续区间）
dev = np.abs(A - DEFAULT).max(axis=1)
moving = dev > 0.05
idx = np.flatnonzero(moving)
segs, s = [], idx[0]
for a_, b_ in zip(idx[:-1], idx[1:]):
    if b_ - a_ > 10:
        segs.append((s, a_)); s = b_
segs.append((s, idx[-1]))
print(f"策略段 {len(segs)} 个: {[(int(a),int(b)) for a,b in segs]}")


def replay(seg, cmd_vx, n=120, smooth_vel=0):
    """用真机 obs 递推 n 帧，返回 (帧号列表, 离线target, 真机action)。"""
    s0, _ = seg
    last, out_t, out_r = np.zeros(10), [], []
    for i in range(n):
        f = s0 + i
        if f >= len(ta):
            break
        ji, ii = nearest(tj, ta[f]), nearest(ti, ta[f])
        q, qd = J[ji], JV[ji].copy()
        if smooth_vel and i >= smooth_vel:
            qd = JV[ji - smooth_vel:ji + 1].mean(axis=0)
        g = gravity_b(Q[ii])
        obs = np.concatenate([W[ii], g, q - DEFAULT, qd, last, [cmd_vx, 0, 0]])
        o = sess.run([oname], {iname: obs.astype(np.float32)[None, :]})[0][0]
        out_t.append(o * SCALE + DEFAULT)
        out_r.append(A[f])
        last = o
    return np.array(out_t), np.array(out_r)


print()
print("=" * 80)
print("① 扫 cmd_vx：哪个假设最能解释真机 /action")
print("=" * 80)
print(f"  {'vx':>6}{'平均|差|':>12}{'最大|差|':>12}{'差(°)':>10}")
best = None
for vx in (0.30, 0.35, 0.40, 0.45, 0.50):
    tgt, real = replay(segs[0], vx)
    err = np.abs(tgt - real)
    m, mx = err.mean(), err.max()
    print(f"  {vx:>6.2f}{m:>12.4f}{mx:>12.4f}{m*57.3:>10.2f}")
    if best is None or m < best[1]:
        best = (vx, m)
print(f"\n  ⇒ 残差最小的 vx = {best[0]:.2f}  (平均|差| {best[1]:.4f} rad = {best[1]*57.3:.2f}°)")

print()
print("=" * 80)
print("② 噪声假设：joint_vel 平滑后再喂 onnx（真机速度可能是噪声）")
print("=" * 80)
print(f"  {'平滑窗口':>10}{'平均|差|':>12}{'差(°)':>10}")
for sm in (0, 2, 5, 10):
    tgt, real = replay(segs[0], best[0], smooth_vel=sm)
    err = np.abs(tgt - real)
    print(f"  {sm:>10}{err.mean():>12.4f}{err.mean()*57.3:>10.2f}")

print()
print("=" * 80)
print("③ gravity_b 零偏检查（静止段应接近 [0,0,-1]）")
print("=" * 80)
gs = np.array([gravity_b(Q[i]) for i in range(len(Q))])
# 找最静止的 100 帧（角速度最小）
sp = np.linalg.norm(W, axis=1)
calm = np.argsort(sp)[:100]
print(f"  最静止 100 帧的 gravity_b: 均值 {np.round(gs[calm].mean(axis=0), 4)}")
print(f"                             标准差 {np.round(gs[calm].std(axis=0), 4)}")
print(f"  理想值 [0, 0, -1] ⇒ 偏差 {np.round(gs[calm].mean(axis=0) - [0,0,-1], 4)}")
ang = np.degrees(np.arccos(np.clip(-gs[calm].mean(axis=0)[2], -1, 1)))
print(f"  ⇒ 等效倾角偏差 {ang:.2f}°")

print()
print("=" * 80)
print("④ 幅度比：在【最优 vx】下重算 真机 vs 仿真")
print("=" * 80)
s_all = slice(int(segs[0][0]), int(segs[-1][1]) + 1)
real_amp = A[s_all].max(axis=0) - A[s_all].min(axis=0)
# 仿真在同 vx 下的幅度：只有 0.4 那条参考跑，这里用比值说明 vx 的影响
rt = REF["target"]
sim_amp = rt.max(axis=0) - rt.min(axis=0)
print(f"  {'关节':<8}{'真机幅度':>12}{'仿真幅度(0.4)':>15}{'倍数':>8}")
for k in range(10):
    r = real_amp[k] / sim_amp[k] if sim_amp[k] > 1e-9 else float("nan")
    print(f"  {SHORT[k]:<8}{real_amp[k]:>12.3f}{sim_amp[k]:>15.3f}{r:>8.1f}")
