#!/usr/bin/env python3
"""⭐ 决定性对照：真机 obs → 同一个 onnx → 和真机 /action 比

这是判断「软件链路对不对」的最强手段：
  · 离线算出的 act == 真机 /action  ⇒ 推理链路 100% 正确，问题在物理侧
  · 不等                          ⇒ 软件有问题，能看出错在哪一段

obs 定序（已核实 = 训练侧）:
  ang_vel:3 | gravity_b:3 | dof_pos:10 | dof_vel:10 | last_action:10 | cmd_vel:3
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

D = Path("/home/zhan/robot-deploy-toolkit/results/dm10_real_run")
E = np.load(D / "real_run_export.npz", allow_pickle=True)
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

ta = E["action__t"]
tj = E["joint_states__t"]
ti = E["imu__t"]
A = E["action__position"]
J = E["joint_states__position"]
JV = E["joint_states__velocity"]
Q = E["imu__quat"]
W = E["imu__angvel"]

print("=" * 90)
print("① 数据概览")
print("=" * 90)
print(f"/action      {A.shape}   起始 {ta[0]/1e9:.3f}")
print(f"/joint_states {J.shape}  起始 {tj[0]/1e9:.3f}")
print(f"/imu         {Q.shape}  起始 {ti[0]/1e9:.3f}")
print(f"跨度 {(ta[-1]-ta[0])/1e9:.1f}s   /action {len(ta)/((ta[-1]-ta[0])/1e9):.1f} Hz")
print(f"/joint_states 频率 {len(tj)/((tj[-1]-tj[0])/1e9):.1f} Hz")

# ---------------------------------------------------------------- 分段
dev = np.abs(A - DEFAULT).max(axis=1)
moving = dev > 0.05
print(f"\n策略在动帧数 {int(moving.sum())}/{len(A)}")
idx = np.flatnonzero(moving)
segs = []
if len(idx):
    s = idx[0]
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if b_ - a_ > 10:
            segs.append((s, a_)); s = b_
    segs.append((s, idx[-1]))
print(f"策略段 {len(segs)} 个：")
for s, e in segs:
    print(f"  帧 {s}-{e}  ({(ta[e]-ta[s])/1e9:.1f}s)")

# ---------------------------------------------------------------- ② 关节角对照
print()
print("=" * 90)
print("② 真机实测关节角 vs 下发的目标角（看电机跟不跟得上）")
print("=" * 90)
for si, (s, e) in enumerate(segs):
    sl = slice(s, e + 1)
    err = np.abs(J[sl] - A[sl])
    print(f"段{si+1}  帧{s}-{e}  ({(ta[e]-ta[s])/1e9:.1f}s)")
    print(f"  {'关节':<8}{'实测均值':>12}{'目标均值':>12}{'平均|误差|':>12}{'最大|误差|':>12}{'误差°':>10}")
    for k in range(10):
        print(f"  {SHORT[k]:<8}{J[sl,k].mean():>12.4f}{A[sl,k].mean():>12.4f}"
              f"{err[:,k].mean():>12.4f}{err[:,k].max():>12.4f}{err[:,k].max()*57.3:>10.1f}")
    print(f"  整体: 平均|误差| {err.mean():.4f} rad ({err.mean()*57.3:.1f}°)  "
          f"最大 {err.max():.4f} rad ({err.max()*57.3:.1f}°)")

# ---------------------------------------------------------------- ③ 离线复现 onnx
print()
print("=" * 90)
print("③ 真机 obs → onnx → 对比真机 /action  （决定性）")
print("=" * 90)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name


def nearest(tarr, t):
    return int(np.argmin(np.abs(tarr - t)))


for si, (s, e) in enumerate(segs):
    print()
    print(f"── 段{si+1} (帧 {s}..{e}) ──")
    # 逐帧复现（用真机自己的 last_action 递推，与节点一致）
    last_act = np.zeros(10)
    n = min(e - s + 1, 40)
    errs = []
    rows = []
    for i in range(n):
        f = s + i
        t = ta[f]
        ji = nearest(tj, t)
        ii = nearest(ti, t)
        q = J[ji]
        qd = JV[ji]
        w, x, y, z = Q[ii]
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
        ])
        g_b = R.T @ np.array([0., 0., -1.])
        # 真机的 cmd_vel 未知（摇杆推的），用 clip 上限区间的中值试
        for cmd in (0.4, 0.5, 0.3):
            obs = np.concatenate([W[ii], g_b, q - DEFAULT, qd, last_act, [cmd, 0, 0]])
            out = sess.run([oname], {iname: obs.astype(np.float32)[None, :]})[0][0]
            if cmd == 0.4:
                tgt = out * 0.25 + DEFAULT
                errs.append(np.abs(tgt - A[f]))
                rows.append((f, out.copy(), tgt.copy(), A[f].copy()))
        last_act = out   # 用离线输出递推（与部署一致：last_action 存的是【原始输出】）

    errs = np.array(errs)
    print(f"  cmd_vx=0.4 假设下，离线 target vs 真机 /action：")
    for f, out, tgt, real in rows[:2]:
        print(f"\n  帧 {f}:")
        print(f"    离线 act     {np.array2string(out, precision=3, suppress_small=True)}")
        print(f"    离线 target  {np.array2string(tgt, precision=3, suppress_small=True)}")
        print(f"    真机 /action {np.array2string(real, precision=3, suppress_small=True)}")
        print(f"    差           {np.array2string(tgt-real, precision=3, suppress_small=True)}")
    print(f"\n  前 {len(rows)} 帧整体: 平均|差| {errs.mean():.4f} rad "
          f"({errs.mean()*57.3:.1f}°)  最大 {errs.max():.4f} ({errs.max()*57.3:.1f}°)")

# ---------------------------------------------------------------- ④ 真机 vs 仿真
print()
print("=" * 90)
print("④ 真机动作幅度 vs 仿真（同一条策略）")
print("=" * 90)
ref = np.load("/home/zhan/robot-deploy-toolkit/results/dm10_ref_run/ref_trace.npz",
              allow_pickle=True)
rt = ref["target"]
print(f"  {'关节':<8}{'真机幅度':>12}{'仿真幅度':>12}{'倍数':>8}{'真机峰值|Δ|':>14}{'仿真峰值|Δ|':>14}")
for k in range(10):
    ra = A[segs[0][0]:segs[-1][1]+1, k].max() - A[segs[0][0]:segs[-1][1]+1, k].min()
    sa = rt[:, k].max() - rt[:, k].min()
    print(f"  {SHORT[k]:<8}{ra:>12.3f}{sa:>12.3f}{ra/sa if sa>1e-6 else 0:>8.1f}"
          f"{np.abs(A[segs[0][0]:segs[-1][1]+1,k]-DEFAULT[k]).max():>14.3f}"
          f"{np.abs(rt[:,k]-DEFAULT[k]).max():>14.3f}")
