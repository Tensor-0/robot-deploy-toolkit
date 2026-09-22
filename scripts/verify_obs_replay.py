#!/usr/bin/env python3
"""A3 验收：用 bag 里的原始话题离线重建 obs，与 bag 里 `/policy_obs` 逐帧比。

判据：每个分量 |diff| < 1e-6（float32 往返的量级）。

**为什么需要它**：obs 是"策略真正吃进去的东西"，但它原本哪儿都没落盘
⇒ 只能隔着 `/action` 比，而 `/action` 又隔着 act_alpha 平滑和 250/50 Hz 采样差，
永远比不到 1e-6。节点现在把【喂给 onnx 的那份输入】原样发到 `/policy_obs`，
于是"原始话题 → 重建 obs → 逐位比"这条路可以直接验收。

**重建参数全部取自 bag 里的 `/node_metadata`**（obs 定序、默认角、各 scale、
clip_cmd、clip_observations）—— 不写死；写死的话脚本自己就成了不一致源，
而它要验的恰恰是"两边一致"。

⚠️ **反序列化在板上做、不在本地**：bag 里的消息交给 ROS 自己解（`dump_bag.py`），
   本地只做数值比较。手写 CDR 这条路上我错了三次 —— 那份数据的字符串长度字段
   把 padding 也算进去（关节名解析出来带尾部 NUL），逐字段推 offset 必定错位。

用法:
    # 板上（有 ROS）：
    python3 dump_bag.py <bag_dir> /tmp/a3_dump.npz
    # 本地：
    python3 scripts/verify_obs_replay.py --npz /tmp/a3_dump.npz
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

TOL = 1e-6


def scale_of(meta, key):
    """⚠️ 缺了就【大声】退回 1.0 —— 不静默。缺说明录这份 bag 的节点版本旧。"""
    if key in meta:
        return float(meta[key])
    print("⚠️ metadata 缺 %s ⇒ 按 1.0 处理（这份 bag 的节点版本较旧）" % key)
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="板上 dump_bag.py 导出的 npz")
    ap.add_argument("--onnx", help="可选：给 chain 检验用（要和 metadata 里的权重指纹一致）")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)

    meta = json.loads(str(d["meta_json"][0]))
    print("metadata: git_sha=%s  onnx_fingerprint=%s  robot_config_fnv=%s"
          % (meta.get("git_sha"), meta.get("onnx_weight_fingerprint"),
             meta.get("robot_config_fnv1a64")))
    for ev in d["mode_json"]:
        ev = json.loads(str(ev))
        print("  事件 %s -> %s (%s) %s" % (ev.get("prev"), ev.get("mode"),
                                          ev.get("reason"), ev.get("detail", "")))

    fields, off = [], 0
    for item in meta["obs_layouts"][0].split(","):
        name, _, size = item.strip().partition(":")
        fields.append((name, int(size), off))
        off += int(size)
    obs_dim = off
    print("obs 定序:", ", ".join("%s:%d" % (n, s) for n, s, _ in fields), "共 %d 维" % obs_dim)

    default = np.array(meta["joint_default_angle"], dtype=np.float64)
    s_av = scale_of(meta, "obs_scales_ang_vel")
    s_dp = scale_of(meta, "obs_scales_dof_pos")
    s_dv = scale_of(meta, "obs_scales_dof_vel")
    s_gb = scale_of(meta, "obs_scales_gravity_b")
    s_lv = scale_of(meta, "obs_scales_lin_vel")
    clip_obs = float(meta["clip_observations"])
    clip_cmd = list(meta["clip_cmd"])

    T_obs, O_obs = d["obs_t"], d["obs_1"]
    T_imu, IMU_q, IMU_av = d["imu_t"], d["imu_1"], d["imu_2"]
    T_js, JS_q, JS_v = d["js_t"], d["js_1"], d["js_2"]
    T_joy, JOY_ax = d["joy_t"], d["joy_1"]
    print("帧数: /policy_obs %d  /imu %d  /joint_states %d  /joy %d"
          % (len(T_obs), len(T_imu), len(T_js), len(T_joy)))
    if len(O_obs) == 0 or O_obs.shape[1] != obs_dim:
        sys.exit("obs 维度对不上：%s vs %d" % (O_obs.shape, obs_dim))

    # ⚠️ 对齐方式：按【序号】而不是"时间上最近的一条"。
    #   三个话题都是同一个控制循环在同一拍里发的（imu/joint_states/obs 各一帧），
    #   但走 DDS 到达顺序可能交错 ⇒ 按时间取"最近"会串到上一拍，
    #   整体差出一拍的数据（实测 ang_vel 差 5.8、dof_vel 差 5.4）。
    #   帧数相同（945/945/945）说明没有丢帧 ⇒ 序号一一对应。
    #   下游用时间戳做一次抽检，串拍了要能看出来。
    if not (len(T_obs) == len(T_imu) == len(T_js)):
        sys.exit("帧数不等（%d/%d/%d）⇒ 有丢帧，序号对齐不成立，先查录制"
                 % (len(T_obs), len(T_imu), len(T_js)))
    dt_imu = np.abs(T_obs - T_imu) / 1e6
    dt_js = np.abs(T_obs - T_js) / 1e6
    print("同拍时间差（ms）: /imu 中位 %.2f 最大 %.2f   /joint_states 中位 %.2f 最大 %.2f"
          % (np.median(dt_imu), dt_imu.max(), np.median(dt_js), dt_js.max()))
    if dt_imu.max() > 5.0 or dt_js.max() > 5.0:
        print("⚠️ 有帧的时间差 > 5 ms —— 可能串拍，下面的结论要存疑")

    worst = np.zeros(obs_dim)
    all_diff = {}
    per_field = {n: 0.0 for n, _, _ in fields}
    n_checked = 0
    for k in range(1, len(T_obs)):
        i = j = k                      # 按序号对齐（见上）
        y = int(np.searchsorted(T_joy, T_obs[k], side="right") - 1)   # 手柄流 100 Hz，取最近
        q = JS_q[j]
        if len(q) != 10:
            continue
        w, x, yy, z = IMU_q[i]
        R = np.array([
            [1 - 2 * (yy * yy + z * z), 2 * (x * yy - z * w), 2 * (x * z + yy * w)],
            [2 * (x * yy + z * w), 1 - 2 * (x * x + z * z), 2 * (yy * z - x * w)],
            [2 * (x * z - yy * w), 2 * (yy * z + x * w), 1 - 2 * (x * x + yy * yy)],
        ])
        g_b = R.T @ np.array([0.0, 0.0, -1.0])

        axes = JOY_ax[y] if len(JOY_ax[y]) >= 6 else np.zeros(6)
        cv = np.empty(3)
        cv[0] = np.clip(axes[4] * clip_cmd[1], clip_cmd[0], clip_cmd[1])
        cv[1] = np.clip(axes[3] * clip_cmd[3], clip_cmd[2], clip_cmd[3])
        cv[2] = (np.clip(-axes[2] * clip_cmd[5], clip_cmd[4], clip_cmd[5]) if axes[2] < 0 else
                 np.clip(axes[5] * clip_cmd[5], clip_cmd[4], clip_cmd[5]) if axes[5] < 0 else 0.0)

        rebuilt = np.empty(obs_dim)
        for name, size, o0 in fields:
            if name == "ang_vel":
                rebuilt[o0:o0 + size] = IMU_av[i] * s_av
            elif name == "gravity_b":
                rebuilt[o0:o0 + size] = g_b * s_gb
            elif name == "dof_pos":
                rebuilt[o0:o0 + size] = (q - default) * s_dp
            elif name == "dof_vel":
                rebuilt[o0:o0 + size] = JS_v[j] * s_dv
            elif name == "cmd_vel":
                rebuilt[o0:o0 + size] = np.array([cv[0] * s_lv, cv[1] * s_lv, cv[2] * s_av])
            elif name == "last_action":
                # 递归段（= 上一帧的 onnx 输出）：无法从原始话题推出，
                # 期望值取上一帧 obs 的同一段。其余各段都是独立重建。
                rebuilt[o0:o0 + size] = O_obs[k - 1][o0:o0 + size]
            else:
                rebuilt[o0:o0 + size] = O_obs[k][o0:o0 + size]
        rebuilt = np.clip(rebuilt, -clip_obs, clip_obs)

        diff = np.abs(rebuilt - O_obs[k].astype(np.float64))
        worst = np.maximum(worst, diff)
        for name, size, o0 in fields:
            per_field[name] = max(per_field[name], float(diff[o0:o0 + size].max()))
            all_diff.setdefault(name, []).append(float(diff[o0:o0 + size].max()))
        n_checked += 1

    print()
    print("对比帧数 %d" % n_checked)
    for name, size, _ in fields:
        a = np.array(all_diff.get(name, [0.0]))
        print("  %-12s %-3d 最大 %.3e  中位 %.2e  p95 %.2e  <1e-6 的帧占比 %.1f%%%s"
              % (name, size, per_field[name], np.median(a), np.percentile(a, 95),
                 100.0 * float((a < TOL).mean()),
                 "   ← 递归段，非独立" if name == "last_action" else ""))
    # 总判定只看【独立重建的段】；last_action 是递归段，由下面的链式检验单独负责
    indep = np.concatenate([worst[o0:o0 + sz] for n, sz, o0 in fields if n != "last_action"])
    mx = float(indep.max())
    print()
    print("独立重建段（除 last_action）最大 |diff| = %.3e（判据 %.0e）" % (mx, TOL))
    if mx >= TOL:
        print("  ⇒ FAIL ✗")

    # ---- last_action 的单独检验 ----
    # 它是【递归段】（= 上一拍的 onnx 输出），无法从原始话题推出，
    # 所以不能用上面的"原始话题 → obs"那条路。正确的链是：
    #     onnx(obs[k])  ==  obs[k+1] 的 last_action 段
    # 因为 /policy_obs 发的就是喂给 onnx 的那份输入。
    all_pass = mx < TOL
    la = [o0 for n, sz, o0 in fields if n == "last_action"]
    if la and args.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        outname = sess.get_outputs()[0].name
        o0 = la[0]
        size = [sz for n, sz, _ in fields if n == "last_action"][0]
        worst = 0.0
        n_ok = 0
        for k in range(len(O_obs) - 1):
            out = sess.run([outname], {iname: O_obs[k][None, :].astype(np.float32)})[0][0]
            d = float(np.abs(out - O_obs[k + 1][o0:o0 + size].astype(np.float64)).max())
            worst = max(worst, d)
            n_ok += int(d < TOL)
        print()
        print("last_action 链式检验（onnx(obs[k]) vs obs[k+1]）:")
        print("  帧数 %d  <1e-6 占比 %.1f%%  最大 |diff| %.3e  ⇒ %s"
              % (len(O_obs) - 1, 100.0 * n_ok / (len(O_obs) - 1), worst,
                 "PASS ✓" if worst < TOL else "FAIL ✗"))
        all_pass = all_pass and worst < TOL
    elif la:
        print()
        print("⚠️ 未给 --onnx ⇒ last_action 段（递归）没验 ⇒ 总判定不算 PASS")
        all_pass = False
    print()
    print("A3 验收（obs 逐帧一致 < 1e-6）⇒ %s" % ("PASS ✓" if all_pass else "FAIL ✗"))
    sys.exit(0 if all_pass else 1)


main()
