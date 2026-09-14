#!/usr/bin/env python3
"""gain_tune.py — kp/kd 阶跃响应整定（阶段 3）

**解决什么问题**：现有部署仓库**没有任何时序记录**（无时间戳、无采样缓存、无 CSV），
所以调 kp/kd 只能靠肉眼。这个脚本补上这块，把响应曲线采下来。

⚠️ 三个必知的坑（脚本会主动警告）：
  1. **增益被静默 clamp**：kp ∈ [0,500]、kd ∈ [0,5]，超出的被削平且**不报错**
  2. **kd = 0 会震荡失控**（说明书明确）
  3. **仿真 PD ≠ 真机 PD**：仿真 `<position>` 是隐式约束、真机用当前状态算
     → **别从仿真的 kp=30 起步**；从官方值起步（DM10 是 kp=16-20 / kd=3）

用法:
  # 单组：看一条响应曲线
  python3 gain_tune.py --config <robot.yaml> --joint 3 --kp 20 --kd 3

  # 扫 kp
  python3 gain_tune.py --config <robot.yaml> --joint 3 --sweep kp --values 10,16,20,30,40

  # 扫 kd（kp 固定）
  python3 gain_tune.py --config <robot.yaml> --joint 3 --sweep kd --values 0.5,1,3,5 --kp 20

输出:
  results/gain_tune_<joint>.json   完整采样序列（含每组试验的轨迹）
  results/gain_tune_<joint>.csv    便于直接画图

⚠️ 安全：幅度小（默认 0.1 rad）、检测错误码即停、finally 全部失能。
   机器人必须悬空/有支撑，人站侧面，急停在手边。
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RED, YEL, GRN, RST, LOOP_DT, need_motors_py, load_yaml,
                     build_motor_specs, create_motors, safe_deinit,
                     read_error_clean, describe_error, confirm)

# 硬件上限【实测 dm_motor_driver.cpp 的 OKpMax/OKdMax】
KP_MAX, KD_MAX = 500.0, 5.0
SAMPLE_DIV = 5          # 每 N 个控制周期采一次（200Hz → 40Hz 采样）


def run_step(mv, q0, amp, kp, kd, duration, abort_on_error=True):
    """做一次阶跃并采回完整时序。

    返回 (traj, err)
      traj = [(t, q_cmd, q_act, vel, tau), ...]
      err  = 期间检测到的最高错误码（0/1 为正常）
    """
    traj, err_peak = [], 0
    t0 = time.time()
    n = int(duration / LOOP_DT)
    target = q0 + amp
    for k in range(n):
        mv.motor_mit_cmd(target, 0.0, kp, kd, 0.0)
        time.sleep(LOOP_DT)
        if k % SAMPLE_DIV == 0:
            mv.refresh_motor_status()
            traj.append((
                round(time.time() - t0, 4),
                round(target, 5),
                round(mv.get_motor_pos(), 5),
                round(mv.get_motor_spd(), 5),
                round(mv.get_motor_current(), 5),
            ))
            e = mv.get_error_id()
            if isinstance(e, (int, float)) and e >= 8:
                err_peak = max(err_peak, int(e))
                if abort_on_error:
                    break
    return traj, err_peak


def analyze(traj, q0, amp):
    """从时序算响应指标"""
    if not traj:
        return {}
    qs = [p[2] for p in traj]
    target = q0 + amp
    peak = max(qs) if amp > 0 else min(qs)
    overshoot = abs(peak - target) / abs(amp) * 100 if amp else 0.0

    # 上升时间：首次达到目标 90%
    rise_t, th = None, q0 + 0.9 * amp
    for t, _, q, _, _ in traj:
        if (amp > 0 and q >= th) or (amp < 0 and q <= th):
            rise_t = t
            break

    # 稳态误差 + 纹波（取最后 20%）
    tail = qs[-max(1, len(qs) // 5):]
    ss_err = sum(abs(q - target) for q in tail) / len(tail) if tail else 0.0
    ripple = (max(tail) - min(tail)) if tail else 0.0

    return {
        "overshoot_pct": round(overshoot, 1),
        "rise_time_s": rise_t,
        "steady_state_err": round(ss_err, 4),
        "ripple": round(ripple, 4),
        "peak": round(peak, 5),
        "settled_q": round(tail[-1], 5) if tail else None,
    }


def main():
    ap = argparse.ArgumentParser(description="kp/kd 阶跃响应整定")
    ap.add_argument("--config", required=True)
    ap.add_argument("--joint", type=int, required=True, help="关节索引（配置顺序）")
    ap.add_argument("--sweep", choices=["kp", "kd", "none"], default="none")
    ap.add_argument("--values", help="扫的参数值，逗号分隔")
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=3.0)
    ap.add_argument("--amp", type=float, default=0.1, help="阶跃幅度 rad（默认 0.1）")
    ap.add_argument("--duration", type=float, default=1.5, help="每次阶跃时长 s")
    ap.add_argument("--out", help="输出前缀（默认 results/gain_tune_<joint>）")
    ap.add_argument("--yes", action="store_true", help="跳过确认（危险）")
    args = ap.parse_args()

    # ── 先做纯参数校验（不需要硬件，所以放在依赖检查之前）──
    warn = []
    if args.kp > KP_MAX:
        warn.append(f"kp={args.kp} 超上限 {KP_MAX}，会被**静默削平**")
    if args.kd > KD_MAX:
        warn.append(f"kd={args.kd} 超上限 {KD_MAX}，会被**静默削平**")
    if args.kd == 0:
        warn.append("kd=0 —— 说明书明确：**会震荡失控**")

    # 构造试验矩阵（也不需要硬件）
    if args.sweep == "kp":
        vals = [float(x) for x in (args.values or "10,16,20,30,40").split(",")]
        trials = [(v, args.kd) for v in vals]
    elif args.sweep == "kd":
        vals = [float(x) for x in (args.values or "0.5,1,3,5").split(",")]
        trials = [(args.kp, v) for v in vals]
    else:
        trials = [(args.kp, args.kd)]

    print("=" * 70)
    print(f" kp/kd 整定  ·  关节 {args.joint}  ·  阶跃 ±{args.amp} rad  ·  {args.duration}s")
    print(f" 共 {len(trials)} 组: " + "  ".join(f"(kp={k},kd={d})" for k, d in trials))
    print("=" * 70)
    for w in warn:
        print(f"{RED}⚠️  {w}{RST}")
    if warn:
        print()

    # ── 到这里才需要硬件 ──────────────────────────
    motors_py = need_motors_py()
    cfg = load_yaml(args.config)
    specs = [s for s in build_motor_specs(cfg) if s["index"] == args.joint]
    if not specs:
        sys.exit(f"{RED}找不到关节索引 {args.joint}（配置里共 "
                 f"{len(build_motor_specs(cfg))} 个）{RST}")

    if not args.yes:
        confirm("机器人悬空/有支撑？人站侧面？急停在手边？")

    out = args.out or f"results/gain_tune_{args.joint}"
    motors = create_motors(motors_py, specs)
    s, mv = motors[0]
    all_json, all_csv = [], []

    try:
        mv.init_motor()
        time.sleep(0.3)
        mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
        time.sleep(0.1)

        for i, (kp, kd) in enumerate(trials, 1):
            mv.refresh_motor_status(); time.sleep(0.05)
            q0 = mv.get_motor_pos()

            # 先稳定住
            for _ in range(int(0.3 / LOOP_DT)):
                mv.motor_mit_cmd(q0, 0.0, kp, kd, 0.0)
                time.sleep(LOOP_DT)

            print(f"[{i}/{len(trials)}] kp={kp} kd={kd}  "
                  f"q0={q0:+.4f} → {q0+args.amp:+.4f} ...", end=" ", flush=True)
            traj, err = run_step(mv, q0, args.amp, kp, kd, args.duration)
            m = analyze(traj, q0, args.amp)

            ok = err in (0, 1)
            print(f"{GRN}✓{RST}" if ok else f"{RED}err={err} {describe_error(err)}{RST}")
            print(f"        超调 {m.get('overshoot_pct')}%  "
                  f"上升 {m.get('rise_time_s')}s  "
                  f"稳差 {m.get('steady_state_err')}  "
                  f"纹波 {m.get('ripple')}")

            all_json.append({"kp": kp, "kd": kd, "q0": round(q0, 5), **m,
                             "error_id": err, "n_samples": len(traj),
                             "traj": traj})
            for (t, qc, qa, v, tau) in traj:
                all_csv.append([args.joint, kp, kd, t, qc, qa, v, tau])

            if not ok:
                print(f"{RED}      检测到错误码，中止后续试验{RST}")
                break

            # 回原位
            for _ in range(int(1.0 / LOOP_DT)):
                mv.motor_mit_cmd(q0, 0.0, kp, kd, 0.0)
                time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print(f"\n{YEL}中断{RST}")
    finally:
        safe_deinit(motors)

    # ── 落盘 ────────────────────────────────────────
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": os.path.abspath(args.config),
            "joint": args.joint, "amp": args.amp, "duration": args.duration,
            "limits": {"kp_max": KP_MAX, "kd_max": KD_MAX},
            "warnings": warn,
            "trials": all_json,
        }, f, ensure_ascii=False, indent=2)

    with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["joint", "kp", "kd", "t_s", "q_cmd", "q_act", "vel", "tau"])
        w.writerows(all_csv)

    print()
    print("=" * 70)
    print(" 汇总")
    print(f" {'kp':>7} {'kd':>6} | {'超调%':>7} {'上升s':>8} {'稳态误差':>10} {'纹波':>9}")
    print("-" * 70)
    for t in all_json:
        rt = t.get("rise_time_s")
        print(f" {t['kp']:>7} {t['kd']:>6} | {t.get('overshoot_pct', 0):>7} "
              f"{(f'{rt:.3f}' if rt is not None else '—'):>8} "
              f"{t.get('steady_state_err', 0):>10} {t.get('ripple', 0):>9}")
    print("=" * 70)
    print(f" 已落盘: {out}.json  /  {out}.csv")
    print()
    print(f"{YEL}判读提示（怎么看这张表）：{RST}")
    print("  * 超调 > 20%      → kp 偏高")
    print("  * 纹波大 + 嗡嗡声 → kd 偏低（但**别设 0**）")
    print("  * 上升慢、反应迟钝 → kd 偏高，或 kp 偏低")
    print("  * 稳态误差大      → 前馈/重力补偿不足，**不是 kp 的问题**")
    print()
    print(f"{YEL}⚠️  单关节整定的结果在整机上未必适用{RST}（负载变了）。")
    print("    这一步的目的是「确认关节健康 + 找到大致区间」，不是「找到最优值」。")
    print("    真正的 kp/kd 整定，是在整机能站住之后、对着实际负载调的。")


if __name__ == "__main__":
    main()
