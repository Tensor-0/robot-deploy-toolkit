#!/usr/bin/env python3
"""direction_check.py — 关节方向验证（阶段 2）

**验证什么**：给一个小的**正向**目标，关节是否朝**预期方向**动。

为什么需要它：
  `motor_sign` 只在 RobotInterface 层生效，
  **motors_py 层读到的是原始硬件符号** —— 所以必须显式比对
  「命令符号」vs「反馈位移符号」。

用法:
  # 交互式：逐台给 +δ，人看方向对不对
  python3 direction_check.py --config <robot.yaml>

  # 自动判据：给 +δ 后位移应为正（配合 motor_sign 使用）
  python3 direction_check.py --config <robot.yaml> --auto --delta 0.05

⚠️ 机器人必须悬空/有支撑。幅度默认很小（0.05 rad）。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RED, YEL, GRN, RST, LOOP_DT, need_motors_py, load_yaml,
                     build_motor_specs, create_motors, safe_deinit,
                     read_error_clean, describe_error, confirm)


def main():
    ap = argparse.ArgumentParser(description="关节方向验证")
    ap.add_argument("--config", required=True)
    ap.add_argument("--delta", type=float, default=0.05, help="测试幅度 rad（默认 0.05，很小）")
    ap.add_argument("--auto", action="store_true",
                    help="自动判据：命令 +δ 后位移应为正（不询问）")
    ap.add_argument("--out", default="results/direction_check.json")
    args = ap.parse_args()

    motors_py = need_motors_py()
    cfg = load_yaml(args.config)
    specs = build_motor_specs(cfg)

    print("=" * 66)
    print(f" 关节方向验证  ·  ±{args.delta} rad")
    print("=" * 66)
    confirm("机器人悬空/有支撑？人站侧面？急停在手边？")

    motors = create_motors(motors_py, specs)
    results = []
    try:
        for s, mv in motors:
            mv.init_motor()
            time.sleep(0.3)
            mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
            time.sleep(0.1)

            mv.refresh_motor_status(); time.sleep(0.05)
            q0 = mv.get_motor_pos()

            # 平滑走到 q0 + delta（避免阶跃）
            target = q0 + args.delta
            n = max(1, int(0.8 / LOOP_DT))
            q = q0
            for k in range(n):
                q = q0 + (target - q0) * (k + 1) / n
                mv.motor_mit_cmd(q, 0.0, 0.0, 1.0, 0.0)
                time.sleep(LOOP_DT)
            time.sleep(0.3)
            mv.refresh_motor_status(); time.sleep(0.05)
            q1 = mv.get_motor_pos()

            dq = q1 - q0
            consistent = dq > 0     # 命令 +δ，位移应为正

            print(f"[{s['index']:2d}] can/{s['motor_id']:<3} "
                  f"q0={q0:+.4f} → q1={q1:+.4f}  Δ={dq:+.4f}  ", end="")
            if consistent:
                print(f"{GRN}方向一致 ✓{RST}")
            else:
                print(f"{RED}方向相反 ✗{RST}")

            if not args.auto:
                ans = input("     方向对吗？(y/n/回车跳过) ").strip().lower()
                if ans == "n":
                    consistent = False
                elif ans == "":
                    consistent = None

            results.append({
                "index": s["index"], "motor_id": s["motor_id"],
                "interface": s["interface"],
                "q0": round(q0, 4), "q1": round(q1, 4), "dq": round(dq, 4),
                "direction_ok": consistent,
                "error_id": read_error_clean(mv),
            })

            # 回原位
            q = q1
            for k in range(n):
                q = q1 + (q0 - q1) * (k + 1) / n
                mv.motor_mit_cmd(q, 0.0, 0.0, 1.0, 0.0)
                time.sleep(LOOP_DT)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\n{YEL}中断{RST}")
    finally:
        safe_deinit(motors)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bad = [r for r in results if r["direction_ok"] is False]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "delta": args.delta, "results": results,
                   "mismatched": [r["index"] for r in bad]},
                  f, ensure_ascii=False, indent=2)

    print()
    print("=" * 66)
    if bad:
        print(f"{RED}✗ {len(bad)} 个关节方向相反：{[r['index'] for r in bad]}{RST}")
        print("  修法：在配置的 motor_sign 里把对应项改成 -1")
    else:
        print(f"{GRN}✓ 方向检查通过（{len(results)} 个关节）{RST}")
    print(f" 已落盘: {args.out}")
    print("=" * 66)


if __name__ == "__main__":
    main()
