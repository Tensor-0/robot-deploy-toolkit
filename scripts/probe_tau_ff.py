#!/usr/bin/env python3
"""probe_tau_ff.py — 判定 MIT 的 tau_ff 到底生不生效（+ 顺带量静摩擦带）。

背景
----
前馈验收失败：同一姿态两次量到的静态负载差了 ~1 N·m 甚至反号。
两个候选解释，本脚本一次判掉：

  H1  固件【根本没使用】tau_ff     ⇒ 给大 τ_ff，关节不动
  H2  单次测量被【静摩擦】污染      ⇒ 同一姿态、不同历史，读数差 2F

做法（全部保持同一个目标姿态 = 默认位）
----------------------------------------
    q0    τ_ff = 0     （从默认位直接来）
    q+    τ_ff = +A
    q-    τ_ff = -A
    q0'   τ_ff = 0     （回到 0，但历史已经不同）

判读
----
  · 若 |Δq+| ≈ |Δq-| ≈ A/Kp 且反号  ⇒ tau_ff 【生效】        (H1 否)
  · 若 Δq+ ≈ Δq- ≈ 0                ⇒ tau_ff 【被忽略】      (H1 是)
  · |q0' − q0| × Kp                 ⇒ 静摩擦带 2F 的估计     (H2)

用法:
    python3 scripts/probe_tau_ff.py --config <robot.yaml> --infer-config <default.yaml>
    python3 scripts/probe_tau_ff.py ... --joint knee --leg left --tau 3.0

⚠️ 会用 A N·m 推关节 —— 悬吊下安全，但别站在腿的活动范围里。
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    GRN, RED, RST, YEL,
    build_motor_specs, create_motors, load_yaml, need_motors_py, safe_deinit,
)

FRAME_HZ = 100.0
SAMPLE_HZ = 50.0
WINDOW = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--infer-config", required=True)
    ap.add_argument("--joint", default="knee")
    ap.add_argument("--leg", default="left", choices=["left", "right"])
    ap.add_argument("--tau", type=float, default=3.0, help="探针力矩幅值（N·m）")
    ap.add_argument("--settle", type=float, default=5.0)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    motors_py = need_motors_py()
    specs = build_motor_specs(cfg)
    inf = load_yaml(args.infer_config)
    inf = inf.get("inference_node", {}).get("ros__parameters", inf)

    kp, kd = list(cfg["robot"]["kp"]), list(cfg["robot"]["kd"])
    sign = list(cfg["robot"]["motor_sign"])
    q_def = list(inf["joint_default_angle"])
    n = len(specs)

    NAMES = ["hip_p", "hip_r", "hip_y", "knee", "ankle"]
    j = (0 if args.leg == "left" else 5) + NAMES.index(args.joint)
    jn = f"{'l' if j < 5 else 'r'}{args.joint}"
    print(f"探针关节: {jn} (index {j})  默认位 {q_def[j]:+.3f}  Kp={kp[j]:.1f}")
    print(f"预期位移 |Δq| = A/Kp = {args.tau}/{kp[j]:.1f} = {args.tau/kp[j]:.4f} rad"
          f" = {math.degrees(args.tau/kp[j]):.1f}°\n")

    motors = create_motors(motors_py, specs)
    for _, mv in motors:
        mv.init_motor()
    time.sleep(0.5)
    print(f"{GRN}✓ {len(motors)} 台电机已使能{RST}\n")

    target = list(q_def)
    cur_ff = 0.0

    def send_once():
        for i, (_, mv) in enumerate(motors):
            ff = cur_ff if i == j else 0.0
            mv.motor_mit_cmd(target[i] * sign[i], 0.0, kp[i], kd[i], ff * sign[i])

    def settle_and_measure(tag):
        for _ in range(int(2.0 * FRAME_HZ)):
            send_once(); time.sleep(1.0 / FRAME_HZ)
        time.sleep(args.settle)
        qs = []
        for _ in range(int(WINDOW * SAMPLE_HZ)):
            send_once(); time.sleep(1.0 / SAMPLE_HZ)
            qs.append(motors[j][1].get_motor_pos() * sign[j])
        avg = sum(qs) / len(qs)
        print(f"  {tag:<22} q = {avg:+.5f} rad   ({math.degrees(avg):+7.3f}°)")
        return avg

    try:
        print("=========== 探针 ===========")
        cur_ff = 0.0
        q0 = settle_and_measure("q0   (τ_ff = 0)")
        cur_ff = +args.tau
        qp = settle_and_measure(f"q+   (τ_ff = +{args.tau})")
        cur_ff = -args.tau
        qm = settle_and_measure(f"q-   (τ_ff = -{args.tau})")
        cur_ff = 0.0
        q0b = settle_and_measure("q0'  (τ_ff = 0，历史已变)")

        dp, dm, d0 = qp - q0, qm - q0, q0b - q0
        exp = args.tau / kp[j]
        print(f"\n=========== 判读 ===========")
        print(f"  Δq+  = {dp:+.5f} rad  ({math.degrees(dp):+7.3f}°)")
        print(f"  Δq-  = {dm:+.5f} rad  ({math.degrees(dm):+7.3f}°)")
        print(f"  若 τ_ff 生效，两者应≈ {exp:+.4f} 与 {-exp:+.4f}（反号、等大）")
        print()
        if abs(dp) < 0.2 * exp and abs(dm) < 0.2 * exp:
            print(f"  {RED}⇒ τ_ff 【被忽略】—— 位移远小于预期。H1 成立。{RST}")
        elif dp * dm < 0 and abs(abs(dp) - abs(dm)) < 0.5 * exp:
            print(f"  {GRN}⇒ τ_ff 【生效】—— 反号且量级对得上。H1 否。{RST}")
        else:
            print(f"  {YEL}⇒ 不确定 —— 既不像完全生效也不像完全没用，看上面的数{RST}")

        print(f"\n  摩擦带估计：")
        print(f"    q0' − q0 = {d0:+.5f} rad  ⇒  Kp×|Δ| = {kp[j]*abs(d0):.4f} N·m"
              f"  （≈ 2F，同一个 τ_ff=0 姿态、只是历史不同）")
    except KeyboardInterrupt:
        print(f"\n{YEL}⚠️  中断{RST}")
    finally:
        cur_ff = 0.0
        try:
            for jj in range(n):
                target[jj] = q_def[jj]
            for _ in range(int(2.0 * FRAME_HZ)):
                send_once(); time.sleep(1.0 / FRAME_HZ)
        except Exception:
            pass
        safe_deinit(motors)
        print(f"{GRN}✓ 已回默认位并失能{RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
