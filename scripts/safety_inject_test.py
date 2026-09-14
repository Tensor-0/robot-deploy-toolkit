#!/usr/bin/env python3
"""safety_inject_test.py — 安全层注入测试（上机前最后一道自检）

**为什么需要它**：
现有的安全机制看起来都实现了，但**没人验证过它们真的会触发**。
这个脚本主动注入"坏数据"，验证保护是否生效。

【官方】UniLab 部署文档的「安全层隔离测试」三条：
  1. 注入一个 **NaN 动作**，验证该指令被拒绝
  2. 注入一个 **超范围的关节目标**，验证钳制生效
  3. 在运行途中 **切断策略输入**，验证进入安全状态

⚠️ 两种模式：
  --dry-run  （默认）**不碰硬件**，只检查代码里安全机制的存在性与阈值配置
  --live            真的注入（**需要硬件，风险高**）

**建议先跑 --dry-run**，它会告诉你「你的部署栈里哪些保护是缺的」。

用法:
  python3 safety_inject_test.py --config <robot.yaml>                    # 只做静态检查
  python3 safety_inject_test.py --config <robot.yaml> --deploy <policy.yaml>
  python3 safety_inject_test.py --config <robot.yaml> --live --joint 0   # 真注入（危险）
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RED, YEL, GRN, RST, LOOP_DT, load_yaml,
                     build_motor_specs, describe_error, confirm)

# ⚠️ 这些是「部署栈里应当存在」的安全机制
# 每项: (名称, 怎么查, 查什么, 官方依据)
SAFETY_CHECKS = [
    ("NaN 拒绝", "reject_nan",
     "新动作含 NaN/Inf 时应拒绝或替换"),
    ("关节限位钳制", "clamp_joint",
     "下发目标应被钳制到 joint_limits 内"),
    ("Δ 钳制", "clamp_delta",
     "单步动作增量应有上限"),
    ("速率限制", "rate_limit",
     "钳制后应施加变化率限制"),
    ("看门狗", "watchdog",
     "超时无新动作 → 保持安全目标或进安全状态"),
    ("姿态监控", "attitude",
     "roll/pitch 超范围应触发故障"),
    ("操作员停止", "estop",
     "硬件/软件急停应立即切断力矩"),
]


def analyze_deploy_config(path):
    """静态分析：部署配置里有哪些安全机制"""
    txt = open(path, encoding="utf-8").read()
    found = {}

    def has(pattern):
        return bool(re.search(pattern, txt))

    found["clip_actions"] = has(r"clip_actions")
    found["clip_observations"] = has(r"clip_observations")
    found["action_scale"] = has(r"action_scale")
    found["joint_limits"] = has(r"joint_limits")
    found["gravity_z_upper"] = has(r"gravity_z_upper")
    found["act_alpha"] = has(r"act_alpha")
    found["clip_cmd"] = has(r"clip_cmd")

    # 取实际值
    vals = {}
    for key in ("clip_actions", "act_alpha", "gravity_z_upper"):
        m = re.search(rf"{key}:\s*([-\d.]+)", txt)
        if m:
            try:
                vals[key] = float(m.group(1))
            except ValueError:
                pass

    # joint_limits 是否是"等于关掉"的 ±3.14
    m = re.search(r"joint_limits:\s*\[([^\]]+)\]", txt, re.S)
    if m:
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", m.group(1))]
        if nums and all(abs(abs(x) - 3.14) < 0.01 for x in nums if x):
            vals["joint_limits_disabled"] = True

    return found, vals


def static_report(cfg, deploy_path):
    """不碰硬件的静态检查"""
    print("=" * 70)
    print(" 安全层静态检查（dry-run，不碰硬件）")
    print("=" * 70)

    if deploy_path and os.path.exists(deploy_path):
        found, vals = analyze_deploy_config(deploy_path)
        print(f"\n部署配置: {deploy_path}\n")
        print(f" {'机制':<20} {'配置项':<22} {'值':<12}")
        print("-" * 70)
        rows = [
            ("数值裁剪", "clip_actions", vals.get("clip_actions", "—")),
            ("观测裁剪", "clip_observations", found.get("clip_observations")),
            ("动作缩放", "action_scale", found.get("action_scale")),
            ("关节限位", "joint_limits", found.get("joint_limits")),
            ("跌倒判定", "gravity_z_upper", vals.get("gravity_z_upper", "—")),
            ("动作平滑", "act_alpha", vals.get("act_alpha", "—")),
            ("速度指令限幅", "clip_cmd", found.get("clip_cmd")),
        ]
        for name, key, v in rows:
            if isinstance(v, bool):
                mark = f"{GRN}有{RST}" if v else f"{RED}无{RST}"
            else:
                mark = str(v)
            print(f" {name:<20} {key:<22} {mark}")

        print()
        # 关键警告
        if vals.get("act_alpha") == 1.0:
            print(f"{RED}⚠️  act_alpha = 1.0 → **动作完全不平滑**")
            print(f"    建议调到 0.7-0.9{RST}")
        if vals.get("joint_limits_disabled"):
            print(f"{RED}⚠️  joint_limits 被设成 ±3.14 → **等于关闭了限位保护**{RST}")
        if vals.get("gravity_z_upper") == 1.0:
            print(f"{RED}⚠️  gravity_z_upper = 1.0 → **等于关闭了跌倒保护**")
            print(f"    （单位向量 z 分量不可能 >1，判据永远不触发）{RST}")
        if vals.get("clip_actions") and vals["clip_actions"] >= 100:
            print(f"{YEL}⚠️  clip_actions ≥ 100 → 基本等于不裁剪{RST}")
    else:
        print(f"{YEL}未提供 --deploy，跳过部署配置分析{RST}")

    print()
    print("=" * 70)
    print(" 应有但**通常缺失**的保护（对照检查）")
    print("=" * 70)
    print()
    print(f" {'机制':<16} {'作用':<46}")
    print("-" * 70)
    for name, _, desc in SAFETY_CHECKS:
        print(f" {name:<16} {desc:<46}")
    print()
    print(f"{YEL}⚠️  以上这些**需要你去代码里确认**。")
    print(f"    本脚本只能检查配置，查不了 C++ 实现。{RST}")
    print()
    print(" 最需要用 --live 验证的三条（官方清单）：")
    print("   1. 注入 NaN 动作 → 是否被拒绝")
    print("   2. 注入越界关节目标 → 是否被钳制")
    print("   3. 切断策略输入 → 是否进入安全状态")


def live_inject(motors_py, cfg, joint_idx, out):
    """真注入测试（危险）"""
    specs = [s for s in build_motor_specs(cfg) if s["index"] == joint_idx]
    if not specs:
        sys.exit(f"找不到关节 {joint_idx}")

    print()
    print("=" * 70)
    print(f" ⚠️  LIVE 注入测试 —— 关节 {joint_idx}")
    print("=" * 70)
    print(f"""
{RED}本测试会主动向电机发送**异常指令**：
  1. NaN 位置目标
  2. 超出限位的位置目标
  3. 突然停止发送目标（模拟策略挂掉）

这些操作在**有支撑 + 有人看着**的前提下才是安全的。
{RST}""")
    confirm("确认机器人悬空/有支撑？确认周围无人？确认急停在手边？")

    from _common import create_motors, safe_deinit, read_error_clean
    motors = create_motors(motors_py, specs)
    s, mv = motors[0]
    results = []
    try:
        mv.init_motor()
        time.sleep(0.3)
        mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
        time.sleep(0.1)
        mv.refresh_motor_status(); time.sleep(0.05)
        q0 = mv.get_motor_pos()

        # ── 1. NaN 注入 ────────────────────────────
        print(f"\n[1/3] 注入 NaN 目标（5 个周期）...")
        for _ in range(5):
            try:
                mv.motor_mit_cmd(float("nan"), 0.0, 0.0, 1.0, 0.0)
            except Exception as e:
                print(f"      驱动抛异常（说明有防护）: {type(e).__name__}")
            time.sleep(LOOP_DT)
        time.sleep(0.2)
        mv.refresh_motor_status(); time.sleep(0.05)
        q_after_nan = mv.get_motor_pos()
        nan_ok = not math.isnan(q_after_nan)
        print(f"      NaN 后读回位置 = {q_after_nan}  "
              f"{GRN}位置有效 ✓{RST}" if nan_ok else
              f"      {RED}位置变成 NaN ✗ —— 没有防护！{RST}")
        results.append({"test": "nan_inject", "passed": nan_ok,
                        "q_after": q_after_nan})

        # 回位
        for _ in range(int(0.5 / LOOP_DT)):
            mv.motor_mit_cmd(q0, 0.0, 0.0, 1.0, 0.0)
            time.sleep(LOOP_DT)

        # ── 2. 越界注入 ────────────────────────────
        print(f"\n[2/3] 注入越界目标（q0 + 5.0 rad）...")
        try:
            mv.motor_mit_cmd(q0 + 5.0, 0.0, 0.0, 1.0, 0.0)
        except Exception as e:
            print(f"      驱动抛异常（说明有防护）: {type(e).__name__}")
        time.sleep(0.3)
        mv.refresh_motor_status(); time.sleep(0.05)
        q_clamped = mv.get_motor_pos()
        # 驱动内部 range_map 会 clamp；5.0 rad 远超量程（±12.5），看它是否被限制住
        clamp_ok = abs(q_clamped - q0) < 1.0
        print(f"      越界后位置 = {q_clamped:+.4f}（原 {q0:+.4f}）  "
              f"{GRN}未大幅跑飞 ✓{RST}" if clamp_ok else
              f"      {YEL}位移较大，检查是否被正确钳制{RST}")
        results.append({"test": "clamp_out_of_range", "passed": clamp_ok,
                        "q_after": q_clamped, "delta": q_clamped - q0})

        # 回位
        for _ in range(int(0.5 / LOOP_DT)):
            mv.motor_mit_cmd(q0, 0.0, 0.0, 1.0, 0.0)
            time.sleep(LOOP_DT)

        # ── 3. 断流 ────────────────────────────────
        print(f"\n[3/3] 模拟策略断流（停止发送目标 1 秒）...")
        print(f"      ⚠️ 观察：关节应该「保持」而不是「失控」或「卸力瘫软」")
        time.sleep(1.0)
        mv.refresh_motor_status(); time.sleep(0.05)
        q_drift = mv.get_motor_pos()
        drift = abs(q_drift - q0)
        drift_ok = drift < 0.5
        print(f"      断流 1s 后漂移 = {drift:.4f} rad  "
              f"{GRN}✓{RST}" if drift_ok else f"      {YEL}漂移较大{RST}")
        results.append({"test": "input_loss", "passed": drift_ok,
                        "drift": drift})

    except KeyboardInterrupt:
        print(f"\n{YEL}中断{RST}")
    finally:
        safe_deinit(motors)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "joint": joint_idx, "mode": "live",
                   "results": results}, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    passed = sum(1 for r in results if r["passed"])
    print(f" 结果: {passed}/{len(results)} 通过")
    for r in results:
        mark = f"{GRN}✓{RST}" if r["passed"] else f"{RED}✗{RST}"
        print(f"   {mark} {r['test']}")
    print(f" 已落盘: {out}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="安全层注入测试")
    ap.add_argument("--config", required=True, help="robot.yaml")
    ap.add_argument("--deploy", help="部署 policy yaml（用于静态分析）")
    ap.add_argument("--live", action="store_true",
                    help="真的注入（⚠️ 需硬件，危险）。默认只做静态检查")
    ap.add_argument("--joint", type=int, default=0, help="--live 时测哪个关节")
    ap.add_argument("--out", default="results/safety_inject.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config) if os.path.exists(args.config) else {}
    static_report(cfg, args.deploy)

    if args.live:
        from _common import need_motors_py
        motors_py = need_motors_py()
        live_inject(motors_py, cfg, args.joint, args.out)
    else:
        print()
        print(f"{YEL}以上是静态检查。要真注入，加 --live（需要硬件）。{RST}")


if __name__ == "__main__":
    main()
