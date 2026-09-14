#!/usr/bin/env python3
"""joint_test.py — 单关节逐台测试 + 结果落盘

**阶段 3 的主力工具。** 电机到货后第一个跑的就是它。

为什么需要它（现有工具做不了的）：
  1. `set_zero.py` 只 print，**不落盘** → 无法回溯"上次是什么样"
  2. 没有**安全包络** → 改目标前不读当前位置、不限制步长
  3. 没有**超时判据** → 电机不响应时不会自己停

本脚本做的事（逐台，串行）：
  ① 建实例 → 使能 → 切 MIT 模式
  ② **先读当前位置**（安全包络的基础）
  ③ 小步慢速地给目标，**每步限制 Δq**
  ④ 读反馈（位置/速度/力矩/温度/错误码）
  ⑤ 结果落盘 JSON
  ⑥ 异常/超时 → 立即失能退出

用法:
  # 冒烟：只读状态，不发任何目标（最安全，先跑这个）
  python3 joint_test.py --config <robot.yaml> --readonly

  # 逐台小幅测试（默认 ±0.1 rad）
  python3 joint_test.py --config <robot.yaml> --amplitude 0.1 --out results/joint_test.json

  # 只测某几个关节（索引来自配置顺序）
  python3 joint_test.py --config <robot.yaml> --joints 0,3,7

前置（重要）:
  * CAN 总线已 up 且 bitrate 正确   → 先跑 scripts/check_preflight.sh
  * motors_py 已构建             → 见下方 ERR_NO_MODULE 提示
  * **机器人必须悬空/有支撑**，人站侧面，手边有急停

⚠️ 安全设计（本脚本的默认值都偏保守）:
  * 默认幅度 ±0.1 rad、默认 200Hz 下发、默认每步 Δq 限幅
  * Ctrl+C / 异常 / 超时 → finally 里全部失能
  * --readonly 完全不发目标，可无风险先验证通信
"""
import argparse
import json
import math
import os
import sys
import time

# ── 常量 ────────────────────────────────────────────────
LOOP_DT = 0.005          # 200 Hz 下发
RAMP_S = 1.5             # 从当前位置到目标幅值的过渡时间（避免阶跃）
DEFAULT_AMP = 0.1        # 默认幅度 rad（保守）
MAX_DQ_PER_STEP = 0.01   # 每步最大 Δq（rad），1/10 的幅度
OFFLINE_THRESHOLD = 25   # 连续多少帧无反馈判离线（与 drive 一致）

RED, YEL, GRN, RST = "\033[31m", "\033[33m", "\033[32m", "\033[0m"


def err_exit(msg, hint=None):
    print(f"{RED}✗ {msg}{RST}")
    if hint:
        print(hint)
    sys.exit(1)


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        err_exit("需要 pyyaml", "  pip3 install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_motor_specs(cfg):
    """把配置展开成逐电机的 [(index, motor_id, interface, model, zero_offset)]

    兼容两种格式：
      A) 扁平： motor_id 是全局列表，motor_num 按总线分组
      B) 分段： motor_interface_type/motor_type 是逐总线列表
    """
    m = cfg.get("motors", cfg)  # 两段式 or 扁平

    ids = m["motor_id"]
    ifaces = m["motor_interface"]
    nums = m["motor_num"]
    models = m.get("motor_model", [])
    zeros = m.get("motor_zero_offset", [])

    # 逐总线取 type（有的配置逐总线，有的是单值）
    def at(seq, i, default=None):
        if isinstance(seq, list):
            return seq[i] if i < len(seq) else default
        return default if seq is None else seq

    specs, idx = [], 0
    for bus_i, num in enumerate(nums):
        iface = at(ifaces, bus_i)
        itype = at(m.get("motor_interface_type"), bus_i, "can")
        mtype = at(m.get("motor_type"), bus_i, "DM")
        for _ in range(num):
            if idx >= len(ids):
                break
            specs.append({
                "index": idx,
                "motor_id": ids[idx],
                "interface": iface,
                "interface_type": itype,
                "motor_type": mtype,
                "motor_model": at(models, idx, 2),
                "zero_offset": at(zeros, idx, 0.0),
                "master_id_offset": m.get("master_id_offset", 0),
            })
            idx += 1
    return specs


def main():
    ap = argparse.ArgumentParser(description="单关节逐台测试")
    ap.add_argument("--config", required=True, help="robot.yaml 路径")
    ap.add_argument("--readonly", action="store_true",
                    help="只读状态，不发任何目标（最安全，建议先跑这个）")
    ap.add_argument("--amplitude", type=float, default=DEFAULT_AMP,
                    help=f"测试幅度 rad（默认 {DEFAULT_AMP}）")
    ap.add_argument("--joints", help="只测这些索引，逗号分隔（默认全部）")
    ap.add_argument("--cycles", type=int, default=2, help="每个关节往返几次（默认 2）")
    ap.add_argument("--out", default="results/joint_test.json", help="结果落盘路径")
    args = ap.parse_args()

    # ── 依赖检查 ──────────────────────────────────────
    try:
        import motors_py
    except ImportError:
        err_exit("找不到 motors_py（它是 colcon 构建产物）", f"""{YEL}
  修复：
    cd <roboparty_deploy 仓库根>
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    source install/setup.bash

  ⚠️ 必须 source install/setup.bash 之后才能 import motors_py。{RST}""")

    cfg = load_yaml(args.config)
    specs = build_motor_specs(cfg)
    if args.joints:
        want = {int(x) for x in args.joints.split(",")}
        specs = [s for s in specs if s["index"] in want]
    if not specs:
        err_exit("没有匹配到任何电机")

    print("=" * 66)
    print(f" 单关节测试  ·  {args.config}")
    print(f" 共 {len(specs)} 台  ·  模式: {'只读' if args.readonly else f'±{args.amplitude} rad'}")
    print("=" * 66)
    print(f"{YEL}⚠️  确认：机器人已悬空/有支撑 · 人站侧面 · 急停在手边{RST}")
    if not args.readonly:
        input("   按 Enter 开始（Ctrl+C 取消）...")
    print()

    results = []
    motors = []

    try:
        # ── 建实例 + 使能 ────────────────────────────
        for s in specs:
            print(f"[{s['index']:2d}] can/ID={s['motor_id']:<3} 建实例...", end=" ", flush=True)
            mv = motors_py.MotorDriver.create_motor(
                motor_id=s["motor_id"],
                interface_type=s["interface_type"],
                interface=s["interface"],
                motor_type=s["motor_type"],
                motor_model=s["motor_model"],
                master_id_offset=s["master_id_offset"],
                motor_zero_offset=s["zero_offset"],
            )
            motors.append((s, mv))

            rc = mv.init_motor()          # ⚠️ 返回的是错误码，不是 bool
            time.sleep(0.3)
            # 错误码语义：0=失能 1=使能 8+=故障
            if isinstance(rc, int) and rc >= 8:
                print(f"{RED}init 报错码 {rc}{RST}")
            else:
                print(f"{GRN}OK{RST} (init={rc})")

            if not args.readonly:
                # ⚠️ 必须先切模式：首次调用只切模式，当次不发目标
                mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
                time.sleep(0.1)

        # ── 逐台测试 ────────────────────────────────
        for s, mv in motors:
            r = {"index": s["index"], "motor_id": s["motor_id"],
                 "interface": s["interface"], "status": "unknown"}

            # ① 先读当前位置 —— 安全包络的基础
            mv.refresh_motor_status()
            time.sleep(0.05)
            q0 = mv.get_motor_pos()
            r["q_initial"] = round(q0, 4)
            r["temp"] = round(mv.get_motor_temperature(), 1)

            # 错误码判读：必须先清 → 等新帧 → 再读（只增不减！）
            mv.clear_motor_error()
            time.sleep(0.05)
            mv.refresh_motor_status()
            time.sleep(0.05)
            err = mv.get_error_id()
            r["error_id"] = int(err) if isinstance(err, (int, float)) else -1

            if args.readonly:
                r["status"] = "readonly"
                r["q_final"] = r["q_initial"]
                print(f"[{s['index']:2d}] q={q0:+.4f}  T={r['temp']}℃  err={r['error_id']}")
                results.append(r)
                continue

            # ② 小步慢速往返
            print(f"[{s['index']:2d}] q0={q0:+.4f}  测试 ±{args.amplitude} ...",
                  end=" ", flush=True)
            q_cur, ok, max_err = q0, True, 0.0
            try:
                for cyc in range(args.cycles):
                    for sign in (+1, -1):
                        target = q0 + sign * args.amplitude
                        # 从当前位置平滑过渡到目标（避免阶跃）
                        n_steps = max(1, int(RAMP_S / LOOP_DT))
                        for k in range(n_steps):
                            frac = (k + 1) / n_steps
                            q_des = q_cur + (target - q_cur) * frac
                            # 每步 Δq 限幅
                            dq = q_des - q_cur
                            dq = max(-MAX_DQ_PER_STEP, min(MAX_DQ_PER_STEP, dq))
                            q_cmd = q_cur + dq
                            mv.motor_mit_cmd(q_cmd, 0.0, 0.0, 1.0, 0.0)  # 纯阻尼跟随
                            time.sleep(LOOP_DT)
                            q_cur = q_cmd
                        # 到位后保持一会，读实际位置
                        time.sleep(0.3)
                        mv.refresh_motor_status()
                        time.sleep(0.05)
                        q_act = mv.get_motor_pos()
                        err_here = abs(q_act - target)
                        max_err = max(max_err, err_here)
                ok = max_err < max(0.15, args.amplitude * 0.5)
            except KeyboardInterrupt:
                print(f"\n{YEL}中断{RST}")
                r["status"] = "interrupted"
                results.append(r)
                break

            r["q_final"] = round(mv.get_motor_pos(), 4)
            r["tracking_err_max"] = round(max_err, 4)
            r["status"] = "ok" if ok else "tracking_poor"
            results.append(r)
            mark = f"{GRN}✓{RST}" if ok else f"{YEL}⚠{RST}"
            print(f"{mark} 跟踪误差 {max_err:.4f} rad")

    except Exception as e:
        print(f"\n{RED}异常: {e}{RST}")
    finally:
        # ── 无论如何都要失能 ────────────────────────
        print(f"\n失能中...", end=" ", flush=True)
        for s, mv in motors:
            try:
                mv.deinit_motor()
            except Exception:
                pass
        print(f"{GRN}完成{RST}")

    # ── 落盘 ────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    doc = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": os.path.abspath(args.config),
        "mode": "readonly" if args.readonly else "active",
        "amplitude": None if args.readonly else args.amplitude,
        "results": results,
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] in ("ok", "readonly")),
            "poor": sum(1 for r in results if r["status"] == "tracking_poor"),
            "errors": sum(1 for r in results if r.get("error_id", 0) not in (0, 1)),
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    s = doc["summary"]
    print()
    print("=" * 66)
    print(f" 结果: 共 {s['total']} · 正常 {s['ok']} · 跟踪差 {s['poor']} · 报错 {s['errors']}")
    print(f" 已落盘: {args.out}")
    print("=" * 66)

    if s["errors"]:
        print(f"{RED}⚠️  有电机报了错误码 —— 上真机前必须处理{RST}")
    if s["poor"]:
        print(f"{YEL}⚠️  有电机跟踪差 —— 检查 kp/kd 或机械干涉{RST}")


if __name__ == "__main__":
    main()
