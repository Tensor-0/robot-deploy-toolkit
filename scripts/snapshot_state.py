#!/usr/bin/env python3
"""snapshot_state.py — 全状态只读快照（⭐ 全程不使能电机）

为什么需要它
------------
标定/对齐之前先要有一把"尺子"：机器人现在摆着某个姿态，每个关节的编码器
到底读多少？IMU 的三个轴朝哪？这份快照就是后面所有标定的基准。
它同时一次性回答三个悬而未决的问题：
  ① 哪条总线（can1/can2）对应哪条腿 —— 三份文档说法互相矛盾，读数一次裁决
  ② 电机的"出厂零位"在哪个姿态 —— 直腿位？还是屈膝站姿？（见下）
  ③ IMU 的 Z 轴朝上还是朝下 —— **这是安全项**：Z 轴朝下会让直立时的
     gravity_b 变成 +1，触发唯一的跌倒保护直接关机（obs_manager.cpp:203）

⚠️⚠️ 安全设计（本脚本与其它脚本最大的不同）
------------------------------------------
**本脚本绝不给电机使能。** 现有测试脚本走 `RobotInterface::init_motors()`，
而它的最后一步是 `lock_motor()`（使能）—— 那是"会动"的路径。
这里只走逐台 motors_py API：
    unlock_motor()          ← 发 FF×7 + FD = 失能（驱动注释："失能 = 进入读模式"）
    refresh_motor_status()  ← 发 0xCC 请求一次状态帧（纯读请求）
    get_motor_pos() 等      ← 读回
全程**不发任何 MIT / POS / SPD 指令**，也不改控制模式（那是写进电机的持久参数），
所以物理上不可能产生力矩。

⚠️ 仍然必须：机器人架住/吊住、腿悬空、旁边有人。
   失能状态下腿是软的，会自己垂下来 —— 这次没事，但别让人站在腿下面。

用法:
    source /opt/ros/humble/setup.bash && source <roboparty_deploy>/install/setup.bash
    python3 scripts/snapshot_state.py --config <roboparty_deploy>/src/inference/robots/dm10/robot.yaml

退出码: 0 正常 / 1 硬件错误 / 2 未确认
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    GRN,
    RED,
    RST,
    YEL,
    build_motor_specs,
    confirm,
    create_motors,
    describe_error,
    load_yaml,
    need_motors_py,
    read_error_clean,
)

# 模型侧参考值（唯一来源：UniLab 的 MJCF，见 memory `dm10-frame-alignment`）
# ⚠️ 2026-09-21：dm10 的 home keyframe 从【屈膝】改成了【直腿全零】。所以下面按
#    【位形】命名 —— 消除 "home 到底指哪个姿态" 这个歧义正是那次改动的目的之一。
BENT_QPOS = [-0.4, 0.0, 0.0, 0.8, -0.4] * 2  # 旧 home：屈膝站姿
STRAIGHT_QPOS = [0.0] * 10                   # 新 home：两腿完全伸直

# 本工具的操作前提：机器人被摆成【屈膝站姿】。要改成直腿，必须同时改 POSED 和
# POSED_NAME —— 只改一个会让下面 expected 的配对反过来（见那段注释）。
POSED = BENT_QPOS
POSED_NAME = "屈膝站姿"


def parse_args():
    ap = argparse.ArgumentParser(description="只读全状态快照（不使能电机）")
    ap.add_argument("--config", required=True, help="robot.yaml（含 motors 段与 imu 段）")
    ap.add_argument("--out", default=None, help="结果 JSON 路径（默认 results/state_<时间戳>.json）")
    ap.add_argument("--no-imu", action="store_true", help="跳过 IMU")
    ap.add_argument("--settle", type=float, default=0.15, help="每条指令后等待秒数（默认 0.15）")
    ap.add_argument("--imu-samples", type=int, default=20, help="IMU 采样次数（默认 20，用于判断是否静止）")
    ap.add_argument("--yes", action="store_true", help="跳过交互确认（非交互环境必须显式给）")
    return ap.parse_args()


def banner():
    print(f"{YEL}{'=' * 68}")
    print(" 只读全状态快照 —— 本脚本【不会】给电机使能")
    print("=" * 68)
    print(" 会做什么：逐台失能 → 请求一次状态帧 → 读位置/速度/电流/温度/错误码")
    print(" 不会做：使能、发动作指令、改控制模式、标零、写参数")
    print(f"{'=' * 68}{RST}")
    print(f"{YEL}  ⚠️ 前提：机器人已架住/吊住、腿悬空、旁边有人{RST}")
    print(f"{YEL}  ⚠️ 失能状态下腿是软的，会自然下垂 —— 这是预期现象{RST}")
    print(f"{YEL}  ⭐ 请先把机器人摆成【屈膝站姿】(home: 髋 -0.4 / 膝 +0.8 / 踝 -0.4)，{RST}")
    print(f"{YEL}     否则下面的「零位在哪」两假设对比无法解释（读数会整体平移）{RST}")
    print()


def snapshot_motors(mv, spec, settle):
    """读一台电机的全部可读量。全程不发控制指令。"""
    rec = {
        "index": spec["index"],
        "interface": spec["interface"],
        "motor_id": spec["motor_id"],
    }
    try:
        # ① 先确保失能（幂等；即使上一状态是使能，这一步也会把它放开）
        mv.unlock_motor()
        time.sleep(settle)
        # ② 请求一次状态帧（纯读请求 0xCC）
        mv.refresh_motor_status()
        time.sleep(settle)
        rec["pos_rad"] = round(float(mv.get_motor_pos()), 6)
        rec["vel_rad_s"] = round(float(mv.get_motor_spd()), 6)
        rec["current_a"] = round(float(mv.get_motor_current()), 6)
        rec["temp_c"] = round(float(mv.get_motor_temperature()), 2)
        try:
            rec["control_mode"] = int(mv.get_motor_control_mode())
        except Exception:
            rec["control_mode"] = None
        try:
            rec["response_count"] = int(mv.get_response_count())
        except Exception:
            rec["response_count"] = None
        rec["ok"] = True
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    # 错误码单独走"先清后读"（error_id 只增不减，直接读会读到历史错误）
    rec["error_id"] = read_error_clean(mv, settle=settle)
    rec["error_text"] = describe_error(rec["error_id"])
    return rec


def snapshot_imu(cfg, samples, settle):
    imu_cfg = cfg.get("imu") or {}
    if not imu_cfg:
        return {"ok": False, "error": "robot.yaml 里没有 imu 段"}
    try:
        import imu_py
    except ImportError:
        return {"ok": False, "error": "找不到 imu_py（需先 source install/setup.bash）"}

    try:
        imu = imu_py.IMUDriver.create_imu(
            int(imu_cfg.get("imu_id", 1)),
            str(imu_cfg.get("imu_interface_type", "serial")),
            str(imu_cfg.get("imu_interface", "/dev/ttyACM0")),
            str(imu_cfg.get("imu_type", "DM_IMU_L1")),
            int(imu_cfg.get("baudrate", 0)),
        )
    except Exception as exc:
        return {"ok": False, "error": f"创建 IMU 失败: {exc}"}

    acc, gyro = [], []
    quat = None
    for _ in range(max(1, samples)):
        try:
            quat = [round(float(v), 5) for v in imu.get_quat()]
            acc.append([float(v) for v in imu.get_lin_acc()])
            gyro.append([float(v) for v in imu.get_ang_vel()])
        except Exception:
            pass
        time.sleep(0.02)

    if not acc:
        return {"ok": False, "error": "IMU 一帧都没读到（串口？波特率？）"}

    n = len(acc)
    mean = lambda seq: [round(sum(v[i] for v in seq) / n, 4) for i in range(3)]  # noqa: E731
    std = lambda seq: [  # noqa: E731
        round((sum((v[i] - sum(u[i] for u in seq) / n) ** 2 for v in seq) / n) ** 0.5, 4)
        for i in range(3)
    ]
    return {
        "ok": True,
        "samples": n,
        "quat_wxyz": quat,
        "lin_acc_mean_ms2": mean(acc),
        "lin_acc_std": std(acc),
        "ang_vel_mean_rad_s": mean(gyro),
        "ang_vel_std": std(gyro),
        "interface": imu_cfg.get("imu_interface"),
    }


def imu_verdict(imu):
    """Z 轴朝向判定 —— 安全关键（装反 = 上电瞬间触发跌倒保护关机）"""
    if not imu.get("ok"):
        return "未读到 IMU", False
    az = imu["lin_acc_mean_ms2"][2]
    spread = max(imu["lin_acc_std"])
    if spread > 1.0:
        return f"仍在晃动（加速度标准差 {spread:.2f} m/s²）—— 静止后重测", None
    if az > 5.0:
        return f"✅ Z 轴朝上（静止时 a_z = {az:+.2f} ≈ +9.8）—— gravity_b ≈ (0,0,-1)，安全", True
    if az < -5.0:
        return f"❌ Z 轴【朝下】（a_z = {az:+.2f}）—— gravity_b.z 会变成 +1，开机即触发跌倒保护关机", False
    return f"⚠️ 不竖直（a_z = {az:+.2f}；倾斜 ≈ {np_acos_deg(az / 9.81):.0f}°）—— 摆直后重测", None


def np_acos_deg(x):
    import math

    return math.degrees(math.acos(max(-1.0, min(1.0, x))))


def print_hypotheses(reads):
    """读数 vs 两个假设：出厂零位在直腿位，还是在屈膝站姿？

    ⚠️ 前提：机器人**当前**处于屈膝站姿（home）。此时
        读数 = 当前姿态的模型角 − 设零时的姿态
      · 出厂零位在【直腿位】→ 读数 ≈ home − 0 = home
      · 出厂零位在【屈膝站姿】→ 读数 ≈ home − home = 0
    （当前姿态若不是 home，两个数的解释都要跟着平移，别硬套。）
    """
    pos = [r.get("pos_rad") for r in reads]
    if any(p is None for p in pos):
        print(f"{YEL}  （有电机没读到，跳过零位假设对比）{RST}")
        return
    print()
    print("─" * 68)
    print(f" 出厂零位在哪？（前提：机器人【当前】摆的是{POSED_NAME}）")
    print("─" * 68)

    def stats(expected):
        d = [p - e for p, e in zip(pos, expected, strict=True)]
        return max(abs(x) for x in d), (sum(x * x for x in d) / len(d)) ** 0.5

    # ⚠️ 配对不能反：expected = 摆成的姿态 − 零位所在的姿态
    #    摆屈膝时：零位在直腿 ⇒ 读数 = 屈膝 − 0 = BENT_QPOS
    #              零位在屈膝 ⇒ 读数 = 屈膝 − 屈膝 = 0
    best_label, best_rms = None, float("inf")
    for label, zero_pose in (
        ("A 零位在【直腿位】", STRAIGHT_QPOS),
        ("B 零位在【屈膝位】", BENT_QPOS),
    ):
        expected = [p - z for p, z in zip(POSED, zero_pose, strict=True)]
        mx, rms_v = stats(expected)
        if rms_v < best_rms:
            best_label, best_rms = label, rms_v
        print(f"  {label:30s} 最大残差 {mx:7.3f} rad    RMS {rms_v:7.3f} rad")
    print(f"  ⇒ 更像：{best_label}（RMS {best_rms:.3f} rad）")
    if best_rms > 0.05:
        print(f"  {YEL}⚠️ 两个假设的残差都很大 —— 可能是：装配姿态与 home 不同 / 某个关节装反了 /"
              f" ID↔关节对应不是 1:1。先别下结论，把原始读数拿来看。{RST}")
    print()
    print(f"  {'关节':<6}{'读数 (rad)':>12}{POSED_NAME:>10}{'差':>16}")
    for i, (p, h) in enumerate(zip(pos, POSED, strict=True)):
        print(f"  [{i}]   {p:>12.4f}{h:>10.2f}{p - h:>16.4f}")


def main():
    args = parse_args()
    banner()
    if not args.yes:
        confirm(f"机器人已架住/吊住、腿悬空、旁边有人？且已摆成{POSED_NAME}？")
    print()

    cfg = load_yaml(args.config)
    specs = build_motor_specs(cfg)
    if not specs:
        print(f"{RED}✗ 配置里没解析出电机{RST}")
        sys.exit(1)
    buses = {}
    for s in specs:
        buses.setdefault(s["interface"], []).append(s["motor_id"])
    print(f"  从配置解析出 {len(specs)} 台电机：")
    for b, ids in buses.items():
        print(f"    {b}: ID {ids}")
    print()

    motors_py = need_motors_py()
    motors = create_motors(motors_py, specs)

    # ⚠️ 先全部失能，再读 —— 顺序不能反
    print("  确保所有电机处于失能状态（发 FF×7+FD）...", end=" ", flush=True)
    for _, mv in motors:
        try:
            mv.unlock_motor()
        except Exception:
            pass
    time.sleep(args.settle)
    print(f"{GRN}完成{RST}")

    print("  逐台读状态 ...")
    reads = []
    for spec, mv in motors:
        rec = snapshot_motors(mv, spec, args.settle)
        reads.append(rec)
        if rec["ok"]:
            print(
                f"    {rec['interface']} ID={rec['motor_id']:<2d}  "
                f"pos={rec['pos_rad']:+.4f} rad  vel={rec['vel_rad_s']:+.3f}  "
                f"t={rec['temp_c']:5.1f}°C  err={rec['error_id']}({rec['error_text']})"
            )
        else:
            print(f"    {rec['interface']} ID={rec['motor_id']:<2d}  {RED}读失败: {rec['error']}{RST}")

    imu = {"ok": False, "skipped": True} if args.no_imu else snapshot_imu(cfg, args.imu_samples, args.settle)
    verdict, imu_ok = imu_verdict(imu)

    print_hypotheses(reads)

    print()
    print("─" * 68)
    print(" IMU")
    print("─" * 68)
    if imu.get("ok"):
        print(f"  四元数 (w,x,y,z)   : {imu['quat_wxyz']}")
        print(f"  线加速度 均值      : {imu['lin_acc_mean_ms2']}  (标准差 {imu['lin_acc_std']})")
        print(f"  角速度   均值      : {imu['ang_vel_mean_rad_s']}  (标准差 {imu['ang_vel_std']})")
        print(f"  样本数             : {imu['samples']}")
    else:
        print(f"  {imu.get('error', '跳过')}")
    print(f"  判定: {verdict}")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join("results", f"state_{ts}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload = {
        "schema": "deploy-state-snapshot/1",
        "created": dt.datetime.now().isoformat(),
        "config": os.path.abspath(args.config),
        "note": "只读快照：电机全程未使能，未发任何控制指令",
        "motors": reads,
        "imu": imu,
        "imu_verdict": verdict,
        "reference": {"model_zero_all_joints": MODEL_ZERO, "home_qpos": HOME_QPOS,
                      "source": "UniLab scene_flat.xml:34-37"},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print()
    print(f"{GRN}💾 已落盘: {out}{RST}")

    # 收尾：确保失能（幂等）
    for _, mv in motors:
        try:
            mv.unlock_motor()
        except Exception:
            pass

    n_bad = sum(1 for r in reads if not r.get("ok"))
    sys.exit(1 if (n_bad or imu_ok is False) else 0)


if __name__ == "__main__":
    main()
