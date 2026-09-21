#!/usr/bin/env python3
"""probe_set_zero.py — 非交互的"单台点零"探针（⭐ 只对指定的 1 台操作）

为什么需要它
------------
`scripts/set_zero.py` 是交互式的（等你按 Enter）。**远程/自动化场景没法用**：
SSH 非交互会话送不进键盘输入，而且它全屏刷新、输出全是控制字符。

本脚本把同样的动作做成**一次性、非交互**的：使能 → 读 → 点零 → 读回 → 失能，
一次跑完，适合"助手远程驱动、操作员在机器旁边摆位"的协作方式。

⚠️⚠️ 安全与不可逆性
------------------
- 本脚本会**使能**目标电机（`init_motor()` 内含 lock_motor），并以 kd=1 纯阻尼保持
  （kp=0，不会硬顶；但会有阻力）
- 它会执行 `set_motor_zero()` —— **把当前姿态写进电机硬件零点，不可逆**
- 因此：**跑之前必须已经摆好位置**。参数 `--confirm` 必须显式给，否则拒绝执行。

用法:
    # 先只看不写（dry-run，默认）
    python3 scripts/probe_set_zero.py --bus can2 --motor-id 2
    # 确认要写（不可逆）
    python3 scripts/probe_set_zero.py --bus can2 --motor-id 2 --confirm
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    GRN,
    RED,
    RST,
    YEL,
    describe_error,
    need_motors_py,
    read_error_clean,
)


def main():
    ap = argparse.ArgumentParser(description="单台点零探针（非交互）")
    ap.add_argument("--bus", required=True, help="总线名，如 can2")
    ap.add_argument("--motor-id", type=int, required=True, help="电机 CAN ID")
    ap.add_argument("--model", type=int, default=2, help="motor_model（2=DM4340P_24V）")
    ap.add_argument("--master-id-offset", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.25)
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="⭐ 必须显式给才会真正写零点；不给则只读不写（dry-run）",
    )
    args = ap.parse_args()

    motors_py = need_motors_py()
    mv = motors_py.MotorDriver.create_motor(
        motor_id=args.motor_id,
        interface_type="can",
        interface=args.bus,
        motor_type="DM",
        motor_model=args.model,
        master_id_offset=args.master_id_offset,
        motor_zero_offset=0.0,
    )

    tag = f"{args.bus} ID={args.motor_id}"
    print(f"=== 单台点零探针 · {tag} ===")
    print(f"   模式: {'⚠️ 会写零点（不可逆）' if args.confirm else '只读（dry-run，不写）'}")
    print()

    # ① 先失能读一次（安全态的读数）
    mv.unlock_motor()
    time.sleep(args.settle)
    mv.refresh_motor_status()
    time.sleep(args.settle)
    before = float(mv.get_motor_pos())
    err0 = read_error_clean(mv, settle=0.1)
    print(f"  ① 失能读数   pos = {before:+.6f} rad  ({before*57.29578:+.2f}°)")
    print(f"     错误码     {err0} ({describe_error(err0)})")

    if not args.confirm:
        print()
        print(f"{YEL}ⓘ dry-run 结束（未写零点）。要真正点零，加 --confirm{RST}")
        return 0

    # ② 使能 + 纯阻尼（与 set_zero.py 同款做法）
    print("  ② 使能（kd=1 纯阻尼，kp=0）...")
    mv.init_motor()
    time.sleep(args.settle)
    mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
    time.sleep(0.1)
    mv.motor_mit_cmd(0.0, 0.0, 0.0, 1.0, 0.0)
    time.sleep(args.settle)
    mv.refresh_motor_status()
    time.sleep(0.1)
    at_zero = float(mv.get_motor_pos())
    print(f"     使能后读数 pos = {at_zero:+.6f} rad  ({at_zero*57.29578:+.2f}°)")

    # ③ 点零（⭐ 接返回值）
    print("  ③ 写零点 ...")
    ok = mv.set_motor_zero()
    time.sleep(args.settle)
    mv.refresh_motor_status()
    time.sleep(args.settle)
    after = float(mv.get_motor_pos())
    print(f"     驱动返回   {ok}")
    print(f"     点后读数   pos = {after:+.6f} rad  ({after*57.29578:+.2f}°)")

    # ④ 失能收尾
    mv.deinit_motor()
    time.sleep(args.settle)

    print()
    # ⚠️ 0.05 这个数在【三处】各写了一遍，改一处要记得改另外两处：
    #      roboparty_deploy/scripts/set_zero.py 的 calibrate_motor()
    #      本文件
    #      probe_set_zero_all.py
    # ⚠️ 驱动自己的判据是 0.01（motor_driver.hpp 的 judgment_accuracy_threshold）
    #    ⇒ 这里比驱动松 5 倍，只用来抓"驱动放行之后又漂走"，不是更严的闸。
    verdict_ok = bool(ok) and abs(after) < 0.05
    if verdict_ok:
        print(f"{GRN}  ✅ 点零成功：{before:+.4f} rad 的偏差已清零，残留 {after:+.6f} rad{RST}")
    elif ok:
        print(f"{YEL}  ⚠️ 驱动报成功，但残留 {after:+.6f} rad 偏大 —— 请复核{RST}")
    else:
        print(f"{RED}  ❌ 点零失败（驱动返回 False）—— 未生效，可重试{RST}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
