#!/usr/bin/env python3
"""sample_motors.py — 抓一次电机读数快照（只读、不使能）

给"远程协作"用：操作员在机器人旁边动，助手通过 SSH 反复调用本脚本抓快照，
对比两次的差值就能判断"刚才动的是哪条总线的哪个关节"。

⚠️ 全程不使能（不调用 init_motors），只发失能帧 + 状态查询帧（0xCC）。

用法:
    python3 scripts/sample_motors.py --config <robot.yaml>
输出: 每行 "总线 ID=<id> pos=<rad>"，便于 diff。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (build_motor_specs, check_tx_alive_all, create_motors,  # noqa: E402
                     load_yaml, need_motors_py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--settle", type=float, default=0.05)
    ap.add_argument("--skip-tx-check", action="store_true",
                    help="跳过 TX 假死检测（默认每次都查）")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    specs = build_motor_specs(cfg)

    # ⭐ TX 假死检测（2026-09-17）—— 必须先做，否则下面全是假阴性
    #   ⚠️ 只发失能帧 FF×7 FD，不使能、不产生力矩。
    if not args.skip_tx_check:
        print("TX 通道健康检查：")
        bad = check_tx_alive_all(specs)
        if bad:
            print()
            print("⛔ 有接口 TX 假死 —— 下面的读数会是【假阴性】，先解卡再跑。")
            return 2
        print()

    motors = create_motors(need_motors_py(), specs)

    for _, mv in motors:
        try:
            mv.unlock_motor()
        except Exception:
            pass
    time.sleep(0.2)

    for spec, mv in motors:
        iface = spec["interface"]
        mid = spec["motor_id"]
        try:
            mv.refresh_motor_status()
            time.sleep(args.settle)
            pos = float(mv.get_motor_pos())
            print(f"{iface} ID={mid} pos={pos:+.5f} deg={pos * 57.29578:+.2f}")
        except Exception as exc:
            print(f"{iface} ID={mid} FAIL {type(exc).__name__}")

    for _, mv in motors:
        try:
            mv.unlock_motor()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
