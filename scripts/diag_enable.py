#!/usr/bin/env python3
"""diag_enable.py — 只读诊断：电机到底使能了没有

⚠️ 为什么需要它
--------------
`read_error_clean()`（_common.py）会**先发 clear_motor_error() 再读** ——
而达妙的 error_id 里 **1 = 「使能」是状态不是错误**。
⇒ 用它会**把「使能」清成 0**，让人误以为电机没使能。
**诊断使能状态必须读 clear 之前的原始值。**

本脚本做两件事：
  ① 读 init_motor() 的返回值（驱动自己判的使能结果）
  ② 读 clear 之前的原始 error_id，以及 clear 之后的对照值

⚠️ 全程只读 —— 不下发位置指令、不产生力矩、电机不失能也不动。
   唯一的"写"是 init_motor() 内部的使能帧（这样才能知道能不能使能），
   跑完立刻 deinit 失能。

用法:
    python3 scripts/diag_enable.py --bus can2 --motor-id 2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import GRN, RED, RST, YEL, describe_error, need_motors_py  # noqa: E402


def raw_error(mv):
    """⚠️ 读【未清理】的原始 error_id —— 不能先 clear"""
    try:
        e = mv.get_error_id()
        return int(e) if isinstance(e, (int, float)) else -1
    except Exception as exc:
        return f"读取失败({type(exc).__name__})"


def main():
    ap = argparse.ArgumentParser(description="只读诊断：电机使能状态")
    ap.add_argument("--bus", required=True)
    ap.add_argument("--motor-id", type=int, required=True)
    ap.add_argument("--model", type=int, default=2)
    ap.add_argument("--master-id-offset", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.25)
    args = ap.parse_args()

    motors_py = need_motors_py()
    mv = motors_py.MotorDriver.create_motor(
        motor_id=args.motor_id, interface_type="can", interface=args.bus,
        motor_type="DM", motor_model=args.model,
        master_id_offset=args.master_id_offset, motor_zero_offset=0.0,
    )

    print("=" * 72)
    print(f"  只读诊断 · {args.bus} ID={args.motor_id}")
    print("=" * 72)
    print(f"  {YEL}全程不下发位置指令、不产生力矩{RST}")
    print()

    try:
        # ---------- 阶段 1：失能态（基线） ----------
        mv.unlock_motor()
        time.sleep(args.settle)
        mv.refresh_motor_status()
        time.sleep(args.settle)
        pos_dis = float(mv.get_motor_pos())
        err_dis = raw_error(mv)
        print(f"  ① 失能态")
        print(f"     位置     {pos_dis:+.6f} rad")
        print(f"     原始err  {err_dis}  ({describe_error(err_dis) if isinstance(err_dis, int) else '—'})")

        # ---------- 阶段 2：使能 ----------
        print()
        print(f"  ② 调 init_motor()（内部会 unlock → set_mode(MIT) → lock 使能）...")
        rc = mv.init_motor()
        print(f"     ⭐ init_motor() 返回值 = {rc}")
        try:
            rc_i = int(rc)
            if rc_i == 0:
                print(f"        {GRN}→ 驱动判定：成功（DM_DOWN=0x00 在驱动里代表无错误）{RST}")
            elif rc_i == 1:
                print(f"        {YEL}→ 驱动判定：DM_UP=0x01（即 error_id 读到 1 = 使能）{RST}")
            elif rc_i == 13:
                print(f"        {RED}→ 驱动判定：LOST_CONN 通讯丢失{RST}")
            else:
                print(f"        {RED}→ 驱动判定：{describe_error(rc_i)}{RST}")
        except Exception:
            pass

        time.sleep(args.settle)
        mv.refresh_motor_status()
        time.sleep(args.settle)

        # ⭐ 关键：clear 之前读！
        err_raw_enabled = raw_error(mv)
        pos_en = float(mv.get_motor_pos())
        print(f"     使能后位置  {pos_en:+.6f} rad  (变化 {pos_en - pos_dis:+.6f})")
        print(f"     ⭐ 原始err（clear 之前）= {err_raw_enabled}"
              f"  ({describe_error(err_raw_enabled) if isinstance(err_raw_enabled, int) else '—'})")

        # ---------- 阶段 3：keep-alive 观察 ----------
        print()
        print(f"  ③ 保持使能并连读 3 秒（每 0.3s 一帧，不发位置指令）")
        positions = []
        for i in range(10):
            time.sleep(0.3)
            mv.refresh_motor_status()
            p = float(mv.get_motor_pos())
            positions.append(p)
            mark = ""
            if i > 0 and abs(p - positions[0]) > 1e-4:
                mark = "  ← 有变化"
            print(f"     t={0.3*(i+1):4.1f}s  pos={p:+.6f}{mark}")

        spread = max(positions) - min(positions)
        print(f"     3 秒内极差 = {spread:.6f} rad  "
              f"({'纹丝不动' if spread < 1e-4 else '有变化'})")

        # ---------- 阶段 4：clear 之后对照 ----------
        print()
        print(f"  ④ 对照：调 clear_motor_error() 之后再读（证明它会把状态清掉）")
        try:
            mv.clear_motor_error()
            time.sleep(args.settle)
            mv.refresh_motor_status()
            time.sleep(args.settle)
            err_cleared = raw_error(mv)
            print(f"     clear 之后 err = {err_cleared}"
                  f"  ({describe_error(err_cleared) if isinstance(err_cleared, int) else '—'})")
            if isinstance(err_raw_enabled, int) and isinstance(err_cleared, int):
                if err_raw_enabled != err_cleared:
                    print(f"     {YEL}⚠️ 证实：clear 把 {err_raw_enabled} → {err_cleared}，"
                          f"状态被清掉了{RST}")
                else:
                    print(f"     两者相同（{err_raw_enabled}）—— 未观察到 clear 的副作用")
        except Exception as exc:
            print(f"     clear 失败：{exc}")

        # ---------- 判定 ----------
        print()
        print("=" * 72)
        print("  判定")
        print("=" * 72)
        if isinstance(err_raw_enabled, int) and err_raw_enabled == 1:
            print(f"  {GRN}✅ 电机【已使能】（原始 err=1）{RST}")
            print(f"     ⇒ 前面没动，是【力矩被算没了】或【kp 不足】，不是使能问题")
        elif isinstance(err_raw_enabled, int) and err_raw_enabled == 0:
            print(f"  {RED}❌ 电机【未使能】（原始 err=0）—— 使能帧没生效{RST}")
            print(f"     ⇒ 查：电机电源 / 急停 / CAN 使能帧是否被驱动发出去")
        elif isinstance(err_raw_enabled, int) and err_raw_enabled > 7:
            print(f"  {RED}❌ 电机报错：{describe_error(err_raw_enabled)}{RST}")
        else:
            print(f"  {YEL}⚠️ 读数异常：{err_raw_enabled}{RST}")

    finally:
        try:
            mv.deinit_motor()
        except Exception:
            pass
        print()
        print("  （已失能收尾）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
