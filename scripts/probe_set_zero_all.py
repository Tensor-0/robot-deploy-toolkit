#!/usr/bin/env python3
"""probe_set_zero_all.py — 按顺序对多台电机连续点零（非交互）

⚠️ 不可逆：会对列表里的每台执行 set_motor_zero()。
   必须显式 --confirm，否则只预览不执行。

顺序【从 robot.yaml 读】，不写死。
   接口名会随 USB 枚举漂移（can1/can2 ↔ can0/can1），robot.yaml 是唯一真源。
   写死的表曾在改名后把【左腿】指到 can1（实际是右腿）——
   对只读探针那是个错读数，对本脚本是【点错腿且不可逆】。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import GRN, RED, RST, YEL, describe_error, need_motors_py, read_error_clean  # noqa: E402

# 策略关节顺序（= robot.yaml 里 motor_id / motor_interface 的索引顺序）
_JOINT_NAMES = [
    "左腿 髋pitch", "左腿 髋roll", "左腿 髋yaw", "左腿 膝", "左腿 踝",
    "右腿 髋pitch", "右腿 髋roll", "右腿 髋yaw", "右腿 膝", "右腿 踝",
]

ROBOT_YAML_DEFAULT = os.path.expanduser(
    "~/roboparty_deploy/src/inference/robots/dm10/robot.yaml")


def load_order(path):
    """robot.yaml → [(总线, 电机ID, 关节名), ...]

    ⚠️ 本脚本【刻意不做兜底】。probe_direction.py 读不到配置时退回 ["can0","can1"]，
       对只读探针可以接受；这里执行的是 set_motor_zero()，点错腿不可逆
       ⇒ 读不到配置就【拒绝执行】，绝不猜。
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["motors"]
    ifaces, nums, ids = m["motor_interface"], m["motor_num"], m["motor_id"]
    total = sum(int(n) for n in nums)
    if len(ids) != total:
        raise ValueError(f"motor_id 有 {len(ids)} 项，motor_num 合计 {total} —— 配置自相矛盾")
    if total > len(_JOINT_NAMES):
        raise ValueError(f"motor_num 合计 {total}，超过已知的 {len(_JOINT_NAMES)} 个关节")
    out, k = [], 0
    for bus, n in zip(ifaces, nums):
        for _ in range(int(n)):
            out.append((str(bus), int(ids[k]), _JOINT_NAMES[k]))
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, default=2)
    ap.add_argument("--master-id-offset", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--robot-yaml", default=ROBOT_YAML_DEFAULT,
                    help="robot.yaml —— 电机顺序的唯一真源")
    ap.add_argument("--skip", default="", help="要跳过的 'bus:id' 逗号列表，如 can0:2")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    # ⚠️ 顺序必须在【任何电机操作之前】确定，且读不到就退出 ——
    #    在 --confirm 之后才发现读不到配置，等于先建了对象再说不干。
    try:
        order = load_order(args.robot_yaml)
    except Exception as exc:
        print(f"{RED}⛔ 读不到电机顺序：{exc}{RST}")
        print(f"   路径 {args.robot_yaml}")
        print(f"   set_motor_zero() 不可逆 ⇒ 本脚本不做兜底猜测，拒绝执行。")
        return 2
    print(f"顺序来源: {os.path.abspath(args.robot_yaml)}")

    if not args.confirm:
        print(f"{YEL}ⓘ 未给 --confirm，只预览不执行：{RST}")
        for bus, mid, name in order:
            print(f"   {bus} ID={mid:<2} {name}")
        return 0

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    motors_py = need_motors_py()

    print("=" * 60)
    print(" 连续点零（不可逆）")
    print("=" * 60)
    print(f"{'序':<3}{'电机':<14}{'关节':<14}{'点前(°)':>10}{'点后(°)':>10}{'结果':>8}")
    print("-" * 60)

    results = []
    for i, (bus, mid, name) in enumerate(order, 1):
        tag = f"{bus}:{mid}"
        if tag in skip:
            print(f"{i:<3}{bus+' ID='+str(mid):<14}{name:<14}{'—':>10}{'—':>10}{'跳过':>8}")
            results.append((tag, name, None, None, "skipped"))
            continue

        mv = motors_py.MotorDriver.create_motor(
            motor_id=mid, interface_type="can", interface=bus, motor_type="DM",
            motor_model=args.model, master_id_offset=args.master_id_offset,
            motor_zero_offset=0.0,
        )
        try:
            mv.unlock_motor()
            time.sleep(args.settle)
            mv.refresh_motor_status()
            time.sleep(args.settle)
            before = float(mv.get_motor_pos())

            mv.init_motor()
            time.sleep(args.settle)
            mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
            time.sleep(0.1)
            mv.motor_mit_cmd(0.0, 0.0, 0.0, 1.0, 0.0)
            time.sleep(args.settle)

            ok = mv.set_motor_zero()
            time.sleep(args.settle)
            mv.refresh_motor_status()
            time.sleep(args.settle)
            after = float(mv.get_motor_pos())
            mv.deinit_motor()
            time.sleep(0.1)

            # ⚠️ 0.05 这个数在【三处】各写了一遍，改一处要记得改另外两处：
            #      roboparty_deploy/scripts/set_zero.py 的 calibrate_motor()
            #      probe_set_zero.py
            #      本文件
            # ⚠️ 驱动自己的判据是 0.01（motor_driver.hpp 的 judgment_accuracy_threshold）
            #    ⇒ 这里比驱动松 5 倍，只用来抓"驱动放行之后又漂走"，不是更严的闸。
            #
            # 三态。⚠️ 中间那一态以前被记成 "failed" —— 屏幕上看得到、事后翻
            # 记录看不到，而"驱动说成功但读回偏大"恰恰是最需要人复核的一类。
            if not ok:
                state = "failed"
            elif abs(after) < 0.05:
                state = "ok"
            else:
                state = "suspect"
            verdict = {"ok": "✅", "suspect": "⚠️", "failed": "❌"}[state]
            print(f"{i:<3}{bus+' ID='+str(mid):<14}{name:<14}"
                  f"{before*57.29578:>+10.2f}{after*57.29578:>+10.2f}{verdict:>8}")
            results.append((tag, name, before, after, state))
        except Exception as exc:
            print(f"{i:<3}{bus+' ID='+str(mid):<14}{name:<14}{'—':>10}{'—':>10}{'ERR':>8}  {exc}")
            results.append((tag, name, None, None, f"error:{exc}"))
            try:
                mv.deinit_motor()
            except Exception:
                pass

    print("-" * 60)
    ok_n = sum(1 for r in results if r[4] == "ok")
    suspect = [r for r in results if r[4] == "suspect"]
    bad = [r for r in results if r[4] not in ("ok", "skipped", "suspect")]
    line = f"\n{GRN}成功 {ok_n} / {len(order)}{RST}"
    if suspect:
        line += f"   {YEL}待复核 {len(suspect)}: {[r[0] for r in suspect]}{RST}"
    if bad:
        line += f"   {RED}失败 {len(bad)}: {[r[0] for r in bad]}{RST}"
    print(line)
    # ⚠️ 退出码与改动前一致 —— "待复核"以前被记成 failed，本来就是非零。
    return 0 if not bad and not suspect else 1


if __name__ == "__main__":
    sys.exit(main())
