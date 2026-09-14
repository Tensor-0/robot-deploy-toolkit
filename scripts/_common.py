#!/usr/bin/env python3
"""_common.py — 共享基础（配置解析 / 依赖检查 / 安全退出）

其他脚本从这里 import，避免重复。
"""
import os
import sys
import time

RED, YEL, GRN, RST = "\033[31m", "\033[33m", "\033[32m", "\033[0m"

LOOP_DT = 0.005          # 200 Hz
OFFLINE_THRESHOLD = 25   # 与驱动一致：连续 N 帧无反馈判离线


def need_motors_py():
    """检查 motors_py 可用（它是 colcon 构建产物）"""
    try:
        import motors_py
        return motors_py
    except ImportError:
        print(f"{RED}✗ 找不到 motors_py（colcon 构建产物）{RST}")
        print(f"""{YEL}
  修复：
    cd <roboparty_deploy 仓库根>
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    source install/setup.bash

  ⚠️ 必须 source install/setup.bash 之后才能 import。{RST}""")
        sys.exit(1)


def need_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        print(f"{RED}✗ 需要 pyyaml{RST}")
        print("  pip3 install pyyaml")
        sys.exit(1)


def load_yaml(path):
    yaml = need_yaml()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_motor_specs(cfg):
    """配置 → 逐电机规格列表。

    兼容两种格式：
      A) 扁平（set_zero.yaml 风格）：motor_id 全局列表，motor_num 按总线分组
      B) 分段（robot.yaml 风格）：同上，但 motor_interface_type/motor_type 逐总线

    返回 [{index, motor_id, interface, interface_type, motor_type,
           motor_model, zero_offset, master_id_offset}]
    """
    m = cfg.get("motors", cfg)

    ids = m["motor_id"]
    ifaces = m["motor_interface"]
    nums = m["motor_num"]
    models = m.get("motor_model", [])
    zeros = m.get("motor_zero_offset", [])

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


def create_motors(motors_py, specs):
    """建实例。⚠️ 同总线同 ID 会静默顶掉前者 —— 本函数会检测重复。"""
    seen = {}
    motors = []
    for s in specs:
        key = (s["interface"], s["motor_id"])
        if key in seen:
            print(f"{RED}✗ 冲突：{s['interface']} 上 ID={s['motor_id']} 重复"
                  f"（index {seen[key]} 与 {s['index']}）{RST}")
            print(f"{YEL}  同一总线同 ID 会静默顶掉前者的回调。请检查配置。{RST}")
            sys.exit(1)
        seen[key] = s["index"]

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
    return motors


def safe_deinit(motors):
    """无论如何都要失能"""
    print("失能中...", end=" ", flush=True)
    for _, mv in motors:
        try:
            mv.deinit_motor()
        except Exception:
            pass
    print(f"{GRN}完成{RST}")


def read_error_clean(mv, settle=0.05):
    """⚠️ 正确读错误码：先清 → 等新帧 → 再读

    原因：【实测】error_id 只增不减（只有错误码 >7 时才写入），
    直接读会永远读到历史错误。
    """
    try:
        mv.clear_motor_error()
        time.sleep(settle)
        mv.refresh_motor_status()
        time.sleep(settle)
        e = mv.get_error_id()
        return int(e) if isinstance(e, (int, float)) else -1
    except Exception:
        return -1


def describe_error(code):
    """DM 错误码 → 人话（err=1 是『使能』不是错误）"""
    table = {
        0: "失能（正常）",
        1: "使能（正常，**不是错误**）",
        8: "超压 OVER_VOLT",
        9: "欠压 UNDER_VOLT",
        10: "过流 OVER_CURRENT",
        11: "MOS 过温 MOS_OVER_TEMP",
        12: "线圈过温 COIL_OVER_TEMP",
        13: "通讯丢失 LOST_CONN",
        14: "过载 OVER_LOAD",
    }
    return table.get(code, f"未知码 {code}")


def confirm(prompt):
    """交互确认。

    ⚠️ 非交互环境（管道 / 重定向 / CI）直接拒绝 —— 避免"没人看着也照样跑"。
    """
    if not sys.stdin.isatty():
        print(f"{RED}✗ 这是需要人工确认的步骤，但当前不是交互终端。{RST}")
        print(f"{YEL}  确认要跳过的话，加 --yes 参数。{RST}")
        sys.exit(2)
    try:
        input(f"{YEL}⚠️  {prompt}{RST}\n   按 Enter 继续（Ctrl+C 取消）... ")
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(0)
