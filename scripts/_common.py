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


def check_tx_alive(iface, n_frames=10, settle=0.02):
    """⭐ TX 假死检测 —— 发 n_frames 帧，看内核 TX 计数是否增长。

    为什么需要（2026-09-17 第二次踩到）：
      gs_usb 在某些情况下会进 TX 假死 —— **发送调用不报错、接口显示 UP/ERROR-ACTIVE、
      bus-off=0，但帧根本不出引脚、内核 TX 计数不增长**。
      后果：
        · 扫描会"全假阴性"（误判成电机坏了/没接）
        · 更坏：以为在下发指令，实际一条没出去 ⇒ 机器人不动却查不出原因
      解卡：`ip link set <iface> down && ip link set <iface> up type can ...`

    ⚠️ 本函数【只发失能帧 FF×7 FD】（read mode），不使能、不产生力矩，安全。

    返回 (是否活, 增长帧数)。设备不存在时返回 (None, -1)。
    """
    import subprocess
    import time as _t
    stat = f"/sys/class/net/{iface}/statistics/tx_packets"

    def _tx():
        try:
            with open(stat) as f:
                return int(f.read().strip())
        except Exception:
            return None

    before = _tx()
    if before is None:
        return None, -1          # 接口不存在（交给上层报错）

    # ⚠️ 接口存在但没 UP ⇒ 必然发不出 ⇒ 直接判假死（不能只靠 TX 计数，那时 cangen 会失败）
    try:
        with open(f"/sys/class/net/{iface}/operstate") as f:
            if f.read().strip() == "down":
                return False, 0
    except Exception:
        pass

    try:
        r = subprocess.run(["cangen", iface, "-I", "7FF", "-L", "8",
                            "-D", "FFFFFFFFFFFFFFFD", "-n", str(n_frames), "-g", "5"],
                           capture_output=True, timeout=10)
        if r.returncode != 0:
            # cangen 跑失败（接口 down / 无权限）⇒ 按假死处理（宁可误报）
            return False, 0
    except Exception:
        return False, 0
    _t.sleep(settle)
    after = _tx()
    grew = (after or 0) - (before or 0)
    return grew >= n_frames, grew


def check_tx_alive_all(cfg_or_specs):
    """对所有用到的接口做 TX 假死检测，有问题就打印醒目告警（不退出，只提示）。"""
    specs = cfg_or_specs if isinstance(cfg_or_specs, list) else build_motor_specs(cfg_or_specs)
    ifaces = sorted({s["interface"] for s in specs})
    bad = []
    print(f"{'接口':<8}{'TX 计数增长':>14}  状态")
    print("-" * 40)
    for i in ifaces:
        alive, grew = check_tx_alive(i)
        if alive is None:
            print(f"{i:<8}{'—':>14}  🔴 接口不存在")
            bad.append(i)
        elif alive:
            print(f"{i:<8}{'+' + str(grew):>14}  ✅ 正常")
        else:
            print(f"{i:<8}{'+' + str(grew):>14}  🔴 TX 假死")
            bad.append(i)
    if bad:
        print()
        print(f"{RED}🔴 检测到 TX 假死：{bad}{RST}")
        print(f"{YEL}  症状：接口显示 UP/ERROR-ACTIVE、bus-off=0、发送不报错，")
        print(f"        但帧不出引脚、TX 计数不增长 ⇒ 扫描会全假阴性。{RST}")
        print(f"{YEL}  解卡：{RST}")
        for i in bad:
            print(f"    sudo ip link set {i} down && \\")
            print(f"    sudo ip link set {i} up type can bitrate 1000000 "
                  f"dbitrate 5000000 fd on loopback off")
    return bad


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
