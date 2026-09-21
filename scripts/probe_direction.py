#!/usr/bin/env python3
"""probe_direction.py — 电机方向符号验证探针（非交互）

目的
----
`robots/dm10/robot.yaml` 里的 `motor_sign: [1]*10` 是一个**从未验证过的假设**
（"所有电机都没装反"）。本脚本用小幅度、低增益的 MIT 指令把**单个关节**推一下，
把可观测的数字摆出来，供操作员肉眼判定方向，进而定案 `motor_sign` 该填几。

⚠️⚠️ 三个设计要点（都不是可选的）
---------------------------------
① **必须把 --assumed-sign 乘进下发值**。
   生产路径（robot_interface.cpp:385-392）是 `motor_mit_cmd(target * motor_sign, ...)`。
   `motor_mit_cmd` 是**电机原始坐标系**的接口；如果探针直接呼它，就**绕过了 motor_sign**，
   于是不管电机装没装反、看起来都"对" —— 测了等于没测。

② **增益用达妙官方起点值**：kp=20 / kd=3（`docs/03-单关节稳停/0-index.md`）。
   ⚠️ 不要自己拍增益。文档明说：「如果某个关节的 kp 需要调到和别的关节差很多才能稳，
   那多半不是 kp 的问题 —— 是零位错了、方向反了、或机械有别劲」，应回去查标定。
   ⚠️ kd 严禁为 0（说明书：会震荡失控）。

④ **⭐ 使能时序**（2026-09-04 实测，见 dm-dual-motor-test 提交 8e64d09）：
   切控制模式会清除使能状态 ⇒ 必须【先失能 → 切 MIT → 再使能 → 保持校验】。
   先使能后切模式会导致电机从未真正使能，且"反馈状态≠1 被静默忽略"。
   ⚠️ 因此本脚本【不用 init_motor()】，改为手动走这四步。

⑤ **⭐ 不能用 mv.get_error_id() 判断使能**：驱动 `dm_motor_driver.cpp:238`
   只在错误码 >7 时才写 error_id_，而 err=1（使能）不 >7 ⇒ 它永远是初值 0。
   本脚本用 BusSniffer 自己解反馈帧的 D[0] 高 4 位。

③ **髋 roll / 髋 yaw 视觉上分不清**（模型数据：roll 脚横移 158mm，yaw 仅 8mm，但都让腿横扫）。
   两个关节都要**同时看位移和朝向**才判得准。脚本会把模型期望的两个量都打出来。

安全
----
- 会**使能**目标电机并**让它真的动**（幅度 ±0.03 rad ≈ ±1.7°）
- 必须 --confirm，否则只读不写（dry-run）
- 机器人必须**吊着**、操作员在旁、Ctrl-C 能立刻失能
- try/finally 保证任何异常路径都会失能，不留电机带电

用法
----
    # 先看计划（dry-run，不动）
    python3 scripts/probe_direction.py --bus can2 --motor-id 2
    # ⭐ 先只验证"照新时序能否保持使能"（不做方向测试）
    python3 scripts/probe_direction.py --bus can2 --motor-id 2 --enable-check --confirm
    # 真跑单台（⭐ 推荐从右膝开始）
    python3 scripts/probe_direction.py --bus can2 --motor-id 2 --confirm
    # 跑完 10 台
    python3 scripts/probe_direction.py --all --confirm
"""
import argparse
import datetime as _dt
import json
import os
import statistics
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

# ⚠️ 硬上限：任何情况下不允许超过这个幅度。
#   2026-09-17 从 0.05 提到 0.25：实测发现关节有 ~0.58 N·m 的恒定阻力，
#   幅度 0.03 时 kp×幅度 = 0.6 N·m 刚好卡在平衡点 ⇒ 位置几乎不动。
#   ⭐ 加大【幅度】比加大 kp 好：kp 保持在官方值(16~20)可避开振荡，
#   而 kp×幅度 同步变大。0.25 rad ≈ 14°，仍在右膝限位 [-0.3, 2.3562] 内。
#   ⚠️ 上限保留（不是取消）：防止误传大值把关节猛推。
AMP_HARD_CAP = 0.25  # rad ≈ 14.3°

# 策略关节顺序（= robot.yaml 里 motor_id / motor_sign 的索引顺序）
# ⚠️ 2026-09-17：接口名不再写死 —— 从 robot.yaml 的 motor_interface 读。
#   原因：接口名会随 USB 枚举漂移（can1/can2 ↔ can0/can1），而 robot.yaml 是唯一真源。
#   写死的表曾在接口改名后给出错误提示（"配置里没有 can0 ID=2"），险些误导结论。
_JOINTS = [
    ("左腿 髋pitch", "leg_l1_joint"), ("左腿 髋roll", "leg_l2_joint"),
    ("左腿 髋yaw", "leg_l3_joint"), ("左腿 膝", "leg_l4_joint"),
    ("左腿 踝", "leg_l5_joint"),
    ("右腿 髋pitch", "leg_r1_joint"), ("右腿 髋roll", "leg_r2_joint"),
    ("右腿 髋yaw", "leg_r3_joint"), ("右腿 膝", "leg_r4_joint"),
    ("右腿 踝", "leg_r5_joint"),
]
_MOTOR_ID = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]


def _load_order():
    """从 robot.yaml 读 motor_interface/motor_num → 构造 (bus, id, name, joint) 列表。"""
    import yaml
    try:
        with open(ROBOT_YAML_DEFAULT, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ifaces = cfg["motors"]["motor_interface"]
        nums = cfg["motors"]["motor_num"]
    except Exception:
        # 兜底：用当前常见值
        ifaces, nums = ["can0", "can1"], [5, 5]
    out, k = [], 0
    for bus, n in zip(ifaces, nums):
        for _ in range(int(n)):
            if k >= len(_JOINTS):
                break
            name, joint = _JOINTS[k]
            out.append((bus, _MOTOR_ID[k], name, joint))
            k += 1
    return out


ORDER = _load_order()


# 策略关节顺序（= robot.yaml 里 motor_id / motor_sign 的索引顺序）—— 用于查 motor_sign
POLICY_ORDER = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
]
ROBOT_YAML_DEFAULT = os.path.expanduser(
    "~/roboparty_deploy/src/inference/robots/dm10/robot.yaml")


def _motor_sign_for(joint, args):
    """取该策略关节的 motor_sign。

    ⚠️ 为什么必须用它：生产路径是 `motor_mit_cmd(关节角目标 × motor_sign)`。
    限位检查要用【配置限位的关节角】当目标，就必须乘上 motor_sign 换算成电机指令；
    否则（motor_sign=−1 时）会跑到【相反方向】，测的不是配置的那个限位。
    2026-09-17 实测踩过这个坑（右膝被当成"+2.206 上限"测，实际关节角是 −2.206）。
    """
    path = args.robot_yaml or ROBOT_YAML_DEFAULT
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ms = cfg["robot"]["motor_sign"]
        idx = POLICY_ORDER.index(joint)
        return int(ms[idx])
    except Exception as exc:
        print(f"  {RED}✗ 读 motor_sign 失败（{path}）：{exc}{RST}")
        return 0


def _load_model_directions(path):
    """读模型期望方向表（export_model_directions.py 的产物）。读不到就返回空。"""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        return {e["joint"]: e for e in doc.get("entries", [])}
    except Exception:
        return {}


def _describe_expect(entry):
    """把模型期望翻译成操作员能对照的一行话"""
    if not entry:
        return "（未找到模型期望值）"
    dp = entry.get("foot_pos_delta_mm", [0, 0, 0])
    rp = entry.get("foot_rot_rpy_deg", [0, 0, 0])
    parts = []
    # 机体坐标系：x+ = 前, y+ = 左, z+ = 上（见 dm10.xml / export_model_directions.py）
    for val, name, pos_word, neg_word in (
        (dp[0], "前后", "后", "前"),   # ⚠️ x+ 是【前】⇒ 负值才是【后】。此表按【数值符号】给词
        (dp[1], "左右", "右", "左"),   # ⚠️ y+ 是【左】⇒ 同理反着给
    ):
        if abs(val) >= 4.0:
            parts.append(f"脚往【{pos_word if val < 0 else neg_word}】{abs(val):.0f}mm")
    for val, axis in ((rp[0], "侧倾"), (rp[1], "前后倾"), (rp[2], "转向")):
        if abs(val) >= 2.0:
            parts.append(f"脚{axis}{val:+.0f}°")
    tail = "（左右**同向**）" if entry.get("expect") == "左右同号" else "（左右**镜像**）"
    return "; ".join(parts) + "  " + tail if parts else "(近乎不动)  " + tail


# ⚠️⚠️ 必须自己解反馈帧，不能用 mv.get_error_id()
#   驱动 `dm_motor_driver.cpp:238` 只在高4位 >7 时才写 error_id_，
#   而 err=1（使能）不 >7 ⇒ error_id_ 永远是初值 0 ⇒ get_error_id() 恒返回 0。
#   （这也是 init_motor() 返回值不可信的原因：它的 switch 命中 DM_DOWN=0。）
#   官方手册 V1.3 第 13 页反馈帧布局：
#     D[0] = ERR(高4位) | 电机ID(低4位)
#     D[1..2] = POS 16bit 【大端】，0x8000 = 0 rad
#     D[3..4] = VEL 12bit，  D[4..5] = TAU 12bit
POS_MAX, VEL_MAX, TAU_MAX = 12.5, 10.0, 28.0   # DM4340P_24V，见 dm_motor_driver.cpp:24


def _u2f(raw, bits, vmax):
    return raw / float((1 << bits) - 1) * (2 * vmax) - vmax


def parse_feedback(hexs):
    """解一帧反馈。hexs 形如 '12 80 01 7F F7 FF 22 1F'。返回 dict 或 None。"""
    try:
        d = [int(x, 16) for x in hexs.split()]
    except Exception:
        return None
    if len(d) < 8:
        return None
    return {
        "err": d[0] >> 4,
        "motor_id": d[0] & 0x0F,
        "pos": _u2f((d[1] << 8) | d[2], 16, POS_MAX),
        "vel": _u2f(((d[3] & 0x0F) << 8) | (d[4] >> 4), 12, VEL_MAX),
        "tau": _u2f(((d[4] & 0x0F) << 8) | d[5], 12, TAU_MAX),
        "t_mos": d[6], "t_rotor": d[7],
        "raw": hexs,
    }


class BusSniffer:
    """后台跑 candump，收集反馈帧（仲裁 ID 应为 master=0）。"""

    def __init__(self, iface="can2", motor_id=None):
        self.iface = iface
        self.motor_id = motor_id
        self.lines = []
        self._p = None
        self._t = None

    def start(self):
        import subprocess
        import threading

        def run():
            try:
                self._p = subprocess.Popen(
                    ["candump", "-t", "d", self.iface],
                    stdout=subprocess.PIPE, text=True, bufsize=1)
                for line in self._p.stdout:
                    self.lines.append(line.rstrip())
            except Exception:
                pass

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        time.sleep(0.8)      # 让 candump 起来

    def stop(self):
        try:
            if self._p:
                self._p.kill()
        except Exception:
            pass

    def latest(self):
        """返回最后一帧【匹配本电机 ID 的】反馈。"""
        import re
        for line in reversed(self.lines):
            m = re.search(r"can\d+\s+([0-9A-F]{3,8})\s+\[(\d+)\]\s+([0-9A-F ]+)$", line)
            if not m:
                continue
            hexs = m.group(3).strip()
            f = parse_feedback(hexs)
            if f and (self.motor_id is None or f["motor_id"] == self.motor_id):
                return f
        return None

    def all_for(self, motor_id):
        import re
        out = []
        for line in self.lines:
            m = re.search(r"can\d+\s+([0-9A-F]{3,8})\s+\[(\d+)\]\s+([0-9A-F ]+)$", line)
            if not m:
                continue
            f = parse_feedback(m.group(3).strip())
            if f and f["motor_id"] == motor_id:
                out.append(f)
        return out


def enable_and_hold(mv, motors_py, kp, kd, hold_s, sniffer, quiet=False):
    """⭐ 照 dm-dual-motor-test 已验证的时序使能并保持。

    ⚠️ 关键顺序（2026-09-04 实测，见 dm-dual-motor-test 提交 8e64d09）：
       切控制模式会清除使能状态 ⇒ 必须在【失能状态下】切模式，切换后再使能。
       原流程（先使能后切模式）会导致电机从未真正使能，且"反馈状态≠1 被静默忽略"。

    返回 (是否保持住, err 序列, 最后一帧)
    """
    # ① 先失能
    mv.unlock_motor()
    time.sleep(0.2)
    # ② ⭐ 在失能状态下切 MIT
    mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
    time.sleep(0.3)
    # ③ 再使能
    mv.lock_motor()
    time.sleep(0.3)

    # ④ 保持维持：5ms 一帧，边发边核 err
    errs = []
    t0 = time.time()
    while time.time() - t0 < hold_s:
        mv.motor_mit_cmd(0.0, 0.0, kp, kd, 0.0)
        time.sleep(0.005)
        f = sniffer.latest()
        if f:
            errs.append(f["err"])
    ok = bool(errs) and errs[-1] == 1
    return ok, errs, sniffer.latest()


def _enable_check_one(motors_py, bus, motor_id, name, joint, args):
    """⭐ 只验证「照新时序能否保持使能」。不下发任何位置指令、不产生运动。

    对照 dm-dual-motor-test 的保持阶段：连发 MIT 帧 2s，自己解反馈核 err==1。
    """
    print()
    print("=" * 78)
    print(f"  使能保持验证 · {name}   [{bus}  ID={motor_id}]")
    print("=" * 78)
    print(f"  ⚠️ 只发 kp={args.kp} 的【零目标】MIT 帧维持使能，不产生运动")

    mv = motors_py.MotorDriver.create_motor(
        motor_id=motor_id, interface_type="can", interface=bus, motor_type="DM",
        motor_model=args.model, master_id_offset=args.master_id_offset,
        motor_zero_offset=0.0,
    )
    sniffer = BusSniffer(bus, motor_id)
    rec = {"bus": bus, "motor_id": motor_id, "joint_name": name,
           "policy_joint": joint, "phase": "enable-check"}
    try:
        sniffer.start()
        mv.unlock_motor()
        time.sleep(0.25)
        mv.refresh_motor_status()
        time.sleep(0.25)
        f0 = sniffer.latest()
        rec["baseline"] = f0
        print(f"  ① 失能态   " + (f"pos={f0['pos']:+.5f} err={f0['err']}"
                                  if f0 else "⚠️ 没抓到帧"))

        print(f"  ② 时序：unlock → set_mode(MIT)【失能态下】→ lock → 保持 2s")
        ok, errs, last = enable_and_hold(
            mv, motors_py, args.kp, args.kd, 2.0, sniffer)
        rec["enable_ok"] = bool(ok)
        rec["errs"] = errs
        rec["last_frame"] = last

        if errs:
            uniq = sorted(set(errs))
            print(f"  err 序列：共 {len(errs)} 帧，取值 {uniq}，末值 {errs[-1]}")
        if last:
            print(f"  ③ 保持结果 pos={last['pos']:+.5f} rad  "
                  f"tau={last['tau']:+.4f} N·m  "
                  f"t_mos={last['t_mos']}°C  t_rotor={last['t_rotor']}°C")
            print(f"     原始帧 {last['raw']}")

        if ok:
            print(f"  {GRN}✅ 使能保持成功（err=1）—— 时序正确，可以跑方向测试{RST}")
        else:
            print(f"  {RED}❌ 使能没保持住（err≠1）—— 时序仍不对{RST}")
            print(f"  {YEL}  查：切模式是否在失能态 / 使能帧是否发出 / 供电 / 是否需先清错{RST}")
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  {RED}✗ 异常 {type(exc).__name__}: {exc}{RST}")
    finally:
        sniffer.stop()
        try:
            mv.unlock_motor()
        except Exception:
            pass
        try:
            mv.deinit_motor()
        except Exception:
            pass
        print("  （已失能收尾）")
    return rec


def _median_sample(mv, settle, samples, gap=0.08):
    """等停稳后连采几个点，取中位数（抗单帧毛刺）"""
    time.sleep(settle)
    vals, vels = [], []
    for _ in range(max(1, samples)):
        try:
            mv.refresh_motor_status()
            time.sleep(gap)
            vals.append(float(mv.get_motor_pos()))
            vels.append(abs(float(mv.get_motor_spd())))
        except Exception:
            pass
    if not vals:
        return None, None
    return statistics.median(vals), (max(vels) if vels else 0.0)


# ⚠️⚠️ 达妙 MIT 需要【持续接收指令】才持续出力。
#   `set_zero.py`（今天上午成功点零 10 台）的写法是 while 循环里每 50ms 发一次；
#   而"发一次 → sleep(0.5)"会让电机只在一个控制周期内出力，然后回到无力状态
#   ⇒ 症状是：力矩读数 ≈ 0（噪声级）、位置纹丝不动，看着像"电机不出力/没使能"。
#   本函数就是那个循环：在 hold_s 秒内持续下发同一目标（50ms 周期）。
HOLD_DT = 0.05


def hold_cmd(mv, target, kp, kd, hold_s):
    """在 hold_s 秒内以 50ms 周期持续下发 MIT 指令。返回实际下发的帧数。"""
    n = max(1, int(round(hold_s / HOLD_DT)))
    for _ in range(n):
        mv.motor_mit_cmd(target, 0.0, kp, kd, 0.0)
        time.sleep(HOLD_DT)
    return n


def sample_while_holding(mv, target, kp, kd, hold_s, samples=5, sniffer=None):
    """边持续下发边采样，返回 (位置中位数, 速度峰值, 力矩中位数)。

    ⚠️ 与 _median_sample 的区别：本函数【在下发过程中】读，
    所以能采到"电机正在出力"时的真实状态，而不是把指令停了再看（那时已经无力了）。
    ⚠️ 传了 sniffer 时，位置/力矩取自【自己解出的反馈帧】，不用驱动缓存
       （驱动 get_error_id() 有 bug，且缓存值可能滞后，见文件头要点⑤）。
    """
    mv.motor_mit_cmd(target, 0.0, kp, kd, 0.0)
    time.sleep(min(0.3, hold_s))          # 先让它动起来
    vals, vels, taus = [], [], []
    n = max(1, samples)
    for _ in range(n):
        mv.motor_mit_cmd(target, 0.0, kp, kd, 0.0)
        time.sleep(HOLD_DT)
        if sniffer is not None:
            f = sniffer.latest()
            if f:
                vals.append(f["pos"])
                vels.append(abs(f["vel"]))
                taus.append(f["tau"])
            continue
        try:
            mv.refresh_motor_status()
            time.sleep(0.02)
            vals.append(float(mv.get_motor_pos()))
            vels.append(abs(float(mv.get_motor_spd())))
            taus.append(float(mv.get_motor_current()))
        except Exception:
            pass
    if not vals:
        return None, None, None
    return (statistics.median(vals),
            max(vels) if vels else 0.0,
            statistics.median(taus) if taus else 0.0)


def run_one(motors_py, bus, motor_id, name, joint, args, model_dirs, results, out_path):
    """测一台。返回 dict 结论。"""
    assumed = args.assumed_sign
    amp = args.amplitude
    # ⭐ --use-config-sign：用 robot.yaml 里该关节的 motor_sign 覆盖 assumed_sign
    #    这样 motor_mit_cmd(q0 + motor_sign × amp) 就是【生产路径】的下发值，
    #    验证的是"配置本身"对不对。
    if getattr(args, "use_config_sign", False):
        cs = _motor_sign_for(joint, args)
        if cs == 0:
            print(f"  {RED}✗ 取不到 {joint} 的 motor_sign，跳过{RST}")
            return {"joint_name": name, "verdict": "跳过（取不到 motor_sign）"}
        assumed = cs
        print(f"  ⭐ 用 robot.yaml 的 motor_sign = {cs:+d}（上机前复核：走生产路径语义）")

    print()
    print("=" * 78)
    print(f"  {name}   [{bus}  ID={motor_id}]   策略关节 = {joint}")
    print("=" * 78)
    entry = model_dirs.get(joint)
    print(f"  模型期望（给正角时）: {_describe_expect(entry)}")
    print(f"  实测将给: assumed_sign={assumed:+d} → 下发 "
          f"{assumed * amp:+.4f} rad / {assumed * -amp:+.4f} rad")
    print()

    rec = {
        "bus": bus, "motor_id": motor_id, "joint_name": name, "policy_joint": joint,
        "assumed_sign": assumed, "amplitude_rad": amp, "kp": args.kp, "kd": args.kd,
        "model_expect": _describe_expect(entry),
        "model_foot_pos_delta_mm": (entry or {}).get("foot_pos_delta_mm"),
        "model_foot_rot_rpy_deg": (entry or {}).get("foot_rot_rpy_deg"),
    }

    mv = motors_py.MotorDriver.create_motor(
        motor_id=motor_id, interface_type="can", interface=bus, motor_type="DM",
        motor_model=args.model, master_id_offset=args.master_id_offset,
        motor_zero_offset=0.0,
    )

    enabled = False
    try:
        # ① 失能态基线（安全）
        mv.unlock_motor()
        time.sleep(args.settle)
        mv.refresh_motor_status()
        time.sleep(args.settle)
        base = float(mv.get_motor_pos())
        rec["baseline_rad"] = base
        print(f"  ① 失能基线      pos = {base:+.5f} rad ({base * 57.29578:+.2f}°)")

        # ② 使能 + MIT（⭐ 照 dm-dual-motor-test 已验证的时序）
        sniffer = BusSniffer(bus, motor_id)
        sniffer.start()
        mv.unlock_motor()
        time.sleep(0.2)
        mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)   # ⭐ 失能态下切模式
        time.sleep(0.3)
        mv.lock_motor()                                             # 再使能
        enabled = True
        time.sleep(0.3)

        # 保持阶段：5ms 一帧维持使能，并自己解 err 校验
        ok_hold, errs, last = enable_and_hold(
            mv, motors_py, args.kp, args.kd, 2.0, sniffer)
        rec["enable_errs_tail"] = errs[-8:] if errs else []
        rec["enable_ok"] = bool(ok_hold)
        hold_f = last
        if hold_f:
            rec["enabled_hold_rad"] = hold_f["pos"]
            rec["enabled_hold_tau"] = hold_f["tau"]
            print(f"  ② 使能保持      pos = {hold_f['pos']:+.5f} rad  "
                  f"err = {hold_f['err']} ({'✅ 已使能' if hold_f['err'] == 1 else '❌ 未使能'})  "
                  f"力矩 {hold_f['tau']:+.4f} N·m  "
                  f"[err 序列尾 {errs[-5:] if errs else '无帧'}]")
        else:
            print(f"  ② 使能保持      ⚠️ 没抓到反馈帧")

        if not ok_hold:
            rec["verdict"] = "不可判（使能未保持住）"
            print(f"  {RED}✗ 使能没保持住（err≠1）⇒ 时序仍不对，本台跳过{RST}")
            print(f"  {YEL}  查：切模式是否在失能态 / 使能帧是否发出 / 供电{RST}")
            return rec

        # ③ 确定基准：⭐ 用【当前实测位置】，不用 0
        # ⚠️ 这个关节的静止位不是 0（失能基线就可能是 +0.12 之类）。
        #    照 direction_check.py 的做法：目标 = 当前位置 ± delta（相对增量）。
        f_base = sniffer.latest()
        q0 = f_base["pos"] if f_base else rec.get("baseline_rad") or 0.0
        rec["q0_rad"] = q0
        print(f"  ③ 基准 q0 = {q0:+.5f} rad ({q0 * 57.29578:+.2f}°)  "
              f"（相对此位置给 ±{amp}）")

        # ⭐ --toward-limit：走到接近配置限位，验证行程/软限位
        if args.toward_limit:
            rec["mode"] = "toward-limit"
            lim = (entry or {}).get("limit_rad")
            if not lim:
                rec["verdict"] = "跳过（模型表里没有该关节的 limit_rad）"
                print(f"  {YEL}⚠️ 模型表里没有 {joint} 的 limit_rad，跳过{RST}")
                return rec
            lo_cfg, hi_cfg = float(lim[0]), float(lim[1])
            margin = args.margin
            sides = {"lo": ["lo"], "hi": ["hi"], "both": ["lo", "hi"]}[args.toward_limit]
            rec["limits_configured"] = [lo_cfg, hi_cfg]
            rec["margin_rad"] = margin

            # ⚠️⚠️ motor_sign 换算 —— 2026-09-17 修正（原版漏了这一步，导致测错方向）
            #
            #   生产路径：motor_mit_cmd(关节角目标 × motor_sign)   (robot_interface.cpp:389)
            #   ⇒ 要用【配置限位的关节角】当目标，必须换算成电机坐标系指令：
            #         电机指令 = 关节角目标 × motor_sign
            #
            #   原版直接把 lo_cfg/hi_cfg 当电机指令下发 ⇒ 对 motor_sign=−1 的关节
            #   实际跑到【相反方向】，测的根本不是配置的那个限位。
            #   （右膝就是这样：脚本以为在测 "+2.206 上限"，实际关节角是 −2.206。）
            sign = _motor_sign_for(joint, args)
            rec["motor_sign_used"] = sign
            if sign == 0:
                rec["verdict"] = "跳过（取不到该关节的 motor_sign）"
                print(f"  {YEL}⚠️ 取不到 {joint} 的 motor_sign，跳过{RST}")
                return rec
            print(f"  ③ 基准 q0 = {q0:+.5f} rad   配置限位(关节角) [{lo_cfg:+.3f}, {hi_cfg:+.3f}]"
                  f"   motor_sign = {sign:+d}   余量 {margin}")

            # ⭐ 自适应余量：窄行程方向按比例收小，否则目标离起点太近、测不到东西。
            #    （2026-09-17 实测：右膝"往前伸"行程仅 17.2°，margin=0.15 时只走到 4.0°。）
            #    规则：该方向行程的 20% 与 args.margin 取小，但不小于 0.03。
            span_lo = lo_cfg        # 从 0 到下限的距离（模型零位=直腿，基准约在 0）
            span_hi = hi_cfg
            m_lo = max(0.03, min(margin, abs(span_lo) * 0.2))
            m_hi = max(0.03, min(margin, abs(span_hi) * 0.2))
            rec["margin_lo"] = round(m_lo, 4)
            rec["margin_hi"] = round(m_hi, 4)
            print(f"  自适应余量: lo {m_lo:.3f} (行程{abs(span_lo)*57.3:.1f}°)"
                  f"  hi {m_hi:.3f} (行程{abs(span_hi)*57.3:.1f}°)")

            all_ok = True
            for side in sides:
                # ① 先算【关节角】目标（各方向用自己的余量）
                mm = m_lo if side == "lo" else m_hi
                ang_tgt = (lo_cfg + mm) if side == "lo" else (hi_cfg - mm)
                # ② 再换算成【电机坐标系】指令
                tgt = ang_tgt * sign
                travel = abs(tgt - q0)
                rec[f"target_angle_{side}"] = ang_tgt
                rec[f"target_{side}"] = tgt
                print(f"\n  ④[{side}] 关节角目标 {ang_tgt:+.4f}"
                      f"  → 电机指令 {tgt:+.4f}（距基准 {travel:.3f} rad）")

                # ⭐ 平滑插值过去（不阶跃），再停住采样
                ramp_s = max(1.0, args.ramp)
                steps = max(1, int(ramp_s / HOLD_DT))
                for k in range(steps):
                    q = q0 + (tgt - q0) * (k + 1) / steps
                    mv.motor_mit_cmd(q, 0.0, args.kp, args.kd, 0.0)
                    time.sleep(HOLD_DT)
                pos_r, vel_r, tau_r = sample_while_holding(
                    mv, tgt, args.kp, args.kd, args.settle_limit, args.samples, sniffer)
                rec[f"reached_{side}"] = pos_r
                rec[f"tau_{side}"] = tau_r
                dlt = (pos_r - q0) if pos_r is not None else None
                rec[f"delta_{side}"] = dlt

                # ⭐ 三条判据（防"没动也报成功"）
                TAU_LSB = 2 * 28.0 / 4095.0
                moved = (dlt is not None and abs(dlt) > 0.3 * travel)
                powered = tau_r is not None and abs(tau_r) > 3 * TAU_LSB
                f_now = sniffer.latest()
                err_now = f_now["err"] if f_now else None
                no_fault = (err_now is None) or (err_now < 8)
                rec[f"moved_{side}"] = bool(moved)
                rec[f"powered_{side}"] = bool(powered)
                rec[f"err_{side}"] = err_now

                print(f"     实际 {pos_r:+.5f} rad  Δ={dlt:+.5f} ({dlt * 57.29578:+.2f}°)  "
                      f"力矩 {tau_r:+.4f} N·m  err={err_now}")
                print(f"     判据: 移动{'✅' if moved else '❌'}"
                      f"  出力{'✅' if powered else '❌'}"
                      f"  无故障{'✅' if no_fault else '🔴'}")
                if not no_fault:
                    print(f"  {RED}🔴 检测到故障码 {err_now} —— 停止本关节{RST}")
                    all_ok = False
                    break
                if not (moved and powered):
                    all_ok = False

                # 回基准再走另一边
                hold_cmd(mv, q0, args.kp, args.kd, max(1.0, args.ramp))
                q0 = (sniffer.latest() or {}).get("pos", q0)

            rec["verdict"] = ("到达限位附近（行程/出力/无故障均通过）" if all_ok
                              else "异常（见 moved/powered/err）")
            print(f"\n  {GRN if all_ok else YEL}▶ {rec['verdict']}{RST}")
            return rec

        # ⭐ --one-way：只做一个方向，保持住让操作员看清，再回位
        if args.one_way is not None:
            d = args.one_way
            tgt = q0 + assumed * d * amp
            rec["one_way"] = d
            rec["target_rad"] = tgt
            print(f"  ④ 只做【{'正' if d > 0 else '负'}】方向：目标 {tgt:+.5f} rad")
            pos_1, vel_1, tau_1 = sample_while_holding(
                mv, tgt, args.kp, args.kd, args.hold, args.samples, sniffer)
            rec["pos_rad"] = pos_1
            rec["tau_nm"] = tau_1
            rec["delta_from_q0"] = (pos_1 - q0) if pos_1 is not None else None
            print(f"     实际停在 {pos_1:+.5f} rad ({pos_1 * 57.29578:+.2f}°)  "
                  f"相对基准 {rec['delta_from_q0']:+.5f} rad "
                  f"({rec['delta_from_q0'] * 57.29578:+.2f}°)  力矩 {tau_1:+.4f} N·m")

            print()
            print(f"  {'=' * 70}")
            print(f"  👀 保持不动 {args.hold_view:.0f} 秒 —— 请看这条腿往哪动了")
            print(f"  {'=' * 70}")
            print(f"  模型期望：给{'正' if d > 0 else '负'}角时 → {_describe_expect(entry)}")
            hold_cmd(mv, tgt, args.kp, args.kd, args.hold_view)
            f_after = sniffer.latest()
            if f_after:
                print(f"  （保持结束时 pos={f_after['pos']:+.5f} "
                      f"tau={f_after['tau']:+.4f}）")

            print(f"  ⑤ 回位到 q0 = {q0:+.5f}")
            hold_cmd(mv, q0, args.kp, args.kd, args.hold)
            rec["verdict"] = "已单向测试（方向待人工确认）"
            return rec

        # ④ 正向：目标 = q0 + assumed*amp
        tgt_p = q0 + assumed * amp
        pos_p, vel_p, tau_p = sample_while_holding(
            mv, tgt_p, args.kp, args.kd, args.hold, args.samples, sniffer)

        # ⑤ 反向：目标 = q0 − assumed*amp
        tgt_n = q0 - assumed * amp
        pos_n, vel_n, tau_n = sample_while_holding(
            mv, tgt_n, args.kp, args.kd, args.hold, args.samples, sniffer)

        rec["target_plus_rad"] = tgt_p
        rec["target_minus_rad"] = tgt_n
        rec["pos_plus_rad"] = pos_p
        rec["pos_minus_rad"] = pos_n
        rec["vel_peak_plus"] = vel_p
        rec["vel_peak_minus"] = vel_n
        rec["tau_median_plus"] = tau_p
        rec["tau_median_minus"] = tau_n
        rec["delta_rad"] = (pos_p - pos_n) if (pos_p is not None and pos_n is not None) else None

        print(f"  ④ 目标 {tgt_p:+.5f} → 实际 {pos_p:+.5f} rad "
              f"({pos_p * 57.29578:+.2f}°)  力矩 {tau_p:+.4f} N·m")
        print(f"  ⑤ 目标 {tgt_n:+.5f} → 实际 {pos_n:+.5f} rad "
              f"({pos_n * 57.29578:+.2f}°)  力矩 {tau_n:+.4f} N·m")
        if rec["delta_rad"] is not None:
            print(f"     Δpos = {rec['delta_rad']:+.5f} rad "
                  f"({rec['delta_rad'] * 57.29578:+.2f}°)  "
                  f"（期望 ≈ {assumed * 2 * amp:+.5f}）")

        # 力矩量化步长（DM4340P_24V：TauMax=28，12bit）—— 低于它就是在噪声里
        TAU_LSB = 2 * 28.0 / 4095.0
        tau_peak = max(abs(tau_p or 0.0), abs(tau_n or 0.0))
        rec["tau_lsb_nm"] = TAU_LSB
        rec["tau_above_noise"] = bool(tau_peak > 3 * TAU_LSB)
        print(f"     力矩量化步长 = {TAU_LSB:.4f} N·m；峰值 {tau_peak:.4f} "
              f"({'✅ 有明显出力' if rec['tau_above_noise'] else '⚠️ 仍在噪声级 ⇒ 可能没出力'})")

        # 一致性：Δ 应与 assumed * 2 * amp 同号。（q0+Δ 与 q0−Δ 的差 = 2Δ）
        expect_delta = assumed * 2 * amp
        # "动了"的判据：Δ 至少 0.3 倍期望（远超读数噪声 ~0.0004）
        moved = (rec["delta_rad"] is not None
                 and abs(rec["delta_rad"]) > 0.3 * abs(expect_delta))
        rec["position_moved"] = bool(moved)
        if rec["delta_rad"] is None:
            rec["verdict"] = "不可判（读数缺失）"
        else:
            same_dir = (rec["delta_rad"] * expect_delta) > 0
            mag_ok = abs(rec["delta_rad"]) > 0.5 * abs(expect_delta)
            rec["same_direction_as_assumed"] = bool(same_dir)
            rec["magnitude_ok"] = bool(mag_ok)
            if not rec["tau_above_noise"]:
                rec["verdict"] = "不可判（力矩仍在噪声级，电机没出力）"
                print(f"  {RED}✗ 力矩没超过噪声级 ⇒ 电机没出力，本台不可判{RST}")
                print(f"  {YEL}  查：MIT 是否持续下发 / 使能时序 / 供电{RST}")
            elif not moved:
                rec["verdict"] = "不可判（位置没动）"
                print(f"  {RED}✗ 力矩有输出但位置几乎没变（Δ={rec['delta_rad']:+.5f} rad）{RST}")
                print(f"  {YEL}  可能是机械干涉 / 顶到限位 / 负载过大。别急着调 kp，先查机械{RST}")
            else:
                rec["verdict"] = "可判（读数有效）" if same_dir else "存疑（Δ 与 assumed 反向）"
                if not same_dir:
                    print(f"  {YEL}⚠️ Δpos 与 assumed_sign 【反向】"
                          f"⇒ 这正是「电机装反了」的信号，motor_sign 该取 {assumed * -1:+d}{RST}")

        print()
        print(f"  {GRN}▶ 现在请【肉眼确认】这条腿往哪动了{RST}")
        print(f"    给 +{assumed * amp:+.4f} 时脚往哪走 —— 对照上面「模型期望」那一行：")
        print(f"    · 方向【一致】⇒ motor_sign 保持 {assumed:+d}")
        print(f"    · 方向【相反】⇒ motor_sign 应为 {assumed * -1:+d}")
        print(f"    提示：髋roll 看脚【横移+侧倾】，髋yaw 看脚【转向但几乎不移动】 —— 两者别混。")

        # ⑥ 回原位（⭐ 回 q0，不是回 0 —— 这个关节的静止位不是 0）
        hold_cmd(mv, q0, args.kp, args.kd, args.hold)

    except Exception as exc:
        rec["verdict"] = f"异常：{type(exc).__name__}: {exc}"
        print(f"  {RED}✗ 异常 {type(exc).__name__}: {exc}{RST}")
    finally:
        try:
            sniffer.stop()
        except Exception:
            sniffer = None
        # ⑦ 无论如何都要失能
        try:
            hold_cmd(mv, 0.0, 0.0, 0.0, 0.1)
        except Exception:
            pass
        try:
            mv.unlock_motor()
            enabled = False
        except Exception:
            pass
        try:
            mv.deinit_motor()
        except Exception:
            pass
        rec["disarmed"] = not enabled
        print(f"  （已失能收尾）")
        results.append(rec)
        _dump(results, out_path)

    return rec


def _dump(results, out_path):
    """⚠️ 追加合并，不覆盖 —— 2026-09-17 实测教训：原先每次运行都覆盖同一文件，
    跑了 10 次只留下 1 条记录，前面 9 台的数据全丢。"""
    if not out_path:
        return
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        old = []
        if os.path.exists(out_path):
            try:
                with open(out_path, encoding="utf-8") as f:
                    old = json.load(f).get("results", [])
            except Exception:
                old = []
        # 去重键：同一电机的同一次测量（bus+id+方向+幅度），保留最新
        merged = {}
        for r in old + list(results):
            k = (r.get("bus"), r.get("motor_id"), r.get("one_way"),
                 r.get("amplitude_rad"), r.get("assumed_sign"), r.get("kp"))
            merged[k] = r
        doc = {
            "schema": "dm10-direction-probe/1",
            "updated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "note": ("verdict 只表示『读数是否可用于判定方向』；"
                     "最终 motor_sign 需人工对照模型期望回填。"
                     "本文件为【累积】记录，同键取最新。"),
            "results": list(merged.values()),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        print(f"  {YEL}⚠️ 落盘失败：{exc}{RST}")


def main():
    ap = argparse.ArgumentParser(description="电机方向符号验证探针")
    ap.add_argument("--bus", help="总线名，如 can2（与 --motor-id 搭配）")
    ap.add_argument("--motor-id", type=int, help="电机 CAN ID")
    ap.add_argument("--all", action="store_true", help="按策略顺序跑全部 10 台")
    ap.add_argument("--model", type=int, default=2, help="motor_model（2=DM4340P_24V）")
    ap.add_argument("--master-id-offset", type=int, default=0)
    ap.add_argument("--amplitude", type=float, default=0.03, help="幅度 rad（默认 0.03≈1.7°）")
    ap.add_argument("--kp", type=float, default=20.0,
                    help="MIT kp（默认 20 = 达妙官方起点，见 docs/03-单关节稳停）")
    ap.add_argument("--kd", type=float, default=3.0, help="MIT kd（默认 3 = 官方值；严禁 0）")
    ap.add_argument("--assumed-sign", type=int, default=1, choices=(-1, 1),
                    help="⭐ 显式乘进下发值（见文件头要点①）")
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--hold", type=float, default=1.2,
                    help="每次持续下发指令的秒数（达妙 MIT 需持续指令才持续出力）")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--model-dirs", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "dm10_model_directions.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None,
                    help="逗号分隔的 'bus:id' 列表，只跑这些（如 can2:1,can1:4）")
    ap.add_argument("--toward-limit", choices=("lo", "hi", "both"), default=None,
                    help="⭐ 走到接近配置限位，验证行程/软限位（防呆：位置没动会报异常）")
    ap.add_argument("--robot-yaml", default=None,
                    help="robot.yaml 路径（取 motor_sign 用）；默认 "
                         "~/roboparty_deploy/src/inference/robots/dm10/robot.yaml")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="距配置限位保留多少余量 rad（默认 0.15，比 limit_check.py 的 0.05 保守）")
    ap.add_argument("--ramp", type=float, default=2.0,
                    help="走到目标的平滑插值秒数（默认 2s，不阶跃）")
    ap.add_argument("--settle-limit", type=float, default=3.0,
                    help="到达后再停多久才判定（默认 3s）")
    ap.add_argument("--use-config-sign", action="store_true",
                    help="⭐ 上机前复核：从 robot.yaml 读该关节的 motor_sign 并乘进下发值，"
                         "走【完整生产路径】验证配置本身对不对（而非仅验证电机原始方向）")
    ap.add_argument("--one-way", type=int, choices=(-1, 1), default=None,
                    help="只做一个方向（+1 或 -1），保持住让你看清，再回位")
    ap.add_argument("--hold-view", type=float, default=2.0,
                    help="--one-way 时保持在该位置让你观察的秒数")
    ap.add_argument("--enable-check", action="store_true",
                    help="只验证「照新时序能否保持使能」，不做方向测试")
    ap.add_argument("--confirm", action="store_true",
                    help="⭐ 必须显式给才会真正使能电机；不给则只打印计划")
    args = ap.parse_args()

    if args.amplitude > AMP_HARD_CAP:
        print(f"{RED}✗ 幅度 {args.amplitude} 超过硬上限 {AMP_HARD_CAP} rad，拒绝执行{RST}")
        return 2

    if args.only:
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        plan = [t for t in ORDER if f"{t[0]}:{t[1]}" in want]
        if not plan:
            print(f"{RED}✗ --only 没匹配到任何关节：{sorted(want)}{RST}")
            return 2
    elif args.all:
        plan = ORDER
    elif args.bus and args.motor_id is not None:
        hit = [t for t in ORDER if t[0] == args.bus and t[1] == args.motor_id]
        if not hit:
            print(f"{RED}✗ 配置里没有 {args.bus} ID={args.motor_id}{RST}")
            print(f"{YEL}  已知组合：{[(b, i) for b, i, _, _ in ORDER]}{RST}")
            return 2
        plan = hit
    else:
        print(f"{RED}✗ 要么给 --all，要么同时给 --bus 和 --motor-id{RST}")
        return 2

    model_dirs = _load_model_directions(args.model_dirs)
    if not model_dirs:
        print(f"{YEL}ⓘ 读不到模型期望表 {args.model_dirs}（先跑 export_model_directions.py）{RST}")

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", f"direction_{_dt.datetime.now():%Y%m%d}.json")

    print("=" * 78)
    print("  电机方向符号验证探针")
    print("=" * 78)
    print(f"  模式      : {'⚠️ 会【使能电机并让它真的动】' if args.confirm else '只读（dry-run，不使能）'}")
    print(f"  台数      : {len(plan)}")
    print(f"  幅度      : ±{args.amplitude} rad (±{args.amplitude * 57.29578:.2f}°)")
    print(f"  增益      : kp={args.kp}  kd={args.kd}")
    print(f"  assumed   : {args.assumed_sign:+d}")
    print(f"  落盘      : {out_path}")
    print()
    print(f"  {'总线':<6}{'ID':<4}{'关节':<14}{'模型期望（给正角时脚怎么动）'}")
    print("  " + "-" * 74)
    for bus, mid, name, jnt in plan:
        print(f"  {bus:<6}{mid:<4}{name:<14}{_describe_expect(model_dirs.get(jnt))}")
    print()

    if not args.confirm:
        print(f"{YEL}ⓘ dry-run 结束，未使能任何电机。要真跑请加 --confirm{RST}")
        print(f"{YEL}  ⚠️ 确认前提：机器人【吊着】、操作员在旁、能随时 Ctrl-C 失能{RST}")
        return 0

    print(f"{RED}⚠️ 3 秒后开始 —— 机器人必须吊着、有人看着。Ctrl-C 立即中断。{RST}")
    try:
        for i in range(3, 0, -1):
            print(f"    {i}...")
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YEL}已取消{RST}")
        return 130

    motors_py = need_motors_py()

    # ⭐ --enable-check：只验证「照新时序能否保持使能」，不做方向测试
    if args.enable_check:
        rc = 0
        for bus, mid, name, jnt in plan:
            rec = _enable_check_one(motors_py, bus, mid, name, jnt, args)
            if not rec.get("enable_ok"):
                rc = 1
        return rc

    results = []
    try:
        for bus, mid, name, jnt in plan:
            run_one(motors_py, bus, mid, name, jnt, args, model_dirs, results, out_path)
    except KeyboardInterrupt:
        print(f"\n{YEL}⏹ 用户中断 —— 已测 {len(results)} 台，结论已落盘{RST}")

    print()
    print("=" * 78)
    print("  汇总")
    print("=" * 78)
    ok = [r for r in results if r.get("verdict") == "可判（读数有效）"]
    ok_limit = [r for r in results if r.get("mode") == "toward-limit"
                and str(r.get("verdict", "")).startswith("到达限位附近")]
    for r in results:
        # ⚠️ toward-limit 模式写的是 delta_lo/delta_hi，不是 delta_rad
        d = r.get("delta_rad")
        if isinstance(d, float):
            ds = f"Δ={d:+.5f}"
        elif r.get("delta_lo") is not None or r.get("delta_hi") is not None:
            parts = []
            for k in ("lo", "hi"):
                v = r.get(f"delta_{k}")
                if isinstance(v, float):
                    parts.append(f"{k}={v:+.3f}")
            ds = " ".join(parts)
        else:
            ds = "Δ=—"
        print(f"  {r['joint_name']:<14} {ds:<22} {r.get('verdict', '—')}")
    print()
    if ok_limit:
        print(f"  {GRN}行程检查通过 {len(ok_limit)} / {len(results)} 台{RST}")
    print(f"  可判 {len(ok)} / {len(results)} 台")
    print(f"  {GRN}▶ 下一步：逐台对照「模型期望」肉眼确认方向，回填 motor_sign{RST}")
    print(f"  记录: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
