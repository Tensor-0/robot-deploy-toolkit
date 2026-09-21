#!/usr/bin/env python3
"""sweep_gravity.py — 悬挂条件下量「重力前馈」表 tau_g(q)。

原理
----
稳态时（q̇=0、q̈=0）关节受力平衡：

    电机力矩 = 静态负载力矩
    Kp·(q_des − q) + Kd·(0 − q̇) + 0 = tau_g(q)
    ⇒  tau_g(q) = Kp × (q_des − q)          （q̇=0 时 Kd 项消失）

所以【不需要任何模型】—— 命令一个姿态、等它静止、读实际角，就能反推出该关节
在那个姿态下要扛多少静态力矩。

⭐ 但必须【双向逼近】，否则量到的是噪声
--------------------------------------
带静摩擦时平衡条件是 τ_PD = τ_load + τ_fric，而 τ_fric 的方向取决于【上一次运动方向】。
⇒ 单次测量的误差就是摩擦带 ±F —— 2026-09-21 实测膝的 F = 0.54 N·m，
   而 τ_load 本身只有 0.28 N·m（误差是信号的两倍）。

对每个点从下方、上方各逼近一次，摩擦反号，取平均即消：
    τ_load = (τ_low + τ_high) / 2      F = |τ_low − τ_high| / 2

⚠️ 这个数不只是重力：还有线束拖拽等一切【非摩擦的】静态负载。
   摩擦已经被双向平均消掉了，而它单独记在 friction 字段里。

⚠️ 只量【悬挂条件下】的。落地时地面反力会加进来，那是另一组数据。

摆动会污染读数（本脚本的 A+B 对策）
-----------------------------------
吊带挂着时，腿一动机身就反倾/摆动，而 tau_g 依赖【重力在机身系里的方向】：

    腿摆 1 rad ⇒ 机身反倾约 10° ⇒ 力矩误差约 0.7 N·m
    而 tau_g 本身只有 1~3 N·m ⇒ 同量级，不处理就是废表

更阴的是：摆动周期固定 + 采样延迟固定 ⇒ 会采到同一相位 ⇒
得到的是【系统性偏差】而不是随机噪声 —— 它看起来"数据很光滑"。

本脚本的对策：
  A. 慢过渡（默认 3 s）+ 每点等够时间（默认 5 s）
  B. 全程读 IMU 并【记录】机身倾角（tilt_mean_deg / tilt_max_deg）

⚠️ 吊带法下倾角是【消不掉】的：腿摆 0.3 m ⇒ CoM 移 0.03 m ⇒ 倾角 ≈ 3.6°，
   而默认姿态本身就有几度（吊点没对准重心）。3~6° 是常态。
   ⇒ 所以倾角是【记录下来的混淆变量】，事后按它过滤/解释，
     不是当场卡掉的阈值（--tilt-max 默认 8°，只当"明显坏了"的哨兵）。
   ⇒ 要一张干净的表，只能把躯干夹住（不用吊带）。

用法:
    # 第 1 步：单关节验证（默认姿态附近 3 个点，约 30 秒）
    python3 scripts/sweep_gravity.py --config <robot.yaml> --joint knee --n 3
    # 第 2 步：整条腿
    python3 scripts/sweep_gravity.py --config <robot.yaml> --leg left
    # 只回默认位，不动（冒烟测试）
    python3 scripts/sweep_gravity.py --config <robot.yaml> --dry-run

⚠️ 安全：悬吊下摔不了，但腿会大幅摆动 —— 别站在腿的活动范围内。
        Ctrl+C 立即失能。
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    GRN, RED, RST, YEL,
    build_motor_specs, create_motors, load_yaml, need_motors_py, safe_deinit,
)

# 顶层兜底：任何未捕获异常都要失能（2026-09-21 踩过 —— 异常跑在 finally 之前，
# 电机就留在使能态了）。main() 建好电机后把引用放进来。
_MOTORS_REF: list = []

FRAME_HZ = 100.0          # 发帧频率（每 10 ms 一帧）
SAMPLE_HZ = 50.0          # 采样频率（每 20 ms 读一次）
OSC_TOL = 0.010           # 振荡判据：窗口内极差 < 0.010 rad（≈26 个计数）
DRIFT_TOL = 0.005         # 漂移判据：|q̇| < 0.005 rad/s（0.5 s 内 < 0.0025 rad）
STABLE_WINDOW = 0.5       # 判稳窗口（秒）
MAX_WAIT = 25.0           # 单点最长等待（秒），超时标 unreliable
RAMP_KP_SCALE = 1.0 / 2.5 # 软启动期间 kp 缩放（对齐节点 reset_joints 的做法）


def tilt_deg(quat_wxyz):
    """机身相对竖直的倾角（度）。quat 是 [w,x,y,z]（DM-IMU-L1 的约定）。

    g_b = R^T·[0,0,-1]；直立时 g_b = [0,0,-1] ⇒ -g_b.z = 1 ⇒ tilt = 0。
    """
    w, x, y, z = quat_wxyz
    # 只算 z 分量，不需要完整旋转矩阵：
    #   R^T 的第三行 · [0,0,-1] 的相反数 = -(1 - 2(x²+y²))
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return float("nan")
    w, x, y, z = w/n, x/n, y/n, z/n
    gz = -(1.0 - 2.0*(x*x + y*y))     # 重力在机身系的 z 分量
    return math.degrees(math.acos(max(-1.0, min(1.0, -gz))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="robot.yaml（电机侧）")
    ap.add_argument("--infer-config", required=True,
                    help="configs/default.yaml（inference_node.ros__parameters 里才有 "
                         "joint_default_angle / joint_limits）"
                         "⚠️ 指向【板上跑的那份】—— 默认位是扫描网格的中心")
    ap.add_argument("--joint", help="只扫这一个关节，名字见 config 的 aliases：knee/hip_p/hip_r/hip_y/ankle")
    ap.add_argument("--leg", choices=["left", "right"], help="扫整条腿")
    ap.add_argument("--n", type=int, default=9, help="每个关节的采样点数")
    ap.add_argument("--ramp", type=float, default=3.0, help="点间斜坡时长（秒）")
    ap.add_argument("--ramp-short", type=float, default=1.5,
                    help="双向逼近时 δ 偏移段的斜坡时长（秒）")
    ap.add_argument("--delta", type=float, default=0.10,
                    help="双向逼近的偏移量（rad）。必须大到能【冲破静摩擦】——"
                         "实测摩擦带 0.5 N·m / Kp=18 ⇒ 0.03 rad，取 0.10 有 3 倍余量")
    ap.add_argument("--settle", type=float, default=5.0, help="到达后先等多久（秒）")
    ap.add_argument("--span", type=float, default=None,
                    help="扫描跨度（rad）。默认 = 关节行程的 80%。"
                         "⚠️ 落地测试必须给小值（或配 --n 1 只测默认位那一点）")
    ap.add_argument("--tilt-abort", type=float, default=15.0,
                    help="⚠️ 硬阈值：机身倾角超过它【立刻中止并失能】（落地安全阀）")
    ap.add_argument("--tilt-max", type=float, default=8.0,
                    help="机身倾角【哨兵】阈值（度）。吊带法下 3~6° 是常态，"
                         "别拿它当过滤器 —— 真正过滤靠记录下来的 tilt_mean_deg")
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="接管后先保持默认位 N 秒再开始扫描。"
                         "换控制权时用：本脚本 init 电机并开始发帧后，"
                         "在这段时间里停掉 inference 节点、把机器人放到地上")
    ap.add_argument("--hold-after", type=float, default=0.0,
                    help="⚠️ 扫完后【继续保持 PD 站立】N 秒再失能。"
                         "落地测试必须给大值 —— 否则测完立刻失能 = 机器人当场瘫倒。\n"
                         "本脚本跑完会打印提示；按 Ctrl+C 才真正结束。")
    ap.add_argument("--out", default=None, help="落盘路径")   # ⚠️ 别删：被 replace 吃掉过一次
    ap.add_argument("--dry-run", action="store_true", help="只回默认位，不扫")
    ap.add_argument("--verify-from",
                    help="验收模式 1（静止）：用 sweep JSON 里【默认姿态】那一行的 τ 当"
                         "常值前馈，对比稳态误差。⚠️ 摩擦带 F/Kp ≈ 0.031 rad 大于负载"
                         "下垂 τ_load/Kp ≈ 0.017 rad ⇒ 静态度量【分辨不出来】。")
    ap.add_argument("--track-from",
                    help="验收模式 2（动态）⭐：沿表的范围慢速扫一遍，A/B 差分。\n"
                         "同轨迹同速度、只差前馈 ⇒ e_A−e_B = τ_ff/Kp 把阻尼和摩擦都差掉。")
    ap.add_argument("--track-seconds", type=float, default=10.0, help="单趟扫描时长")
    args = ap.parse_args()

    # ⚠️ 参数自检（2026-09-21 踩过）：--out 被一次 str.replace 静默吃掉，
    #    语法检查抓不到（args.out 只在运行时访问），结果【跑完 5 分钟测量、
    #    在最后一行落盘时才崩】。这里一次性确认所有会用到的属性都在。
    for _a in ("config", "infer_config", "joint", "leg", "n", "ramp", "ramp_short",
               "delta", "settle", "span", "tilt_abort", "tilt_max", "hold_seconds",
               "hold_after", "out", "dry_run", "verify_from", "track_from", "track_seconds"):
        if not hasattr(args, _a):
            raise SystemExit(f"!! 参数自检失败：args.{_a} 不存在 —— argparse 定义被改坏了")

    cfg = load_yaml(args.config)
    motors_py = need_motors_py()
    specs = build_motor_specs(cfg)
    joint_num = len(specs)

    # joint_default_angle / joint_limits 在 inference config 里，不在 robot.yaml
    inf = load_yaml(args.infer_config)
    inf = inf.get("inference_node", {}).get("ros__parameters", inf)
    kp = list(cfg["robot"]["kp"])
    kd = list(cfg["robot"]["kd"])
    sign = list(cfg["robot"]["motor_sign"])
    q_def = list(inf["joint_default_angle"])
    lim = list(inf["joint_limits"])
    print(f"  默认位（扫网格中心）: {[round(v, 3) for v in q_def]}")
    print(f"  来自: {os.path.abspath(args.infer_config)}\n")

    # IMU（B 要用）
    imu = None
    try:
        import imu_py
        imu_cfg = cfg.get("imu", {})
        imu = imu_py.IMUDriver.create_imu(
            int(imu_cfg.get("imu_id", 1)), "serial",
            imu_cfg.get("imu_interface", "/dev/ttyACM0"),
            imu_cfg.get("imu_type", "DM_IMU_L1"),
            int(imu_cfg.get("baudrate", 921600)),
        )
        time.sleep(0.5)
        print(f"{GRN}✓ IMU 已连（判倾角用）{RST}")
    except Exception as exc:
        print(f"{YEL}⚠️  IMU 不可用（{exc}）—— B 档失效，只靠 A 档等静止{RST}")

    # 关节：按策略顺序 l1..l5, r1..r5
    NAMES = ["hip_p", "hip_r", "hip_y", "knee", "ankle"]
    def idx_of(leg, jname):
        base = 0 if leg == "left" else 5
        return base + NAMES.index(jname)

    if args.joint:
        if args.leg == "right":
            todo = [idx_of("right", args.joint)]
        elif args.leg == "left":
            todo = [idx_of("left", args.joint)]
        else:
            todo = [idx_of("left", args.joint), idx_of("right", args.joint)]
    elif args.leg:
        base = 0 if args.leg == "left" else 5
        todo = list(range(base, base + 5))
    elif args.dry_run:
        todo = []
    else:
        ap.error("给 --joint 或 --leg（或用 --dry-run 只回默认位）")

    motors = create_motors(motors_py, specs)   # [(spec, motor_driver), ...] 按关节顺序
    _MOTORS_REF.clear(); _MOTORS_REF.extend(motors)
    for _, mv in motors:
        mv.init_motor()
    time.sleep(0.5)
    print(f"{GRN}✓ {len(motors)} 台电机已使能{RST}\n")

    target = list(q_def)          # 当前的关节目标（初始 = 默认位）
    FF = [0.0] * joint_num        # 前馈力矩（【关节坐标】，发帧时再乘 sign）
    records = []
    t0 = time.time()

    ramp_kp = False              # 软启动期间降 kp（和节点 reset_joints 的 kp/2.5 同款）

    def send_once():
        """所有电机按当前 target 发一帧。

        ⚠️ 力矩和位置一样要乘 sign：能量守恒 τ_m·dq_m = τ_j·dq_j，而 q_j = sign·q_m
           ⇒ τ_m = sign × τ_j。
        """
        for i, (_, mv) in enumerate(motors):
            k = kp[i] * RAMP_KP_SCALE if ramp_kp else kp[i]
            mv.motor_mit_cmd(target[i] * sign[i], 0.0, k, kd[i], FF[i] * sign[i])

    def ramp_to(j_idx, q_end, seconds):
        """从当前 target 线性斜坡到 q_end（只动 j_idx），期间持续发帧。"""
        q_start = target[j_idx]
        steps = max(1, int(seconds * FRAME_HZ))
        for s in range(steps + 1):
            target[j_idx] = q_start + (q_end - q_start) * s / steps
            send_once()
            time.sleep(1.0 / FRAME_HZ)

    def hold_and_sample(seconds):
        """保持当前 target，按 SAMPLE_HZ 采样 (q, tilt)。"""
        out = []
        n = max(1, int(seconds * SAMPLE_HZ))
        for _ in range(n):
            send_once()
            time.sleep(1.0 / SAMPLE_HZ)
            q = [mv.get_motor_pos() * sign[i] for i, (_, mv) in enumerate(motors)]
            tl = float("nan")
            if imu is not None:
                try:
                    tl = tilt_deg([float(v) for v in imu.get_quat()])
                except Exception:
                    pass
            out.append((q, tl))
        return out

    class TiltAbort(RuntimeError):
        pass

    def settle_measure(j_idx):
        """等稳并测一次。返回 (q_avg, tilt_mean, status, spread, rate)。

        ⚠️ 机身倾角超过 --tilt-abort 时【抛异常立刻中止】—— 落地测试的安全阀。
           倾倒时关节角看不出（腿相对机身的角度不变），只有 IMU 能看出来。
        """
        time.sleep(args.settle)
        q_avg = tilt_mean = spread = rate = float("nan")
        waited = 0.0
        while True:
            samples = hold_and_sample(STABLE_WINDOW)
            qs = [s_[0][j_idx] for s_ in samples]
            tls = [s_[1] for s_ in samples if not math.isnan(s_[1])]
            q_avg = sum(qs) / len(qs)
            spread = max(qs) - min(qs)
            dt = len(qs) / SAMPLE_HZ
            rate = (qs[-1] - qs[0]) / dt if dt > 0 else 0.0
            if tls:
                tilt_mean = sum(tls) / len(tls)
                if max(tls) > args.tilt_abort:
                    raise TiltAbort(
                        f"机身倾角 {max(tls):.1f}° > {args.tilt_abort}° —— 立刻中止并失能")
            if abs(rate) > DRIFT_TOL:
                status = "在漂"
            elif spread > OSC_TOL:
                status = "在振"
            elif tls and max(tls) > args.tilt_max:
                status = "倾角大"
            else:
                return q_avg, tilt_mean, "ok", spread, rate
            waited += STABLE_WINDOW
            if waited >= MAX_WAIT:
                return q_avg, tilt_mean, "超时", spread, rate

    if args.track_from:
        # ── 验收模式 2：动态跟踪差分 ⭐ ────────────────────────────────
        if len(todo) != 1:
            ap.error("--track-from 一次只测一个关节（配 --joint --leg）")
        j_idx = todo[0]
        jname = f"{'l' if j_idx < 5 else 'r'}{NAMES[j_idx % 5]}"
        vd = json.load(open(args.track_from, encoding="utf-8"))
        rs = sorted([r for r in vd["records"]
                     if r.get("joint_index") == j_idx and r.get("status") == "ok"],
                    key=lambda x: x["q_des"])
        if len(rs) < 3:
            ap.error("表里这个关节的有效点太少")
        tq = [r["q_des"] for r in rs]
        tt = [r["tau"] for r in rs]

        def ff_of(q):
            """表插值（线性）。表外按端点夹住。"""
            if q <= tq[0]:
                return tt[0]
            if q >= tq[-1]:
                return tt[-1]
            for i in range(len(tq) - 1):
                if tq[i] <= q <= tq[i + 1]:
                    w = (q - tq[i]) / (tq[i + 1] - tq[i])
                    return tt[i] * (1 - w) + tt[i + 1] * w
            return 0.0

        a, bb = tq[0], tq[-1]
        kp_j = kp[j_idx]
        print(f"动态验收：{jname}  扫 [{a:+.3f}, {bb:+.3f}] rad / {args.track_seconds:.1f}s")
        print(f"  表 {len(rs)} 点，τ 从 {tt[0]:+.3f} 到 {tt[-1]:+.3f} N·m")
        print(f"  预期 e_A − e_B ≈ τ_ff/Kp，量级 {max(abs(x) for x in tt)/kp_j:.4f} rad"
              f" = {math.degrees(max(abs(x) for x in tt)/kp_j):.2f}°\n")

        def run_pass(use_ff, tag):
            n = int(args.track_seconds * FRAME_HZ)
            rows = []
            for k in range(n + 1):
                qd = a + (bb - a) * k / n
                target[j_idx] = qd
                FF[j_idx] = ff_of(qd) if use_ff else 0.0
                send_once()
                time.sleep(1.0 / FRAME_HZ)
                if k % 5 == 0:                     # 20 Hz 记录
                    q = motors[j_idx][1].get_motor_pos() * sign[j_idx]
                    rows.append((qd, q, FF[j_idx]))
            m = max(1, len(rows) // 5)             # 丢掉前 20%（起步瞬态）
            rows = rows[m:]
            e = [r[0] - r[1] for r in rows]
            print(f"  {tag}: 平均跟踪误差 {sum(e)/len(e):+.5f} rad (n={len(rows)})")
            return rows, e

        try:
            print("=========== A：τ_ff = 0 ===========")
            rowsA, eA = run_pass(False, "A")
            print("=========== B：τ_ff = 表 ===========")
            rowsB, eB = run_pass(True, "B")

            # 逐点差分（两次轨迹相同，下标对齐）
            m = min(len(eA), len(eB))
            de = [eA[i] - eB[i] for i in range(m)]
            exp = [rowsB[i][2] / kp_j for i in range(m)]
            # 相关系数（看形状对不对，不只看均值）
            ma, mb = sum(de)/m, sum(exp)/m
            cov = sum((de[i]-ma)*(exp[i]-mb) for i in range(m))
            va = math.sqrt(sum((x-ma)**2 for x in de))
            vb = math.sqrt(sum((x-mb)**2 for x in exp))
            corr = cov/(va*vb) if va*vb > 0 else float("nan")

            print(f"\n=========== 结果 ===========")
            print(f"  e_A 均值      {ma+mb:+.5f} rad  ← 含负载+摩擦+阻尼")
            print(f"  e_B 均值      {ma+mb-sum(de)/m:+.5f} rad")
            print(f"  Δe = e_A−e_B  {ma:+.5f} rad")
            print(f"  期望 τ_ff/Kp   {mb:+.5f} rad")
            print(f"  比值 Δe/期望   {ma/mb if mb else float('nan'):.3f}   （理想 = 1.00）")
            print(f"  逐点相关       {corr:.4f}   （理想 ≈ 1.00 —— 形状对，不只是均值对）")
            ok = (0.7 < (ma/mb if mb else 0) < 1.3) and corr > 0.8
            print(f"\n  {'✅ 前馈在动态下确实生效' if ok else '❌ 未达到判据（比值 0.7~1.3 且相关 >0.8）'}")
        except KeyboardInterrupt:
            print(f"\n{YEL}⚠️  中断{RST}")
        finally:
            FF[:] = [0.0] * joint_num
            try:
                for jj in range(joint_num):
                    target[jj] = q_def[jj]
                for _ in range(int(2.0 * FRAME_HZ)):
                    send_once(); time.sleep(1.0 / FRAME_HZ)
            except Exception:
                pass
            safe_deinit(motors)
            print(f"{GRN}✓ 已回默认位并失能{RST}")
        return 0

    if args.verify_from:
        # ── 验收模式：同一姿态，前馈 off / on 对照 ──────────────────────
        vd = json.load(open(args.verify_from, encoding="utf-8"))
        ff = [0.0] * joint_num
        picked = []
        for j_idx in todo:
            rs = [r for r in vd["records"]
                  if r.get("joint_index") == j_idx and r.get("status") == "ok"]
            if not rs:
                continue
            r = min(rs, key=lambda x: abs(x["q_des"] - q_def[j_idx]))
            ff[j_idx] = r["tau"]
            picked.append((j_idx, r["q_des"], r["tau"]))
        print(f"验收：前馈取自 {os.path.basename(args.verify_from)}")
        for j_idx, q, t in picked:
            jn = f"{'l' if j_idx < 5 else 'r'}{NAMES[j_idx % 5]}"
            print(f"    {jn:<8} 取 q_des={q:+.3f} 处的 τ={t:+.4f} N·m"
                  f"  (默认位 {q_def[j_idx]:+.3f})")
        print()

        def measure(tag):
            """回默认位 -> 等稳 -> 返回 (q_meas 列表, 倾角)"""
            for j in range(joint_num):
                target[j] = q_def[j]
            hold_and_sample(2.0)
            time.sleep(args.settle)
            samples = hold_and_sample(STABLE_WINDOW)
            return [s_[0] for s_ in samples], [s_[1] for s_ in samples]

        try:
            print("=========== A：τ_ff = 0 ===========")
            FF[:] = [0.0] * joint_num
            qa, tla = measure("A")
            qa_avg = [sum(x[i] for x in qa) / len(qa) for i in range(joint_num)]

            print("=========== B：τ_ff = 表值 ===========")
            for j_idx, _, t in picked:
                FF[j_idx] = t
            qb, tlb = measure("B")
            qb_avg = [sum(x[i] for x in qb) / len(qb) for i in range(joint_num)]

            print(f"\n{'关节':<9}{'默认位':>9}{'τ_ff':>9}{'e0(rad)':>11}{'e1(rad)':>11}"
                  f"{'e1/e0':>8}   判读")
            print("-" * 68)
            n_ok = 0
            for j_idx in [p_[0] for p_ in picked]:
                jn = f"{'l' if j_idx < 5 else 'r'}{NAMES[j_idx % 5]}"
                e0 = q_def[j_idx] - qa_avg[j_idx]
                e1 = q_def[j_idx] - qb_avg[j_idx]
                ratio = abs(e1) / abs(e0) if abs(e0) > 1e-9 else float("nan")
                verdict = "✅" if ratio < 0.5 else ("~" if ratio < 0.9 else "❌")
                if ratio < 0.5:
                    n_ok += 1
                print(f"{jn:<9}{q_def[j_idx]:>9.3f}{FF[j_idx]:>9.3f}{e0:>11.4f}{e1:>11.4f}"
                      f"{ratio:>8.2f}   {verdict}")
            ta = sum(x for x in tla if not math.isnan(x)) / max(1, len([x for x in tla if not math.isnan(x)]))
            tb = sum(x for x in tlb if not math.isnan(x)) / max(1, len([x for x in tlb if not math.isnan(x)]))
            print(f"\n  倾角: A {ta:.2f}°  B {tb:.2f}°   （差 {abs(tb-ta):.2f}° —— 越小越说明对照干净）")
            print(f"  改善到 1/2 以下：{n_ok}/{len(picked)}")
        except KeyboardInterrupt:
            print(f"\n{YEL}⚠️  中断{RST}")
        finally:
            FF[:] = [0.0] * joint_num
            try:
                for j in range(joint_num):
                    target[j] = q_def[j]
                for _ in range(int(2.0 * FRAME_HZ)):
                    send_once(); time.sleep(1.0 / FRAME_HZ)
            except Exception:
                pass
            safe_deinit(motors)
            print(f"{GRN}✓ 已回默认位并失能{RST}")
        return 0

    try:
        # ⭐ 软启动：从【当前实测位置】插值到默认位，而不是直接发默认位。
        #   接管时腿可能是垂着的（节点刚失能），阶跃会让腿甩一下。
        #   与节点 reset_joints 同款：4 秒 + kp/2.5。
        cur = [motors[i][1].get_motor_pos() * sign[i] for i in range(joint_num)]
        dev = max(abs(cur[i] - q_def[i]) for i in range(joint_num))
        if dev > 0.02:
            print(f"→ 软启动：当前位置偏离默认位最大 {dev:.3f} rad，4 秒插值过去（kp/2.5）…")
            ramp_kp = True
            n_ramp = int(4.0 * FRAME_HZ)
            for k in range(n_ramp + 1):
                for i in range(joint_num):
                    target[i] = cur[i] + (q_def[i] - cur[i]) * k / n_ramp
                send_once()
                time.sleep(1.0 / FRAME_HZ)
            ramp_kp = False
        else:
            print("→ 已在默认位附近（偏离 %.3f rad），无需软启动" % dev)
        for j in range(joint_num):
            target[j] = q_def[j]
        hold_and_sample(2.0)
        print(f"{GRN}✓ 已在默认位{RST}\n")

        if args.hold_seconds > 0:
            print(f"→ 保持默认位 {args.hold_seconds:.0f} 秒 —— "
                  f"趁这段时间停止 inference 节点 / 把机器人放到地上…")
            for _ in range(int(args.hold_seconds * FRAME_HZ)):
                send_once()
                time.sleep(1.0 / FRAME_HZ)
            print(f"{GRN}✓ 保持结束，开始扫描{RST}\n")

        for j_idx in todo:
            jname = f"{'l' if j_idx < 5 else 'r'}{NAMES[j_idx % 5]}"
            lo, hi = lim[2*j_idx], lim[2*j_idx + 1]
            # 扫点：占量程 80%，尽量以默认位为中心；中心装不下时【整段平移】进限位。
            # ⚠️ 不要逐点 clamp —— 那会把靠近限位的两个点夹成同一个值（2026-09-21 踩过）。
            m = 0.05
            lo_ok, hi_ok = lo + m, hi - m
            if args.span is not None:
                span = min(args.span, hi_ok - lo_ok)
                c = q_def[j_idx]            # 以默认位为中心，不平移
            else:
                span = (hi_ok - lo_ok) * 0.80
                c = max(lo_ok + span / 2, min(hi_ok - span / 2, q_def[j_idx]))
            if args.n <= 1:
                pts = [c]                   # ⚠️ 单点必须取中点：原式 n=1 会落到左边缘
            else:
                pts = [c - span / 2 + span * k / (args.n - 1) for k in range(args.n)]
            print(f"=== {jname}  (默认 {q_def[j_idx]:+.3f}, 限位 [{lo:+.3f}, {hi:+.3f}], "
                  f"Kp={kp[j_idx]:.1f}, 扫 [{pts[0]:+.3f}, {pts[-1]:+.3f}])")
            print(f"{'目标':>9}{'τ_low':>10}{'τ_high':>10}{'τ_load':>10}{'F':>9}"
                  f"{'倾角°':>8}   状态")
            print("   " + "-" * 56)
            for q_t in pts:
                # ⭐ 双向逼近：从下方和上方各逼近一次，摩擦项反号 ⇒ 取平均即消掉。
                #    单次测量的误差就是摩擦带 ±F（实测 0.54 N·m，比信号还大）。
                q_lo_edge = max(lo, q_t - args.delta)
                ramp_to(j_idx, q_lo_edge, args.ramp_short)
                ramp_to(j_idx, q_t, args.ramp)
                q_low, tilt_low, st_low, sp_low, rt_low = settle_measure(j_idx)

                q_hi_edge = min(hi, q_t + args.delta)
                ramp_to(j_idx, q_hi_edge, args.ramp_short)
                ramp_to(j_idx, q_t, args.ramp)
                q_high, tilt_high, st_high, sp_high, rt_high = settle_measure(j_idx)

                tau_low = kp[j_idx] * (q_t - q_low)
                tau_high = kp[j_idx] * (q_t - q_high)
                tau = 0.5 * (tau_low + tau_high)          # 摩擦相消
                fric = 0.5 * abs(tau_low - tau_high)      # 顺带量出摩擦

                exc_lo, exc_hi = q_t - q_lo_edge, q_hi_edge - q_t
                status = st_low if st_low != "ok" else st_high
                if status == "ok" and min(exc_lo, exc_hi) < 0.03:
                    status = "偏移不足"       # 贴着限位，冲不破摩擦 ⇒ 那一侧不可信
                spread, rate = max(sp_low, sp_high), max(abs(rt_low), abs(rt_high))
                tms = [t for t in (tilt_low, tilt_high) if not math.isnan(t)]
                tilt_mean = sum(tms) / len(tms) if tms else float("nan")

                records.append({
                    "joint": jname, "joint_index": j_idx,
                    "q_des": round(q_t, 6),
                    "q_low": round(q_low, 6), "q_high": round(q_high, 6),
                    "q_meas": round(0.5 * (q_low + q_high), 6),
                    "tau_low": round(tau_low, 6), "tau_high": round(tau_high, 6),
                    "tau": round(tau, 6), "friction": round(fric, 6),
                    "exc_lo": round(exc_lo, 4), "exc_hi": round(exc_hi, 4),
                    "spread": round(spread, 6), "rate": round(rate, 6),
                    "tilt_mean_deg": None if math.isnan(tilt_mean) else round(tilt_mean, 3),
                    "status": status,
                })
                tl_s = "  n/a" if math.isnan(tilt_mean) else f"{tilt_mean:5.2f}"
                mark = f"{GRN}✓{RST}" if status == "ok" else f"{YEL}~{RST}"
                print(f"{q_t:>9.3f}{tau_low:>10.3f}{tau_high:>10.3f}{tau:>10.3f}{fric:>9.3f}"
                      f"{tl_s:>8}  {mark} {status}")
            # 每个关节扫完回默认位
            ramp_to(j_idx, q_def[j_idx], args.ramp)
            time.sleep(1.0)
            print()

    except KeyboardInterrupt:
        print(f"\n{YEL}⚠️  用户中断 —— 回默认位并失能{RST}")
    except TiltAbort as exc:
        print(f"\n{RED}🛑 {exc}{RST}")
    finally:
        # 无论如何都回默认位 + 失能
        try:
            for j in range(joint_num):
                target[j] = q_def[j]
            for _ in range(int(2.0 * FRAME_HZ)):
                send_once()
                time.sleep(1.0 / FRAME_HZ)
        except Exception:
            pass
        safe_deinit(motors)
        print(f"{GRN}✓ 已回默认位并失能{RST}")

    # ⚠️ 落地安全：扫完后默认【保持站立】，不是立刻失能
    if args.hold_after > 0 and records:
        print(f"\n{RED}⚠️  机器人还站着！先收紧吊带把它吊起来。{RST}")
        print(f"   脚本会继续保持 PD 站立 {args.hold_after:.0f} 秒。"
              f"之后自动回默认位并失能 —— 那时腿会松掉，请确保吊带已承重。")
        try:
            for k in range(int(args.hold_after * FRAME_HZ)):
                send_once()
                time.sleep(1.0 / FRAME_HZ)
                if k % int(10 * FRAME_HZ) == 0 and k:
                    print(f"   …还剩 {args.hold_after - k/FRAME_HZ:.0f} 秒")
        except KeyboardInterrupt:
            print("\n  收到 Ctrl+C —— 回默认位并失能")

    # 落盘
    ok = [r for r in records if r["status"] == "ok"]
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        f"gravity_sweep_suspended_{datetime.now():%Y%m%d-%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "dm10-gravity-sweep/1",
            "created": datetime.now().isoformat(),
            "config": os.path.abspath(args.config),
            "condition": "suspended (hoist on torso) — NOT ground contact",
            "note": "tau_g(q) = Kp × (q_des − q) at rest; includes friction/cable, not just gravity",
            "params": {"ramp_s": args.ramp, "settle_s": args.settle,
                       "tilt_max_deg": args.tilt_max, "n_points": args.n},
            "records": records,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  记录 {len(records)} 点，其中 {GRN}{len(ok)} 可用{RST}"
          f"（其余 未稳/倾角大/超时）")
    print(f"  📄 {os.path.abspath(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise          # ⚠️ 正常退出也会走 SHUTDOWN 路径，别当成异常
    except BaseException:
        if _MOTORS_REF:
            print(f"{YEL}⚠️  异常退出 —— 兜底失能{RST}", flush=True)
            try:
                safe_deinit(_MOTORS_REF)
            except Exception:
                pass
        raise
