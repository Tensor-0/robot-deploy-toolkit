#!/usr/bin/env python3
"""limit_check.py — 限位 / 行程检查（阶段 2）

**验证什么**：
  1. 关节能在软限位内自由转动（不撞硬限位）
  2. 软限位触发时的角度与配置是否一致
  3. 有没有异常干涉（异响/卡顿/电流突增）

⚠️ **这个脚本比 joint_test.py 更危险** —— 它是往极限位置走。
   默认幅度小、速度慢，且到达配置限位就停。

用法:
  # 从配置读 joint_limits，逐关节慢速扫到限位附近就停
  python3 limit_check.py --config <robot.yaml>

  # 指定关节 + 手动范围
  python3 limit_check.py --config <robot.yaml> --joint 3 --range -0.2,2.0

前置：机器人必须**空载 + 悬空**。别人身站输出轴方向。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RED, YEL, GRN, RST, LOOP_DT, need_motors_py, load_yaml,
                     build_motor_specs, create_motors, safe_deinit,
                     read_error_clean, confirm)

SAFETY_MARGIN = 0.05      # 距配置限位保留多少余量（rad）
SPEED = 0.15              # rad/s，很慢


def _find_key(obj, key):
    """递归找 key，返回第一个命中的值（兼容 ROS2 参数文件的嵌套结构）"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def get_limits(cfg, index):
    """从配置里取该关节的 [min, max]。

    支持多种来源与结构：
      1. robot.joint_limits          （扁平列表，或单一关节的 [min,max]）
      2. ROS2 参数文件嵌套结构        inference_node.ros__parameters.joint_limits
      3. --limits-from 指定的 policy yaml（同一个 key，任意嵌套深度）
      4. --range 手动指定（在 main 里覆盖）
    """
    jl = _find_key(cfg, "joint_limits")
    if not jl:
        return None
    if len(jl) == 2 and all(isinstance(x, (int, float)) for x in jl):
        return (jl[0], jl[1])          # 单一关节写法
    if len(jl) >= 2 * (index + 1):
        return (jl[2 * index], jl[2 * index + 1])
    return None


def main():
    ap = argparse.ArgumentParser(description="限位/行程检查")
    ap.add_argument("--config", required=True)
    ap.add_argument("--joint", type=int, help="只测某个索引")
    ap.add_argument("--range", help="手动给范围 min,max（覆盖配置）")
    ap.add_argument("--limits-from",
                    help="从另一个 yaml 读 joint_limits（通常是 policy yaml，"
                         "如 robots/dm10/configs/default.yaml）")
    ap.add_argument("--out", default="results/limit_check.json")
    ap.add_argument("--yes", action="store_true", help="跳过确认（危险）")
    args = ap.parse_args()

    # ── 先做纯配置检查（不需要硬件）──────────────
    cfg = load_yaml(args.config)
    specs_all = build_motor_specs(cfg)
    specs = [s for s in specs_all if args.joint is None or s["index"] == args.joint]

    # 限位来源：优先 --limits-from（policy yaml 常放这里）
    lim_cfg = cfg
    lim_src = args.limits_from or args.config
    if args.limits_from:
        lim_cfg = load_yaml(args.limits_from)

    print("=" * 66)
    print(f" 限位 / 行程检查  ·  速度 {SPEED} rad/s（慢）")
    print(f" 限位来源: {lim_src}")
    print("=" * 66)

    # 预检：配置里到底有没有限位
    missing = []
    if not args.range:
        for s in specs:
            if get_limits(lim_cfg, s["index"]) is None:
                missing.append(s["index"])
    if missing:
        print(f"{RED}⚠️  没读到 joint_limits（索引 {missing}）{RST}")
        print(f"{YEL}   限位常放在 policy yaml 里，而不是 robot.yaml。")
        print(f"   试试：--limits-from <robots/<robot>/configs/<policy>.yaml>")
        print(f"   或：  --range min,max  手动指定{RST}")
        print()
        if len(missing) == len(specs):
            print(f"{RED}全部关节都缺限位 —— 无法继续{RST}")
            print()
            # 帮忙找一下 policy yaml
            import glob
            base = os.path.dirname(os.path.abspath(args.config))
            cands = glob.glob(os.path.join(base, "configs", "*.yaml"))
            if cands:
                print(f"{YEL}在 {base}/configs/ 下发现这些 policy yaml，")
                print(f"试试其中带 joint_limits 的：{RST}")
                for c in cands:
                    try:
                        if get_limits(load_yaml(c), 0) is not None:
                            print(f"   --limits-from {c}   ← 含 joint_limits ✓")
                        else:
                            print(f"   {c}")
                    except Exception:
                        pass
            sys.exit(1)

    if not args.yes:
        confirm("机器人空载 + 悬空？别人站在输出轴方向？急停在手边？")

    # ── 到这里才需要硬件 ──────────────────────────
    motors_py = need_motors_py()

    motors = create_motors(motors_py, specs)
    results = []
    try:
        for s, mv in motors:
            mv.init_motor()
            time.sleep(0.3)
            mv.set_motor_control_mode(motors_py.MotorControlMode.MIT)
            time.sleep(0.1)

            lim = None
            if args.range:
                lo, hi = (float(x) for x in args.range.split(","))
                lim = (lo, hi)
            else:
                lim = get_limits(lim_cfg, s["index"])

            if lim is None:
                print(f"[{s['index']:2d}] {YEL}配置里没有 joint_limits，跳过"
                      f"（用 --range 手动指定）{RST}")
                continue

            mv.refresh_motor_status(); time.sleep(0.05)
            q0 = mv.get_motor_pos()
            lo, hi = lim[0] + SAFETY_MARGIN, lim[1] - SAFETY_MARGIN

            print(f"[{s['index']:2d}] 配置限位 [{lim[0]:+.3f}, {lim[1]:+.3f}]  "
                  f"当前 {q0:+.4f}  扫描 [{lo:+.3f}, {hi:+.3f}]")

            reached = {"lo": None, "hi": None}
            err_peak = 0
            try:
                for label, tgt in (("lo", lo), ("hi", hi)):
                    q = mv.get_motor_pos()
                    n = max(1, int(abs(tgt - q) / SPEED / LOOP_DT))
                    for k in range(n):
                        q = q + (tgt - q) * min(1.0, (k + 1) / max(1, n - k))
                        mv.motor_mit_cmd(q, 0.0, 0.0, 1.0, 0.0)   # 纯阻尼
                        time.sleep(LOOP_DT)
                        if k % 20 == 0:                            # 定期查错误码
                            e = mv.get_error_id()
                            if isinstance(e, (int, float)) and e >= 8:
                                err_peak = max(err_peak, int(e))
                                raise RuntimeError(f"检测到错误码 {int(e)}，中止")
                    time.sleep(0.3)
                    mv.refresh_motor_status(); time.sleep(0.05)
                    reached[label] = round(mv.get_motor_pos(), 4)
            except (RuntimeError, KeyboardInterrupt) as e:
                print(f"     {RED}{e}{RST}")

            # 回中
            mid = (lim[0] + lim[1]) / 2
            q = mv.get_motor_pos()
            for k in range(max(1, int(abs(mid - q) / SPEED / LOOP_DT))):
                q = q + (mid - q) * 0.02
                mv.motor_mit_cmd(q, 0.0, 0.0, 1.0, 0.0)
                time.sleep(LOOP_DT)

            results.append({
                "index": s["index"], "motor_id": s["motor_id"],
                "limits_configured": lim, "q_initial": round(q0, 4),
                "reached_lo": reached["lo"], "reached_hi": reached["hi"],
                "error_peak": err_peak,
            })
            status = f"{GRN}✓{RST}" if err_peak == 0 else f"{RED}✗ 错误码 {err_peak}{RST}"
            print(f"     到达: {reached['lo']} / {reached['hi']}   {status}")

    except KeyboardInterrupt:
        print(f"\n{YEL}中断{RST}")
    finally:
        safe_deinit(motors)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results}, f, ensure_ascii=False, indent=2)

    err = [r for r in results if r["error_peak"]]
    print()
    print("=" * 66)
    print(f" 完成 {len(results)} 个关节 · 报错 {len(err)} 个")
    if err:
        print(f"{RED}⚠️  有电机在扫描中报错 —— 检查机械干涉{RST}")
    print(f" 已落盘: {args.out}")
    print("=" * 66)
    print()
    print(f"{YEL}提示：本脚本只扫到「配置限位 ± 余量」，未触碰硬限位。{RST}")
    print(f"{YEL}      要验硬限位，请手动缓慢推、听声音，别靠脚本。{RST}")


if __name__ == "__main__":
    main()
