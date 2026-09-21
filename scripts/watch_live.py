#!/usr/bin/env python3
"""watch_live.py — 实时刷屏看电机读数（⭐ 全程不使能电机）

用途
----
手动摆动机器人时**实时**看每个关节的读数，用来回答这类问题：
  · 我动的是哪条腿？ → 哪条总线上的数字在变，就是它
  · 这个关节转 10°，读数变 10° 吗？ → 刻度对不对
  · 我只动一个关节，别的数字动不动？ → 有没有串扰 / ID 映射错
  · 往正方向转，读数是变大还是变小？ → 方向对不对

比"摆好姿态 → 读一次 → 猜"直观得多：你能立刻看到因果。

⚠️⚠️ 安全：本脚本【绝不使能电机】（不调用 init_motors）。
    走逐台 motors_py：unlock_motor(失能) → 反复 refresh_motor_status(纯读请求) → 读位置。
    全程不发任何 MIT/POS/SPD 指令，物理上不可能产生力矩。
    但**仍需**机器人架住/吊住（失能时腿是软的）。

用法:
    source /opt/ros/humble/setup.bash && source <roboparty_deploy>/install/setup.bash
    python3 scripts/watch_live.py --config <roboparty_deploy>/src/inference/robots/dm10/robot.yaml

    按键：  q 或 Ctrl+C 退出     z 把当前读数记为基线（清零对照）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import GRN, RED, RST, YEL, build_motor_specs, create_motors, load_yaml, need_motors_py  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="实时看电机读数（不使能）")
    ap.add_argument("--config", required=True, help="robot.yaml")
    ap.add_argument("--hz", type=float, default=10.0, help="刷新率（默认 10Hz）")
    ap.add_argument("--no-install-hint", action="store_true", help="不打印 colcon 提示")
    return ap.parse_args()


def bar(v, lo=-0.3, hi=0.3, width=13):
    """把变化量画成一条小横条（− 左 / + 右），方便肉眼追踪方向。"""
    if v is None:
        return " " * width + "·"
    p = (v - lo) / (hi - lo)
    p = max(0.0, min(1.0, p))
    idx = int(round(p * (width - 1)))
    cells = ["-"] * width
    cells[width // 2] = "|"
    cells[idx] = "#"
    return "".join(cells)


def main():
    args = parse_args()
    print(f"{YEL}{'=' * 78}")
    print(" 实时读数 —— 本脚本【不会】给电机使能（全程失能，物理上不产生力矩）")
    print("=" * 78)
    print(" ⚠️ 机器人必须已架住/吊住：失能状态下腿是软的、会自然下垂")
    print(" 按键：q=退出   z=把当前读数记为基线（看相对变化）")
    print(f"{'=' * 78}{RST}")

    cfg = load_yaml(args.config)
    specs = build_motor_specs(cfg)
    motors_py = need_motors_py()
    motors = create_motors(motors_py, specs)

    # 先全部失能（幂等；即使上一次是使能状态，这一步也会放开）
    for _, mv in motors:
        try:
            mv.unlock_motor()
        except Exception:
            pass
    time.sleep(0.15)
    print(f"{GRN}✓ {len(motors)} 台电机已失能，开始读{RST}\n")

    # 非阻塞按键
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    baseline: dict[int, float] = {}
    prev: dict[int, float] = {}
    rate: dict[int, float] = {}
    period = 1.0 / max(1.0, args.hz)
    n = 0
    try:
        print("\033[2J\033[H", end="")  # 清屏
        while True:
            t0 = time.perf_counter()
            rows = []
            for spec, mv in motors:
                i = spec["index"]
                try:
                    mv.refresh_motor_status()
                    p = float(mv.get_motor_pos())
                except Exception:
                    rows.append((spec, None, None, None))
                    continue
                if i in prev and period > 0:
                    rate[i] = (p - prev[i]) / period
                prev[i] = p
                rows.append((spec, p, baseline.get(i), rate.get(i)))

            n += 1
            out = ["\033[H"]
            out.append(
                f"{'总线':<7}{'ID':<4}{'读数(rad)':>11}{'读数(°)':>10}"
                f"{'Δ基线(°)':>11}{'变化速率(°/s)':>15}   ─'−'左 '+'右─\n"
            )
            out.append("-" * 78 + "\n")
            for spec, p, b, r in rows:
                tag = f"{spec['interface']:<7}{spec['motor_id']:<4}"
                if p is None:
                    out.append(f"{tag}{RED}读失败{RST}\n")
                    continue
                import math

                deg = math.degrees(p)
                dbase = "        ·" if b is None else f"{math.degrees(p - b):>+10.1f}"
                rdeg = "      ·" if r is None else f"{math.degrees(r):>+13.1f}"
                # 用"相对基线"画条，方便看哪个在动；无基线时用原始 p
                barmark = bar(p - b if b is not None else p)
                out.append(f"{tag}{p:>11.4f}{deg:>10.1f}{dbase}{rdeg}   {barmark}\n")
            out.append("-" * 78 + "\n")
            out.append(f" 刷新 {n} 次 | {args.hz:.0f}Hz | 基线{'已设' if baseline else '未设(按 z 设置)'}\n")
            sys.stdout.write("".join(out))
            sys.stdout.flush()

            # 按键检查
            if select.select([sys.stdin], [], [], period)[0]:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q", "\x03"):
                    break
                if ch in ("z", "Z"):
                    baseline = {spec["index"]: prev.get(spec["index"], 0.0) for spec, _ in motors}
                    sys.stdout.write("\n  ⭐ 基线已设置（现在的读数记为 0）\n")
                    sys.stdout.flush()
                    time.sleep(0.6)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        for _, mv in motors:
            try:
                mv.unlock_motor()
            except Exception:
                pass
        print(f"\n{GRN}已退出（电机保持失能）{RST}")


if __name__ == "__main__":
    main()
