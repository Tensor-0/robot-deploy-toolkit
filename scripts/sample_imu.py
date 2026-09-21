#!/usr/bin/env python3
"""sample_imu.py — 采一段 IMU 数据（用于验证轴向 / 看转动）

不使能电机、不碰 CAN，纯读串口 IMU。

用法:
    python3 scripts/sample_imu.py --seconds 5          # 采 5 秒，打印时序
    python3 scripts/sample_imu.py --seconds 5 --quiet  # 只打印汇总
"""
import argparse
import math
import sys
import time

try:
    import imu_py
except ImportError:
    print("✗ 找不到 imu_py —— 需先 `source install/setup.bash`")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--rows", type=int, default=20, help="打印的行数")
    args = ap.parse_args()

    imu = imu_py.IMUDriver.create_imu(1, "serial", args.port, "DM_IMU_L1", 921600)
    time.sleep(0.3)

    rows = []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        try:
            g = [float(v) for v in imu.get_ang_vel()]
            a = [float(v) for v in imu.get_lin_acc()]
            q = [float(v) for v in imu.get_quat()]
            rows.append((time.time() - t0, g, a, q))
        except Exception:
            pass
        time.sleep(0.02)

    n = len(rows)
    print(f"采到 {n} 个样点（{n / args.seconds:.0f} Hz，{args.seconds:.0f} 秒）")

    header = "  t(s)       wx       wy       wz  |      ax      ay      az"
    if not args.quiet:
        print()
        print(header)
        print("-" * len(header))
        step = max(1, n // args.rows)
        for i in range(0, n, step):
            t, g, a, _q = rows[i]
            print(f"{t:6.2f}{g[0]:+9.3f}{g[1]:+9.3f}{g[2]:+9.3f}  |"
                  f"{a[0]:+8.2f}{a[1]:+8.2f}{a[2]:+8.2f}")

    gx = [r[1][0] for r in rows]
    gy = [r[1][1] for r in rows]
    gz = [r[1][2] for r in rows]
    mx, my, mz = (max(abs(v) for v in s) for s in (gx, gy, gz))

    def mean(s):
        return sum(s) / len(s) if s else 0.0

    print()
    print(f"角速度 峰值 |wx|={mx:.3f} |wy|={my:.3f} |wz|={mz:.3f}  rad/s")
    print(f"角速度 均值 wx={mean(gx):+.4f} wy={mean(gy):+.4f} wz={mean(gz):+.4f}")
    big = max([("wx(点头)", mx), ("wy(侧翻)", my), ("wz(转身)", mz)], key=lambda x: x[1])
    print(f"⇒ 最大分量: {big[0]} = {big[1]:.3f} rad/s")
    print(f"   峰值比 wx:wy:wz = {mx:.3f} : {my:.3f} : {mz:.3f}")
    if big[1] > 0.1:
        print(f"   等效转速 ≈ {big[1] * 57.3:.0f} °/s（该轴）")
    else:
        print("   ⚠️ 三个轴都没明显转动 —— 采样期间可能没在转")


if __name__ == "__main__":
    main()
