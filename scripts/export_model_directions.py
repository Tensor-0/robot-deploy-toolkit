"""从 dm10 MJCF 导出【模型期望方向表】，供实机方向验证做对照基准。

**为什么需要它**：实机测出"给正指令读数往哪变"之后，必须有一个权威的"模型期望值"
才能判定 `motor_sign` 该填 +1 还是 −1。这份表就是这个基准。

输出两个量，分别对应两类关节：
  · `foot_rot_rpy_deg`  脚底相对机体的旋转变化 —— 左右**必须镜像**（髋roll/yaw 靠它判）
  · `foot_pos_delta_mm`  脚底相对机体的位移变化 —— 左右**必须相同**（其余关节靠它判）

⚠️ 关键性质：本模型左右腿的关节【轴向量完全相同，不镜像】。
   因此髋roll/yaw 给正角时两条腿往同一侧倒。这不是 bug，是模型事实；
   实机验证时会看到同样现象，别误判成接线错误。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("MUJOCO_GL", "egl")

XML_DEFAULT = "/home/zhan/UniLab/src/unilab/assets/robots/dm10/scene_flat.xml"

# 顺序 = 策略关节顺序 l1..l5(髋→脚), r1..r5(髋→脚)，即 robot.yaml 里 motor_id 的索引顺序
JOINTS = [
    ("leg_l1_joint", "l1", "左 髋pitch", "hip_pitch"),
    ("leg_l2_joint", "l2", "左 髋roll", "hip_roll"),
    ("leg_l3_joint", "l3", "左 髋yaw", "hip_yaw"),
    ("leg_l4_joint", "l4", "左 膝", "knee"),
    ("leg_l5_joint", "l5", "左 踝", "ankle"),
    ("leg_r1_joint", "r1", "右 髋pitch", "hip_pitch"),
    ("leg_r2_joint", "r2", "右 髋roll", "hip_roll"),
    ("leg_r3_joint", "r3", "右 髋yaw", "hip_yaw"),
    ("leg_r4_joint", "r4", "右 膝", "knee"),
    ("leg_r5_joint", "r5", "右 踝", "ankle"),
]
FOOT_SITE = {"l": "left_foot", "r": "right_foot"}

# 左右在模型里【必须镜像】的关节（旋转量判据）；其余用位移量判据
MIRROR_JOINTS = {"hip_roll", "hip_yaw"}


def _rot_to_rpy_deg(R):
    """旋转矩阵 → (roll, pitch, yaw) 单位度，XYZ 内旋约定。"""
    import numpy as np

    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = np.arcsin(sy)
    if abs(abs(sy) - 1.0) < 1e-6:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def build(xml_path: str, delta_rad: float) -> dict:
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(xml_path)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base < 0:
        raise SystemExit(f"✗ 在 {xml_path} 里找不到 base_link")

    def local(kind, name, d):
        obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        oid = mujoco.mj_name2id(m, obj, name)
        if oid < 0:
            raise SystemExit(f"✗ 找不到 {kind} '{name}'")
        p_w = d.site_xpos[oid] if kind == "site" else d.xpos[oid]
        R_w = (d.site_xmat[oid] if kind == "site" else d.xmat[oid]).reshape(3, 3)
        Rb = d.xmat[base].reshape(3, 3)
        return Rb.T @ (p_w - d.xpos[base]), Rb.T @ R_w

    def solve(joint, val):
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.75
        if joint:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                raise SystemExit(f"✗ 找不到关节 '{joint}'")
            d.qpos[m.jnt_qposadr[jid]] = val
        mujoco.mj_forward(m, d)
        return d

    d0 = solve(None, 0)
    entries = []
    for jname, short, label, kind in JOINTS:
        leg = short[0]
        site = FOOT_SITE[leg]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = (float(v) for v in m.jnt_range[jid])
        axis = m.jnt_axis[jid]
        ax_i = int(np.argmax(np.abs(axis)))

        d1 = solve(jname, delta_rad)
        p0, R0 = local("site", site, d0)
        p1, R1 = local("site", site, d1)
        dp = (p1 - p0) * 1000.0
        rpy = _rot_to_rpy_deg(R1.T @ R0)

        entries.append({
            "index": len(entries),
            "joint": jname,
            "short": short,
            "label": label,
            "primitive": kind,
            "model_axis": ["X", "Y", "Z"][ax_i],
            "model_axis_xyz": [float(v) for v in axis],
            "limit_rad": [round(lo, 4), round(hi, 4)],
            "foot_rot_rpy_deg": [round(float(v), 2) for v in rpy],
            "foot_pos_delta_mm": [round(float(v), 1) for v in dp],
            "criterion": "rot_mirrored" if kind in MIRROR_JOINTS else "pos_same",
            "expect": (
                "左右反号(镜像)" if kind in MIRROR_JOINTS else "左右同号"
            ),
        })

    doc = {
        "schema": "dm10-model-direction/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xml": xml_path,
        "delta_rad": delta_rad,
        "frame": "base_link（机体坐标系；x=前, y=左, z=上）",
        "sign_convention": (
            "脚相对机体的旋转/位移是【相对基线的变化量】。"
            "判定某个 sign 取 +1 还是 −1：把 sign 乘进模型量，看能否复现实测方向。"
        ),
        "model_warning": (
            "左右腿关节轴向量完全相同（不镜像）：髋roll/yaw 给正角时两腿往同一侧倒。"
            "这是模型事实，不是接线错误。"
        ),
        "entries": entries,
    }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 dm10 模型期望方向表")
    ap.add_argument("--xml", default=XML_DEFAULT)
    ap.add_argument("--delta-rad", type=float, default=0.4, help="探针幅度（默认 0.4≈23°）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print", dest="show", action="store_true", help="打印到终端")
    args = ap.parse_args()

    doc = build(args.xml, args.delta_rad)

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "dm10_model_directions.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"→ {out}")

    if args.show:
        print(f"\n探针 +{args.delta_rad} rad ({args.delta_rad * 57.29578:.1f}°)"
              f"，脚相对机体的变化（x=前 y=左 z=上）\n")
        print(f"{'#':<3}{'关节':<12}{'轴':<4}{'限位':<18}{'位移(mm)':<24}"
              f"{'旋转(roll,pitch,yaw)°':<26}{'判据'}")
        print("-" * 104)
        for e in doc["entries"]:
            dp = e["foot_pos_delta_mm"]
            rp = e["foot_rot_rpy_deg"]
            lim = f"[{e['limit_rad'][0]:+.2f},{e['limit_rad'][1]:+.2f}]"
            print(f"{e['index']:<3}{e['label']:<12}{e['model_axis']:<4}{lim:<18}"
                  f"({dp[0]:+7.1f},{dp[1]:+7.1f},{dp[2]:+7.1f})   "
                  f"({rp[0]:+6.1f},{rp[1]:+6.1f},{rp[2]:+6.1f})   "
                  f"{'左右反号' if e['criterion'] == 'rot_mirrored' else '左右同号'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
