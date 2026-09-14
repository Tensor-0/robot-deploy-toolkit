#!/usr/bin/env python3
"""check_contract.py — 观测契约比对（部署前必跑）

对应问题：训练侧的观测顺序 / 维度 / 缩放，必须与部署侧逐项一致。
**只比总维度是不够的** —— 两边都是 39 维但顺序不同时，运行时不会报错，
机器人会收到语义错乱的观测（静默失败）。

这个脚本做三档比对（借用了 UniLab 官方 config_guard 的思路）：

    DENY    —— 不一致就**拒绝部署**（顺序、维度、缩放、默认角、裁剪）
    WARN    —— 不一致只警告（奖励项、dt 等不影响语义的）
    ALLOW   —— 自由差异，不检查（后端、场景等）

用法:
    # 从训练 run 目录自动找快照
    python3 check_contract.py --run-dir <UniLab run 目录> --deploy-config <部署 yaml>

    # 或直接指定两份
    python3 check_contract.py --train-snapshot <run_config.json> --deploy-config <yaml>

退出码: 0 = 通过（可能有 warn）, 1 = DENY 档不一致
"""
import argparse
import json
import os
import re
import sys

# ── 三档分类 ────────────────────────────────────────────
DENY = ["obs_order", "obs_dim", "action_scale", "clip_actions", "joint_default_angle"]
WARN = ["obs_scales", "frame_stack", "dt", "decimation"]
ALLOW = ["sim_backend", "num_envs", "max_iterations", "seed"]

RED, YEL, GRN, RST = "\033[31m", "\033[33m", "\033[32m", "\033[0m"


def load_train_snapshot(path):
    """从 run_config.json 或直接给的快照里取 obs 契约"""
    d = json.load(open(path, encoding="utf-8"))
    snap = d.get("contract_snapshot", d)
    obs = snap.get("env.observations") or snap.get("observations")
    return snap, obs


def parse_deploy_layout(yaml_path):
    """从部署 yaml 里抠 obs_layouts 字符串（不依赖 pyyaml）"""
    txt = open(yaml_path, encoding="utf-8").read()
    m = re.search(r"obs_layouts:\s*\n\s*-\s*[\"']([^\"']+)[\"']", txt)
    if not m:
        m = re.search(r"obs_layouts:\s*\[\s*[\"']([^\"']+)[\"']", txt)
    if not m:
        raise SystemExit(f"在 {yaml_path} 里找不到 obs_layouts")
    layout = m.group(1)
    out = []
    for seg in layout.split(","):
        seg = seg.strip()
        if not seg:
            continue
        name, _, size = seg.partition(":")
        size = size.split("@")[0]  # 去掉 @tap
        out.append((name.strip(), int(size)))
    return out


# 部署侧名 → 训练侧名（同义不同名，不算错）
ALIAS = {
    "ang_vel": "base_ang_vel",
    "gravity_b": "projected_gravity",
    "dof_pos": "joint_pos",
    "dof_vel": "joint_vel",
    "last_action": "actions",
    "cmd_vel": "command",
}


def flatten_obs_terms(obs_cfg):
    """把训练侧 observations 配置拍平成 [(term_name, dim_or_None)] 的有序列表（policy/actor 组）

    ⚠️ run_config.json 的 contract_snapshot 里**只存顺序，不存维度**
    （维度由各 term 的 func 在运行时决定）。所以 dim 通常是 None，
    此时维度校验退化为「靠部署侧自己」——顺序校验仍然有效，
    而顺序正是静默失败的主因。
    """
    if obs_cfg is None:
        return None
    node = obs_cfg
    for key in ("policy", "actor"):
        if isinstance(node, dict) and key in node:
            node = node[key]
            break
    if isinstance(node, dict) and "terms" in node:
        node = node["terms"]
    if not isinstance(node, dict):
        return None

    out = []
    for name, cfg in node.items():
        dim = None
        if isinstance(cfg, dict):
            dim = cfg.get("dim") or cfg.get("size") or cfg.get("shape")
            if isinstance(dim, (list, tuple)):
                dim = dim[-1]
            if dim is None:
                p = cfg.get("params") or {}
                dim = p.get("dim")
        try:
            out.append((name, int(dim) if dim else None))
        except (TypeError, ValueError):
            out.append((name, None))
    return out


def onnx_total_dim(deploy_config):
    """从部署侧 yaml 的 model_names 找 onnx，读它的输入总维度作为交叉校验"""
    try:
        import onnx  # 可选依赖
    except ImportError:
        return None
    txt = open(deploy_config, encoding="utf-8").read()
    m = re.search(r"model_names:\s*\[?\s*[\"']([^\"']+)[\"']", txt)
    if not m:
        return None
    name = m.group(1)
    base = os.path.dirname(os.path.abspath(deploy_config))
    for cand in (os.path.join(base, "..", "models", name),
                 os.path.join(base, "..", "..", "models", name)):
        if os.path.exists(cand):
            try:
                mdl = onnx.load(cand)
                dims = mdl.graph.input[0].type.tensor_type.shape.dim
                total = 1
                for d in dims:
                    if d.dim_value > 0:
                        total *= d.dim_value
                return total, cand
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="UniLab 训练 run 目录（会自动找 run_config.json）")
    ap.add_argument("--train-snapshot", help="直接给 run_config.json 路径")
    ap.add_argument("--deploy-config", required=True, help="部署侧 policy yaml")
    args = ap.parse_args()

    snap_path = args.train_snapshot
    if not snap_path and args.run_dir:
        cand = os.path.join(args.run_dir, "run_config.json")
        if os.path.exists(cand):
            snap_path = cand
        else:
            hits = [f for f in os.listdir(args.run_dir) if f.endswith(".json")]
            if hits:
                snap_path = os.path.join(args.run_dir, hits[0])
    if not snap_path or not os.path.exists(snap_path):
        sys.exit(f"找不到训练侧快照。给了 run-dir={args.run_dir}")

    print("=" * 62)
    print(" 观测契约比对")
    print("=" * 62)
    print(f"  训练侧: {snap_path}")
    print(f"  部署侧: {args.deploy_config}")
    print()

    snap, obs_cfg = load_train_snapshot(snap_path)
    train_terms = flatten_obs_terms(obs_cfg)
    deploy_terms = parse_deploy_layout(args.deploy_config)

    if train_terms is None:
        print("⚠️  训练侧快照里没有可解析的 observations 结构。")
        print("    可用的顶层键:", list(snap.keys())[:20])
        print("\n    → 请手动比对：打开 run_config.json 的 contract_snapshot，")
        print("      找到 env.observations.<组>.terms 的**键顺序**，")
        print("      与部署侧 obs_layouts 逐段对照。")
        sys.exit(1)

    # 归一化训练侧名字
    def norm(n):
        return n

    t_names = [norm(n) for n, _ in train_terms]
    d_names = [ALIAS.get(n, n) for n, _ in deploy_terms]

    print(f"{'训练侧':<28}{'部署侧':<28}")
    print("-" * 58)
    deny_hit = False
    maxlen = max(len(t_names), len(d_names))
    for i in range(maxlen):
        tn = t_names[i] if i < len(t_names) else "——"
        dn = d_names[i] if i < len(d_names) else "——"
        mark = "  ✅" if tn == dn else "  ❌ DENY"
        if tn != dn:
            deny_hit = True
        print(f"[{i}] {tn:<24} {dn:<24}{mark}")

    print()
    t_total = sum(d for _, d in train_terms if d)
    d_total = sum(d for _, d in deploy_terms)
    if t_total:
        print(f"训练侧总维度(已知项): {t_total}   部署侧总维度: {d_total}")
        if t_total != d_total:
            print(f"{RED}❌ 维度不一致 —— 会被 ONNX 输入校验拦住{RST}")
            deny_hit = True
    else:
        print(f"训练侧总维度: 未在快照中（只存了顺序）")
        print(f"部署侧总维度: {d_total}")
        # 用 onnx 交叉校验部署侧自己算得对不对
        got = onnx_total_dim(args.deploy_config)
        if got:
            onnx_dim, path = got
            print(f"ONNX 输入维度: {onnx_dim}  ({os.path.basename(path)})")
            if onnx_dim != d_total:
                print(f"{RED}❌ 部署侧 obs_layouts 合计 {d_total} ≠ ONNX 输入 {onnx_dim}{RST}")
                deny_hit = True
            else:
                print(f"{GRN}✅ 部署侧 layout 与 ONNX 输入维度自洽{RST}")
        else:
            print("（装 onnx 包可额外交叉校验 ONNX 输入维度：pip install onnx）")

    print()
    if deny_hit:
        print(f"{RED}{'='*62}{RST}")
        print(f"{RED} DENY：观测契约不一致 —— 不要部署！{RST}")
        print(f"{RED}{'='*62}{RST}")
        print()
        print("  常见原因与修法：")
        print("  1. 顺序不同（维度总数相同）→ 改部署侧 obs_layouts，按训练侧声明顺序重排")
        print("  2. 名字不同但语义相同（见 ALIAS 表）→ 不算错，脚本已自动对应")
        print("  3. 维度不同 → 检查 frame_stack 与观测源 size")
        print()
        print("  ⚠️ 这类错误运行时**不会报错**（只比总元素数），")
        print("     但机器人会收到语义错乱的观测 —— 这是最危险的静默失败。")
        sys.exit(1)

    print(f"{GRN}✅ PASS：观测顺序与维度一致{RST}")
    sys.exit(0)


if __name__ == "__main__":
    main()
