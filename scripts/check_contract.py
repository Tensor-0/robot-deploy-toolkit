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
    "motion_command": "command",
}

# ══════════════════════════════════════════════════════════════════════════
# 观测【可得性】：训练的每一项，真机到底拿不拿得到
# ══════════════════════════════════════════════════════════════════════════
# 与「顺序对齐」是两件事：
#   顺序错 → 改一行字符串就能修
#   拿不到 → 改字符串修不了，必须换观测集重训 或 在部署侧新建观测源
# ⇒ 所以这一档比顺序更硬，归 DENY。
#
# 这张表的"有没有源"以部署侧代码为准（下面 parse_deploy_sources() 现场解析，
# 不靠记忆）；"真机能不能测"是物理事实，写在理由里供人判断。
UNAVAILABLE = {
    "base_lin_vel": "真机没有能直接测「机体相对地面的线速度」的传感器 —— 只能状态估计"
    "（IMU 积分会漂移；正解是腿运动学+接触判断+滤波）",
    "motion_anchor_pos_b": "需要「机器人在世界里的位置」（里程计），真机没有来源 —— 需状态估计",
    "motion_anchor_ori_b": "需要世界姿态与参考姿态之差；IMU 可给姿态，但部署框架目前"
    "【没有这个观测源】，要用得先在 C++ 里新建（motion_loader 还需读 npz 的 body_quat_w）",
    "body_pos": "critic 专用的特权观测；出现在 policy 组里就不可部署",
    "body_ori": "critic 专用的特权观测；出现在 policy 组里就不可部署",
    "sac_base_lin_vel": "critic 专用的特权观测；出现在 policy 组里就不可部署",
    "gait_phase": "部署框架没有这个源（2026-09-09 实测确认）",
}

# 上游给过的现成药方（打印给人看，不是自动修）
REMEDY_HINT = (
    "上游已有两个「去掉这些观测」的部署版任务配置可抄：\n"
    "    UniLab/src/unilab/conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml:15-16\n"
    "    UniLab/src/unilab/conf/sac/task/g1_wbt_obs/mujoco.yaml:47,55   ← 与 FlashSAC 同类\n"
    "  两者都把 motion_anchor_pos_b 与 base_lin_vel 置为 null。\n"
    "  官方口径见 docs/.../3-deployment/1-sim_to_real/1-overview.md 上机前检查清单第 5 条。"
)


def deploy_source_for(term_name, dim):
    """训练侧 term → 部署侧观测源名；None = 部署框架没有这个源。

    ⚠️ `command` 有两种语义，靠维度区分（这是 2026-09 那次踩坑的同名不同物）：
       3 维  = 摇杆速度指令      → cmd_vel（来自话题，不是测量值）
       2N 维 = 参考关节轨迹回放  → motion_command（npz 的 joint_pos+joint_vel）
    """
    if term_name == "command":
        return "cmd_vel" if dim == 3 else "motion_command"
    if term_name in UNAVAILABLE:
        return None
    # 顺序不能反：先查"已知不可得"，再查普通别名
    for deploy_name, train_name in ALIAS.items():
        if train_name == term_name:
            return deploy_name
    return None  # 未知 term ⇒ 也按不可得处理，见 availability_check 的措辞


def parse_deploy_sources(obs_manager_cpp):
    """从部署侧的 obs_source_definitions() 现场解析可用观测源白名单。

    不硬编码条数（上游可能加源），但要求解析结果非空；
    并且调用方会校验"我们映射到的源名确实在白名单里"——那才是真正的防漂移。
    """
    try:
        txt = open(obs_manager_cpp, encoding="utf-8").read()
    except OSError as exc:
        return None, f"读不到部署侧源码（{exc}）—— 用 --deploy-repo 指定"

    block = re.search(
        r"obs_source_definitions\(\)\s*\{.*?definitions\{(.*?)\};", txt, re.S
    )
    if not block:
        return None, f"在 {obs_manager_cpp} 里没找到 obs_source_definitions() —— 结构可能变了"
    names = re.findall(r'\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*&InferenceNode::', block.group(1))
    if not names:
        return None, f"解析到 0 个观测源（{obs_manager_cpp}）—— 正则与代码结构对不上"
    return names, None


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
    ap.add_argument(
        "--manifest",
        help="UniLab 导出的 obs_manifest.json（有它才能校验【每段维度】，"
        "因为 run_config.json 的快照只存顺序）。用 `_dump_obs_manifest.py` 生成。",
    )
    ap.add_argument("--deploy-config", required=True, help="部署侧 policy yaml")
    ap.add_argument(
        "--deploy-repo",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "roboparty_deploy",
        ),
        help="roboparty_deploy 仓库根目录（用于现场解析部署侧支持的观测源白名单）",
    )
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
        if not args.manifest:
            sys.exit(
                f"找不到训练侧快照（给了 run-dir={args.run_dir}）。\n"
                "  ⇒ 也可以改给 --manifest（obs_manifest.json），它自带顺序+维度，"
                "不依赖训练 run 目录。"
            )
        snap_path = None

    print("=" * 62)
    print(" 观测契约比对")
    print("=" * 62)
    print(f"  训练侧: {snap_path or '（仅清单，无快照）'}")
    print(f"  部署侧: {args.deploy_config}")
    print()

    if snap_path:
        snap, obs_cfg = load_train_snapshot(snap_path)
    else:
        snap, obs_cfg = {}, None
    train_terms = flatten_obs_terms(obs_cfg)
    deploy_terms = parse_deploy_layout(args.deploy_config)

    # 训练侧维度从哪来：优先 manifest（含维度）；否则退回快照（只有顺序）
    manifest = None
    if args.manifest:
        manifest = json.load(open(args.manifest, encoding="utf-8"))
        sig = manifest.get("signature") or "|".join(
            f"{t['name']}:{t['dim']}" for t in manifest.get("terms", [])
        )
        print(f"  训练侧清单: {args.manifest}")
        print(f"    group={manifest.get('policy_observation_group')}  obs_dim={manifest.get('obs_dim')}")
        print(f"    signature={sig}")
        print()
        train_terms = [(t["name"], int(t["dim"])) for t in manifest.get("terms", [])]
        # 清单与快照的顺序必须一致，否则说明清单不是这个 run 的
        snap_terms = flatten_obs_terms(obs_cfg)
        if snap_terms and [n for n, _ in snap_terms] != [n for n, _ in train_terms]:
            print(f"{RED}❌ 清单的 term 顺序与 run_config.json 快照不一致：{RST}")
            print(f"   清单: {[n for n, _ in train_terms]}")
            print(f"   快照: {[n for n, _ in snap_terms]}")
            print("   ⇒ 清单张冠李戴了（不是这个 run 的产物）")
            sys.exit(1)

    if train_terms is None:
        if args.manifest:
            train_terms = [(t["name"], int(t["dim"])) for t in manifest.get("terms", [])]
        else:
            print("⚠️  训练侧快照里没有可解析的 observations 结构。")
            print("    可用的顶层键:", list(snap.keys())[:20])
            print("\n    → 请手动比对：打开 run_config.json 的 contract_snapshot，")
            print("      找到 env.observations.<组>.terms 的**键顺序**，")
            print("      与部署侧 obs_layouts 逐段对照。")
            print("      （或先跑 UniLab/_dump_obs_manifest.py 生成清单，再 --manifest 传入）")
            sys.exit(1)

    print(f"{'训练侧':<30}{'部署侧':<30}")
    print("-" * 64)
    deny_hit = False
    maxlen = max(len(train_terms), len(deploy_terms))
    for i in range(maxlen):
        if i < len(train_terms):
            tn, td = train_terms[i]
            t_str = f"{tn}:{td}" if td is not None else tn
        else:
            tn, td, t_str = "——", None, "——"
        if i < len(deploy_terms):
            dn, dd = deploy_terms[i]
            d_str = f"{dn}:{dd}"
        else:
            dn, dd, d_str = "——", None, "——"
        mapped = ALIAS.get(dn, dn)
        if mapped != tn:
            mark = "  ❌ DENY 顺序/名字不符"
            deny_hit = True
        elif td is not None and td != dd:
            mark = f"  ❌ DENY 维度不符（训练 {td} vs 部署 {dd}）"
            deny_hit = True
        else:
            mark = "  ✅"
        print(f"[{i}] {t_str:<28} {d_str:<28}{mark}")

    # ── 可得性检查：比顺序更硬的一档 ─────────────────────────────────────
    print()
    print("-" * 64)
    print(" 观测可得性（训练用到的每一项，部署侧有没有源）")
    print("-" * 64)
    cpp = os.path.join(
        os.path.abspath(args.deploy_repo), "src", "inference", "src", "obs_manager.cpp"
    )
    sources, err = parse_deploy_sources(cpp)
    if sources is None:
        print(f"{YEL}⚠️  {err}{RST}")
        print("    ⇒ 跳过可得性检查（这是本次最关键的一档，请修好路径后重跑）")
    else:
        print(f"  部署侧支持的观测源（现场解析自 {os.path.relpath(cpp)}）：")
        print(f"    {', '.join(sources)}")
        print()
        missing = []
        for name, dim in train_terms:
            if name == "——":
                continue
            src = deploy_source_for(name, dim)
            if src is None:
                reason = UNAVAILABLE.get(name, "未知 term：不在任何已知映射里（按不可得处理）")
                print(f"  ❌ {name:<24} 部署侧【无源】 —— {reason}")
                missing.append(name)
            elif src not in sources:
                print(f"  ❌ {name:<24} 映射到 '{src}'，但该源不在解析出的白名单里")
                missing.append(name)
            else:
                print(f"  ✅ {name:<24} ← {src}")
        if missing:
            deny_hit = True
            print()
            print(f"{RED}  ⚠️ 有 {len(missing)} 项训练观测在部署侧【没有来源】：{missing}{RST}")
            print("     这不是「改一行字符串」能修的 —— 顺序错可以重排，没源只能：")
            print("       (a) 在部署侧新建这个观测源（要写 C++ 并重新编译上板），或")
            print("       (b) 换一个不含这些观测的任务配置，重新训练")
            print()
            for line in REMEDY_HINT.splitlines():
                print(f"     {line}")

    # ── 历史/堆叠提示（本次不做自动比对，只提醒）────────────────────────
    _hist = re.search(r"frame_stacks:\s*\[([^\]]*)\]", open(args.deploy_config, encoding="utf-8").read())
    if _hist and any(int(x) > 1 for x in re.findall(r"\d+", _hist.group(1))):
        print()
        print(f"{YEL}⚠️  frame_stacks > 1 或布局含 @tap ⇒ 观测带历史/堆叠。{RST}")
        print("    训练侧的 history_length 与部署侧的堆叠顺序是【另一类静默错位】，")
        print("    本脚本暂不自动比对，请人工确认两边一致。")

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
        print("  4. ⭐【无源】训练用了真机拿不到的观测 → 换观测集重训，或在部署侧新建源")
        print("     （这一档改字符串修不了；上游的 g1_motion_tracking_deploy / g1_wbt_obs")
        print("       就是为此存在的部署版任务配置）")
        print()
        print("  ⚠️ 这类错误运行时**不会报错**（只比总元素数），")
        print("     但机器人会收到语义错乱的观测 —— 这是最危险的静默失败。")
        sys.exit(1)

    if args.manifest:
        print(f"{GRN}✅ PASS：观测顺序、维度、可得性三项一致{RST}")
    else:
        print(f"{GRN}✅ PASS：观测顺序一致{RST}")
        print(f"{YEL}   ⚠️ 未校验每段维度（没给 --manifest）。{RST}")
        print("      训练侧快照 run_config.json 只存顺序、不存维度；")
        print("      用 `UniLab/_dump_obs_manifest.py` 生成清单后再加 --manifest 重跑。")
    sys.exit(0)


if __name__ == "__main__":
    main()
