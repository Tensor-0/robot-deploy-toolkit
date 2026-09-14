# robot-deploy-toolkit

> **从新机器人本体到安全跑起来的完整流程指南 + 可执行工具。**
> 面向：拿到一台新机器人、想让它真正动起来的人（尤其是独立开发者）。

---

## 为什么有这个仓库

**调研过的事实**（2026-09）：

| 检查对象 | 结果 |
|---|---|
| 本机 | **零份**「上机前检查清单」，**零个**部署自检脚本 |
| 达妙 `bipedal-robot` 官方仓库 | 全仓搜「安全」**零命中** |
| `unitree_rl_gym` / `humanoid-gym` / `unitree_rl_lab` | README 里**都没有** checklist 和安全章节 |
| `roboparty_deploy` | 安全内容 **100% 外链到飞书**，仓库内为零 |

**⇒ 这不是重复造轮子，是填一个整个生态都没填的坑。**

---

## ⚠️ 先读这个

**如果你今天第一次上机，只读两件事：**

1. **[安全须知（首次上机必读）](docs/05-安全保底/1-安全须知（首次上机必读）.md)**
2. **[上机四步](docs/07-上机四步/0-index.md)**

**核心安全警示**（来自官方《萝博头原型机安全操作指南 V1.1.1》）：

> - **高动态风险**：机器人在步态调试中存在极高的失稳倾倒风险。
>   **任何悬空标零或步态推理测试，必须在配备承重安全吊架（防摔保护）的受控场地内进行。**
> - **物理急停至上**：遇到任何不可控的关节抖动、步态偏离或结构干涉，
>   请立即按遥控手柄的 **"X"** 键触发软件失能；**若软件失控，请立即拍下物理急停开关切断主电源。**

---

## 流程总图

```
阶段 0   硬件齐套        ← 通电前：急停在哪？支撑好了吗？人撤了吗？
   ↓
阶段 1   通讯打通        ← 能读到电机反馈，能解释每个字节
   ↓
阶段 2   标定            ← 零位 / 方向 / 限位（三个都不能跳）
   ↓
阶段 3   单关节稳停      ← 单个关节能停住，再谈多关节
   ↓
阶段 4   模型与契约      ← URDF/MJCF + 观测契约对齐
   ↓
阶段 5   安全保底        ← PD 站立 + 吊装 + 软垫 + 人看着
   ↓
阶段 6   训练            ← "跑几十次，每次改一个变量"
   ↓
阶段 7   上机四步        ← 支架(力矩关) → 手扶 → 半速 → 全速
```

> **⚠️ 注意一个残酷的比例**：**阶段 0-3 占了全部工作量的约一半，而它们和强化学习一点关系都没有。**
> 新手 90% 的时间会花在这里，却以为自己在做阶段 6-7。

---

## 导航

| 阶段 | 文档 | 工具 |
|---|---|---|
| 0 准备 | [环境与工具链 · 硬件清单](docs/00-准备/0-index.md) | [`check_preflight.sh`](scripts/check_preflight.sh) |
| 1 通讯 | [CAN / DM 协议 / IMU](docs/01-通讯打通/0-index.md) | — |
| 2 标定 | [零位 · 方向 · 限位](docs/02-标定/0-index.md) | [`direction_check.py`](scripts/direction_check.py) · [`limit_check.py`](scripts/limit_check.py) |
| 3 单关节 | [逐关节测试 · kp/kd 整定](docs/03-单关节稳停/0-index.md) | [`joint_test.py`](scripts/joint_test.py) · [`gain_tune.py`](scripts/gain_tune.py) |
| 4 契约 | [观测契约对齐](docs/04-模型与契约/0-index.md) | [`check_contract.py`](scripts/check_contract.py) |
| 5 安全 | [**安全须知（必读）**](docs/05-安全保底/1-安全须知（首次上机必读）.md) | [`safety_inject_test.py`](scripts/safety_inject_test.py) |
| 6 训练 | [训练与交接清单](docs/06-训练/0-index.md) | — |
| 7 上机 | [上机四步](docs/07-上机四步/0-index.md) | — |
| 8 排障 | [故障排查（症状索引）](docs/08-故障排查/0-index.md) | — |
| 9 参考 | [参数速查](docs/09-参考/参数速查表.md) · [已知限制](docs/09-参考/已知限制.md) · [经验教训](docs/09-参考/LESSONS_LEARNED.md) | — |

---

## 脚本

**所有脚本都是只读或需显式确认才动作的。** 上机前请先跑自检。

```bash
# 上电前自检（不需要接电机）
./scripts/check_preflight.sh

# 观测契约比对（训练产物 vs 部署配置）
python3 scripts/check_contract.py --run-dir <训练run目录> --deploy-config <部署yaml>

# 单关节逐台测试（结果落盘）
python3 scripts/joint_test.py --config <robot.yaml> --out results/joint_test.json

# kp/kd 整定（阶跃响应 + 采样记录）
python3 scripts/gain_tune.py --config <robot.yaml> --joint 3 --kp-range 10,40
```

---

## 本仓库的写作约定

| 标记 | 含义 |
|---|---|
| **【官方】** | 来自《萝博头原型机安全操作指南 V1.1.1》或厂商文档 |
| **【实测】** | 在 DM10 上真实测过的 |
| **【推断】** | 基于证据的推理，**不是任何文档的原话** |
| **⚠️ 硬阻塞** | **改脚本也绕不过去**，必须改 C++ 或用厂商上位机 |

---

## 适用性

本仓库用 **DM10（达妙双足人形，10 DoF 下肢）+ roboparty_deploy** 作为**完整算例**，
但流程骨架是通用的。换机器人时：

1. 替换 `docs/00-准备/2-硬件清单与接线.md` 的硬件参数
2. 替换 `docs/09-参考/参数速查表.md`
3. 脚本大部分不用改（它们读配置，不硬编码）

---

## 相关资源

| 资源 | 链接 |
|---|---|
| 官方安全操作指南 | [飞书 wiki](https://roboparty.feishu.cn/wiki/ZGtnwpHCjii2XykBYMGchoBBnSl) |
| 官方文档站 | https://roboparty.com/roboto_origin/doc |
| 官方仓库 | https://github.com/Roboparty/roboto_origin |
| 部署框架 | https://github.com/Roboparty/roboparty_deploy |
| 训练框架 | UniLab（清华 AIR）|

---

## 许可证

本仓库内容为整理与原创工具。引用的官方文档版权归原作者所有。
