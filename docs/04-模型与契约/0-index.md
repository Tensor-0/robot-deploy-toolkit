# 阶段 4 · 模型与观测契约

> **目标：让部署侧收到的观测，与训练侧假设的观测，逐项一致。**
> 这是整个流程里**最容易出错、也最不报错**的一环。

---

## 一、为什么这一环最危险

> **因为维度对了，它就不报错。**

**运行时的校验只有一条**：
```cpp
// inference_node.cpp:85-108
if (obs_total_elements != onnx_input_size) throw "ONNX input size mismatch";
```

**⇒ 只比总元素数。**

**后果**：如果训练侧的 tensor 顺序变了、而部署配置没改，
**只要总维度不变（如 39 = 39），运行时不会报错**，
但机器人会收到**语义错乱的观测** —— 它以为的「速度指令」其实是「关节位置」。

**打个比方**：
> 像给一个只会按固定格子填表的人发指令。表格确实是 39 格、一格不多一格不少，
> 但你把「姓名」填进了「年龄」栏。**表交上去了，系统不报错，但内容全乱了。**

---

## 二、⭐ 五样必须手工对齐的东西

`policy.onnx` **只含网络**，下面这些**都不在里面**：

| # | 项 | 训练侧位置 | 部署侧位置 | 有校验吗 |
|---|---|---|---|---|
| **1** | **观测顺序 + 维度** | `base.yaml` 的 term 声明顺序 | `default.yaml` 的 `obs_layouts` 字符串 | ❌ **只比总元素数** |
| **2** | 动作缩放 | `actions.joint_pos.scale` | `action_scale` | ❌ |
| **3** | 动作裁剪 | 环境侧 `clip_actions` | `clip_actions` | ❌ |
| **4** | 关节默认角 | 环境侧 `joint_default_angle` | `joint_default_angle` | ❌ |
| **5** | PD 增益 | 仿真执行器 | `robot.yaml` 的 `kp/kd` | ❌ |
| — | 归一化 | **烘焙进 onnx** | **不做** | ✅ 一致 |

> ⚠️ **这张表就是「静默失败」的温床。**

---

## 三、观测顺序：最常见的错

### 3.1 一个真实的例子【实测】

**DM10 的配置对不上**（2026-09 实测）：

| | 顺序（都是 39 维）|
|---|---|
| **训练侧** `UniLab/.../dm10_joystick_flat/base.yaml:44-83` | `base_ang_vel(3) → projected_gravity(3) → joint_pos(10) → joint_vel(10) → actions(10) → command(3)` |
| **部署侧** `roboparty_deploy/.../dm10/configs/default.yaml:5` | `ang_vel(3) → gravity_b(3) → cmd_vel(3) → dof_pos(10) → dof_vel(10) → last_action(10)` |

**逐段对照**：

| 训练侧 | 部署侧 | 判定 |
|---|---|---|
| `[0:3]` base_ang_vel | `[0:3]` ang_vel | ✅ 同义（只是命名不同）|
| `[3:6]` projected_gravity | `[3:6]` gravity_b | ✅ 同义 |
| `[6:16]` **joint_pos** | `[6:9]` **cmd_vel** | ❌ **错位** |
| `[16:26]` **joint_vel** | `[9:19]` **dof_pos** | ❌ **错位** |
| `[26:36]` **actions** | `[19:29]` **dof_vel** | ❌ **错位** |
| `[36:39]` **command** | `[29:39]` **last_action** | ❌ **错位** |

> ⚠️ **注意前两段：命名不同但位置对，不算错。**
> **不要把「命名差异」和「顺序错位」混为一谈** —— 否则会去改不该改的地方。

### 3.2 怎么修

**改部署侧的一行字符串**：

```yaml
# 改前
- "ang_vel:3, gravity_b:3, cmd_vel:3, dof_pos:10, dof_vel:10, last_action:10"
# 改后（按训练侧声明顺序）
- "ang_vel:3, gravity_b:3, dof_pos:10, dof_vel:10, last_action:10, cmd_vel:3"
```

**⚠️ 改的是「顺序」，不是「名字」** —— 部署侧有自己的源名，与训练侧不同但语义对应。

### 3.3 名字对照表

| 训练侧 term | 部署侧 obs 源名 | 语义 |
|---|---|---|
| `base_ang_vel` | `ang_vel` | 基座角速度 |
| `projected_gravity` | `gravity_b` | 重力投影（姿态）|
| `joint_pos` | `dof_pos` | 关节位置 |
| `joint_vel` | `dof_vel` | 关节速度 |
| `actions` | `last_action` | 上一帧原始动作 |
| `command` | `cmd_vel` | 速度指令 |

---

## 四、⭐ 根治方案：契约快照 + 三档守卫

**改一行字符串只解决「这一次」。真正的解法是防呆机制。**

### 4.1 好消息：基础已经现成

**训练跑完会自动落盘 `run_config.json`**，里面 `contract_snapshot` **逐字保存了配置契约**：

```
<UniLab run 目录>/run_config.json
  └── contract_snapshot
        ├── env.observations          ← ⭐ term 顺序（机器生成的，不会骗人）
        ├── env.actions
        ├── algo.obs_groups
        ├── algo.policy.actor_hidden_dims
        └── env.ctrl_dt
```

> **⇒ 不用新写 dump 逻辑，只要让部署侧去读它、比对。**

### 4.2 三档守卫（借用 UniLab 官方设计）

| 档 | 行为 | 包含字段 |
|---|---|---|
| **DENY** | 不一致就**拒绝部署** | 观测顺序、维度、`action_scale`、`clip_actions`、`joint_default_angle` |
| **WARN** | 只警告 | `obs_scales`、`frame_stack`、`dt`、`decimation` |
| **ALLOW** | 不检查 | 后端、并行度、随机种子 |

**为什么分三档**：
> **有些差异是致命的，有些是无害的，有些是必须的。**
> 一刀切全拒会让人绕过检查；全放过等于没检查。

### 4.3 用工具

```bash
python3 scripts/check_contract.py \
    --run-dir <UniLab run 目录> \
    --deploy-config <部署 policy yaml>
```

**输出示例**（真实跑 DM10 的结果）：

```
训练侧                     部署侧
[0] base_ang_vel         base_ang_vel              ✅
[1] projected_gravity    projected_gravity         ✅
[2] joint_pos            command                   ❌ DENY
...
 DENY：观测契约不一致 —— 不要部署！
退出码: 1
```

---

## 五、⭐ 没接真机时怎么验证

**这是本阶段最有用的技巧：用「支架上站立」抓顺序错误。**

**推理**：静止站立时，各段有**期望量级**：

| 段 | 期望值 | 为什么 |
|---|---|---|
| `ang_vel` | **≈ 0** | 没在转 |
| `gravity_b` | **≈ (0, 0, -1)** | 直立 |
| `dof_pos` | ≈ `joint_default_angle`（如 `-0.4, 0, 0, 0.8, -0.4`）| 保持默认姿势 |
| `dof_vel` | **≈ 0** | 没在动 |
| `last_action` | ≈ 0 附近 | 还没动过 |
| `cmd_vel` | **≈ 0** | 摇杆中位 |

> **⇒ 如果某一段出现了"它不该有的数值"，就是顺序错了。**
>
> **典型症状**：`cmd_vel` 段出现 `0.8` 这样的大数
> —— 那是关节角（`joint_default_angle` 的第 4 项就是 0.8）跑进来了。

**配合上机四步的第①步**（力矩关闭、只看观测）—— 这是最佳时机。

---

## 六、另外四样对齐项

### 6.1 动作缩放（`action_scale`）

```yaml
# 训练侧
actions:
  joint_pos:
    scale: 0.25
```
```yaml
# 部署侧
action_scale: [0.25]
```

**⚠️ 部署侧支持单值自动展开**（写 `[0.25]` 会展开成 N 维）。

### 6.2 关节默认角（`joint_default_angle`）

**部署侧动作的最后一步**：
```cpp
act[usd2urdf[i]] = a * action_scale + joint_default_angle;
```

> ⚠️ **最后那个「加默认角」如果忘了**，机器人会以**直腿姿态**去执行走路目标 —— **当场跪下去**。
> 因为网络输出的是**相对量**（相对屈膝站立位）。

### 6.3 动作裁剪（`clip_actions`）

**⚠️ 各策略不统一**：

| 策略 | `clip_actions` |
|---|---|
| rpo 全部 | 100.0（≈不限）|
| parkour | 10.0 |
| dm10 | 18.0 |

### 6.4 PD 增益（`kp` / `kd`）

**⚠️ 这里是 sim2real 的第二个主战场** —— 因为**仿真 PD 和真机 PD 语义不同**：

| | 仿真（`<position>` 执行器）| 真机（DM MIT 模式）|
|---|---|---|
| PD 在哪算 | 物理引擎内部 | **电机固件里** |
| 用什么状态 | 步末状态（隐式约束 `e_{n+1}=0`）| **当前状态** |
| 效果 | 大增益也稳 | **大增益会振荡** |

> **⇒ 仿真调好的 kp=30，真机上会振荡。别从仿真的值起步。**

**【实测】DM10 的起点值**（来自达妙官方 sim2real 配置）：
```yaml
kp: [16, 20, 20, 18, 20, 16, 20, 20, 18, 20]
kd: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
```

---

## 七、过关判据

| # | 判据 | 怎么测 |
|---|---|---|
| **1** | `check_contract.py` **退出码为 0** | 跑脚本 |
| **2** | **五样对齐项逐项对照过**（不是"应该没问题"）| 人工过一遍表格 |
| **3** | 支架站立时，**各段观测的数值量级合理** | 上机四步第①步 |

---

## 八、常见症状

| 症状 | 最可能原因 |
|---|---|
| 支架上就有异常数值 | **obs 顺序错** |
| 动作幅度整体偏大/偏小 | `action_scale` 不一致 |
| 以直腿姿态执行走路 | **忘了加 `joint_default_angle`** |
| 仿真稳、真机振荡 | **PD 语义差异**（不是数值问题）|
| 一上真机就摔 | **几乎总是三者之一**：关节顺序 / 动作缩放单位 / 观测布局 |

---

## 工具

```bash
# 契约比对（核心）
python3 scripts/check_contract.py --run-dir <run目录> --deploy-config <yaml>

# 可选：装 onnx 以额外交叉校验 ONNX 输入维度
pip install onnx
```
