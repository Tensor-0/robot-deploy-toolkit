# 阶段 0 · 准备

> **目标：通电之前，软件环境和硬件状态都是确定的。**

---

## 一、环境与工具链

### 1.1 一次性装机（每台新机器人只做一次）

**【官方】顺序很重要**（先内核、再权限、再 udev）：

```bash
# ① 实时内核（Orange Pi 5 Plus 需装；RDK X5 烧预装镜像）
cd assets && sudo apt install ./*.deb && cd ..

# ② 授予实时优先级（用户名替换；RDK X5 默认是 sunrise）
sudo nano /etc/security/limits.conf
#   加两行：
#   <你的用户名>   -   rtprio   98
#   <你的用户名>   -   memlock  unlimited
sudo reboot
ulimit -r        # 期望输出 98 ← 验证
```

> ⚠️ **不配这个的后果**：推理节点**一启动就退出**（RT 线程创建失败）。
> 而启动脚本只会报「未检测到 inference_node」，看不出真正原因。

```bash
# ③ udev 规则（CAN 自动 UP + IMU 串口权限）
sudo cp assets/99-auto-up-devs-*.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
sudo reboot
ip a | grep can       # 验证：can0-3 应自动 UP
```

```bash
# ④ 电机 ID + IMU 配置（用达妙上位机）
#    - 每条总线 ID 唯一且 ≤ 15
#    - IMU 波特率 921600，输出频率 ≥ 200Hz
```

### 1.2 每次开发会话

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash          # ⚠️ 必须 source 后才能 import *_py 模块
```

> ⚠️ **`motors_py` / `robot_py` / `imu_py` 都是 colcon 构建产物。**
> 不 source `install/setup.bash` 就 `import motors_py` 会报 `ModuleNotFoundError`。

### 1.3 验证工具链通不通

```bash
# 上电前自检（不需要接电机）
./scripts/check_preflight.sh
```

**它会检查**：RT 权限、CAN 是否 UP、波特率、txqueuelen、IMU 串口权限、手柄，
并列出**必须由人确认**的物理准备项。

---

## 二、硬件清单与接线

### 2.1 通电前必须确认的（脚本查不了）

```
☐ 承重安全吊架已挂好（官方：任何悬空测试必须配备）
☐ 吊点位于重心正上方，保留 20-30cm 自由行程
☐ 地面铺好 5cm+ EVA 软垫，活动半径内无杂物
☐ 物理急停开关可达（人手能够到）
☐ 现场有第二个人，手放在急停上
☐ 机器人已可靠支撑
☐ 电池电压 ≥ 50V
☐ 脚踝连杆紧固已检查
```

### 2.2 接线要点

**【官方】USB 口选择**：

> 将 USB 转 CAN 插在主控的 **USB 3.0** 接口上。
> 如果使用 USB 扩展坞，也请使用 3.0 接口的扩展坞并插在 3.0 接口上；
> **IMU 和手柄插在 USB 2.0 接口即可。**

**CAN 总线映射**：

> ⚠️ **必须物理确认，不要靠猜。**
>
> 【实测】DM10 的映射（**2026-09-08 实测，覆盖旧的"can1=左腿"说法**）：
>
> | 总线 | 部位 |
> |---|---|
> | **can2** | **左腿**（ID 1-5）|
> | **can1** | **右腿**（ID 1-5）|
>
> **和直觉相反**（一般会以为 can1 在前）。

**用 udev 的 `KERNELS==` 绑定物理 USB 口**，或者插一根测一根。

### 2.3 线缆管理

> 【文献说】你是"长电源线从外部供电" → 建议至少做**悬吊式理线**，别让线在地上拖。

---

## 三、过关判据

| # | 判据 | 怎么测 |
|---|---|---|
| **1** | `ulimit -r` 返回 **98** | 命令 |
| **2** | `ip a` 能看到 **can0-3 已 UP** | 命令 |
| **3** | `./scripts/check_preflight.sh` **无 ❌ 项** | 脚本 |
| **4** | **上电前清单全部勾选** | 人工 |

---

## 四、常见问题

| 症状 | 原因 |
|---|---|
| 推理节点一启动就退出 | **`ulimit -r` 不是 98** → RT 线程创建失败 |
| `import motors_py` 报 ModuleNotFoundError | **没 source `install/setup.bash`** |
| CAN 接口没出现 | USB-CAN 没插好 / udev 规则没生效（重启试试）|
| 手柄无数据 | **不是 Xinput 模式** / 没配对 / 插在 3.0 口上 |

---

## 工具

```bash
./scripts/check_preflight.sh
```
