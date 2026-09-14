#!/usr/bin/env bash
# check_preflight.sh — 上电前自检
#
# 在给机器人上电之前跑这个。全部只读，不碰硬件。
# 设计原则：**能在软件层发现的问题，绝不留给真机去发现。**
#
# 用法:
#   ./check_preflight.sh                    # 自动探测
#   ./check_preflight.sh --ifaces can1,can2 # 指定总线
#   ./check_preflight.sh --imu /dev/ttyACM0 # 指定 IMU 串口

set -uo pipefail

IFACES=""
IMU_DEV=""
EXPECT_RTPRIO=98

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ifaces) IFACES="$2"; shift 2 ;;
        --imu)    IMU_DEV="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

PASS=0; FAIL=0; WARN=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
warn() { echo "  ⚠️  $1"; WARN=$((WARN+1)); }

echo "════════════════════════════════════════════════════"
echo " 上电前自检（只读，不碰硬件）"
echo "════════════════════════════════════════════════════"

# ── 1. 实时权限 ──────────────────────────────────────
echo
echo "[1] 实时优先级权限（缺失会导致节点一启动就退出）"
RTP=$(ulimit -r 2>/dev/null || echo 0)
if [[ "$RTP" == "unlimited" ]] || [[ "$RTP" -ge "$EXPECT_RTPRIO" ]] 2>/dev/null; then
    ok "ulimit -r = $RTP"
else
    bad "ulimit -r = $RTP（期望 ≥ $EXPECT_RTPRIO）"
    echo "     修复：/etc/security/limits.conf 加两行后重启"
    echo "       $(whoami)   -   rtprio   98"
    echo "       $(whoami)   -   memlock  unlimited"
fi
MLOCK=$(ulimit -l 2>/dev/null || echo 0)
if [[ "$MLOCK" == "unlimited" ]] || [[ "$MLOCK" -ge 8192 ]] 2>/dev/null; then
    ok "ulimit -l = $MLOCK"
else
    warn "ulimit -l = $MLOCK（mlockall 可能失败 → 内存换页 → 毫秒级卡顿）"
fi

# ── 2. CAN 总线 ──────────────────────────────────────
echo
echo "[2] CAN 总线"
if [[ -z "$IFACES" ]]; then
    IFACES=$(ip -o link show 2>/dev/null | grep -oP 'can\d+' | sort -u | paste -sd,)
fi
if [[ -z "$IFACES" ]]; then
    bad "未发现任何 canX 接口（USB-CAN 没插？驱动没加载？）"
else
    for i in ${IFACES//,/ }; do
        if ip link show "$i" 2>/dev/null | grep -q "state UP"; then
            # 关键：检查 bitrate 配置
            INFO=$(ip -details link show "$i" 2>/dev/null | grep -oP 'bitrate \d+' | head -1)
            QLEN=$(ip link show "$i" 2>/dev/null | grep -oP 'qlen \d+' | head -1)
            ok "$i UP  ($INFO, $QLEN)"
            # 波特率核实
            if echo "$INFO" | grep -q "bitrate 1000000"; then :; else
                warn "$i 波特率不是 1000000，确认是否与电机一致"
            fi
            if echo "$QLEN" | grep -q "qlen 1000"; then :; else
                warn "$i txqueuelen 不是 1000（默认 10 容易丢帧）"
            fi
        else
            bad "$i 未 UP"
            echo "     修复：sudo ip link set $i up type can bitrate 1000000"
            echo "           sudo ip link set $i txqueuelen 1000"
        fi
    done
fi

# ── 3. IMU 串口 ──────────────────────────────────────
echo
echo "[3] IMU 串口"
if [[ -z "$IMU_DEV" ]]; then
    for d in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyUSB0 /dev/ttyUSB1; do
        [[ -e "$d" ]] && IMU_DEV="$d" && break
    done
fi
if [[ -z "$IMU_DEV" ]]; then
    warn "未发现常见串口设备（IMU 没插？）"
else
    if [[ -r "$IMU_DEV" ]] && [[ -w "$IMU_DEV" ]]; then
        ok "$IMU_DEV 可读写"
    else
        bad "$IMU_DEV 权限不足"
        echo "     修复：sudo chmod 666 $IMU_DEV  或把用户加入 dialout 组"
    fi
fi

# ── 4. 手柄 ──────────────────────────────────────────
echo
echo "[4] 手柄（官方要求：必须 Xinput 模式）"
if ls /dev/input/js* >/dev/null 2>&1; then
    ok "发现 $(ls /dev/input/js* 2>/dev/null | tr '\n' ' ')"
    warn "请手动确认手柄处于 Xinput 模式（按 home 键配对）"
else
    warn "未发现 /dev/input/js*（手柄没插或没配对）"
fi

# ── 5. 物理准备（人工核验，脚本只能提醒）────────────
echo
echo "[5] 物理准备 —— ⚠️ 以下必须由人确认，脚本无法代劳"
cat <<'EOF'
  ☐ 承重安全吊架已挂好（官方：任何悬空标零或步态推理测试必须配备）
  ☐ 吊点位于重心正上方，保留 20-30cm 自由行程
  ☐ 地面铺好 5cm+ EVA 软垫，活动半径内无杂物
  ☐ 物理急停开关可达（人手能够到）
  ☐ 现场有第二个人，手放在急停上
  ☐ 机器人已可靠支撑
  ☐ 电池电压 ≥ 50V（官方：低于 50V 必须充电，否则运行中途可能掉电跌倒）
  ☐ 脚踝连杆紧固已检查（官方：松动会导致脚踝发力异常、站不住并损坏结构件）
EOF

# ── 汇总 ─────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════"
printf " 通过 %d · 失败 %d · 警告 %d\n" "$PASS" "$FAIL" "$WARN"
if [[ "$FAIL" -gt 0 ]]; then
    echo " ❌ 有失败项 —— 不要上电。先修掉上面标 ❌ 的。"
    exit 1
elif [[ "$WARN" -gt 0 ]]; then
    echo " ⚠️  有警告项 —— 确认无影响后再上电。"
    exit 0
else
    echo " ✅ 软件层自检全通过。别忘了上面 [5] 的人工核验项。"
    exit 0
fi
