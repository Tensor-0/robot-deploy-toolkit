#!/usr/bin/env bash
# 观测契约检查器的回归测试 —— 用【真实发生过的 bug】当用例。
#
# 为什么必须这样测
# ----------------
# 一个"什么都拒绝"的检查器和没有检查器一样没用，所以除了"该拒的要拒"，
# 还必须有"合法的要过"。三个用例分别覆盖：
#   ① 顺序错位     —— 2026-09-17 实际发生（两侧都 39 维，只比总数 ⇒ 静默通过）
#   ②a 可得性缺口 —— 65 维策略 vs 现网 39 维布局
#   ②b 最优努力布局 —— 65 维策略 vs 删掉无源项后的 56 维布局（必须仍被拒：
#                      "部署时删掉那几项" 是分布外输入，不是解法）
#   ③ 合法样例     —— 修好后的真实配置必须 PASS
#
# 用法: bash tests/test_check_contract.sh
set -uo pipefail
cd "$(dirname "$0")/.."
FIX=tests/fixtures
PASS=0; FAIL=0

check() {  # $1=用例名 $2=期望退出码 $3=必须出现的字符串 $4..=命令
    local name="$1" want="$2" needle="$3"; shift 3
    local out rc
    out=$("$@" 2>&1); rc=$?
    local ok=1
    [ "$rc" = "$want" ] || ok=0
    if [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; then ok=0; fi
    if [ "$ok" = 1 ]; then
        echo "  ✅ $name（退出码 $rc）"
        PASS=$((PASS+1))
    else
        echo "  ❌ $name —— 期望退出码 $want、输出含「$needle」，实得退出码 $rc"
        echo "$out" | tail -12 | sed 's/^/      /'
        FAIL=$((FAIL+1))
    fi
}

echo "=== 观测契约检查器 · 回归测试 ==="
echo

echo "① 顺序错位（改前的真实 bug）—— 应被拒，且点名第 2 段起"
check "顺序错位" 1 "❌ DENY 顺序/名字不符" \
    python3 scripts/check_contract.py --manifest $FIX/manifest_joystick_39.json \
        --deploy-config $FIX/layout_joystick_buggy.yaml

echo
echo "②a 65 维策略 vs 现网 39 维布局 —— 应被拒并点名三个无源观测"
check "可得性缺口" 1 "没有来源" \
    python3 scripts/check_contract.py --manifest $FIX/manifest_dm10_mt_65.json \
        --deploy-config $FIX/layout_joystick_fixed.yaml

echo
echo "②b 65 维策略 vs 56 维「最优努力」布局 —— 删掉无源项【也救不了】"
check "最优努力布局仍应被拒" 1 "没有来源" \
    python3 scripts/check_contract.py --manifest $FIX/manifest_dm10_mt_65.json \
        --deploy-config $FIX/layout_mt56_besteffort.yaml

echo
echo "③ 合法样例（修好后的真实配置）—— 必须 PASS"
check "合法样例通过" 0 "✅ PASS：观测顺序、维度、可得性三项一致" \
    python3 scripts/check_contract.py --manifest $FIX/manifest_joystick_39.json \
        --deploy-config $FIX/layout_joystick_fixed.yaml

echo
echo "=== 结果：$PASS 过 / $FAIL 失 ==="
[ "$FAIL" = 0 ]
