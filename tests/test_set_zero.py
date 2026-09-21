#!/usr/bin/env python3
"""set_zero.py 的回归测试 —— 用 2026-09-17 真实发生过的 bug 当用例。

为什么必须这样测
----------------
`set_zero.py` 把【当前姿态】写进电机硬件零点，**不可逆**。所以不能为了
测试去真跑一遍 —— 真跑一次就把用户 09-17 实测标定出来的零点覆盖掉了。

而这条 bug 偏偏就藏在这条不可逆路径上：
    `motor.set_motor_zero()` 的返回值被丢弃，只要按了 Enter 就一律记成"已标零"。
驱动内部其实是校验过的（`dm_motor_driver.cpp` 的 `set_motor_zero()`：刷状态 →
读回位置 → `|pos| > judgment_accuracy_threshold`(=0.01) 就 `return false`），
只是 Python 侧没接。
⇒ 症状是"标完零发现偏差还在"，而且**不知道是哪台**。10 个电机的腿上，这等于从头再猜一遍。

两层，共 7 个用例：
  ① `calibrate_motor()` 的分支逻辑 —— 假电机 + 脚本化按键，断言返回的三种状态
  ② `main()` 的归类/汇总/退出码/落盘 —— 断言 failed 与 skipped 分得开、退出码可脚本判读

⚠️ 全程没有 CAN 流量，`set_motor_zero()` 是假的。碰不到真硬件。

用法:
    python3 tests/test_set_zero.py [set_zero.py 的路径]
    默认 ~/roboparty_deploy/scripts/set_zero.py，可用 $SET_ZERO_PATH 覆盖。

退出码: 0=全过  1=有用例失败  2=找不到被测文件（明确区别于"过"）
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import types

DEFAULT_TARGET = "~/roboparty_deploy/scripts/set_zero.py"

# 本机（写这个测试的机器）没有编译好的 motors_py；被测脚本只用到
# MotorControlMode.MIT 一个常量，注入假的就够，不必把测试绑到 RDK 上。
try:
    import motors_py  # noqa: F401
except ImportError:
    _fake = types.ModuleType("motors_py")

    class _Mode:
        MIT = 1

    _fake.MotorControlMode = _Mode
    sys.modules["motors_py"] = _fake


class FakeMotor:
    """只实现 calibrate_motor() 会用到的那几个方法。

    `set_motor_zero()` 的返回值就是驱动内部读回校验的结果 —— 用例靠它区分
    "驱动成功" 与 "驱动失败"。
    """

    def __init__(self, zero_ok: bool, before: float, after: float):
        self.zero_ok = zero_ok
        self.before = before
        self.after = after
        self.zero_called = False
        self.deinit_called = False

    def init_motor(self):
        pass

    def set_motor_control_mode(self, _mode):
        pass

    def motor_mit_cmd(self, *_a):
        pass

    def get_motor_pos(self):
        return self.after if self.zero_called else self.before

    def get_error_id(self):
        return 0

    def set_motor_zero(self):
        self.zero_called = True
        return self.zero_ok

    def deinit_motor(self):
        self.deinit_called = True


def load_module(path):
    spec = importlib.util.spec_from_file_location("setzero_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────── ① calibrate_motor 的分支 ─────────────────────────
#        名称,              驱动返回, 标前,  标后,  期望,      说明
CASES = [
    ("驱动成功 + 读数归零", True, 1.20, 0.001, "ok", "正常路径"),
    ("驱动成功 + 读数偏大", True, 1.20, 0.300, "failed", "驱动说 ok 但读回漂了 → 二次确认该抓"),
    ("驱动失败", False, 1.20, 0.150, "failed", "★ 原 bug：这条以前返回 True（=已标零）"),
    ("按空格跳过", None, 1.20, 1.20, "skipped", "主动跳过，不该被算成失败"),
]


def layer1(mod, tally):
    print("① calibrate_motor() 的分支（假电机 + 脚本化按键）")
    for name, zero_ok, before, after, expect, note in CASES:
        motor = FakeMotor(zero_ok, before, after)
        keys = [" " if zero_ok is None else "\r"]
        mod.read_key_nonblocking = lambda _t=0.05: keys.pop(0) if keys else None

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = mod.calibrate_motor(
                {"motor": motor, "motor_id": 7, "interface": "can0", "index": 3}
            )
        out = buf.getvalue()

        ok = got == expect
        # 电机绝不能留在使能态 —— 这条和上面的分支同等重要
        if not motor.deinit_called:
            ok = False
            note += "  ⚠️ deinit_motor 没被调用"
        tally(ok, f"{name:<20} 期望 {expect:<8} 实得 {got!r:<10} {note}")
        for ln in out.strip().splitlines():
            if any(s in ln for s in ("✅", "❌", "⚠️")):
                print(f"        ↳ {ln.strip()}")
    print()


# ───────────────────────── ② main() 的归类与退出码 ─────────────────────────
#        名称,          [(motor_id, 驱动返回, 标前, 标后, 按键)],              期望退出码
SCENARIOS = [
    ("全成功", [(1, True, 1.2, 0.001, "\r"), (2, True, 0.9, 0.002, "\r")], 0),
    ("一台失败", [(1, True, 1.2, 0.001, "\r"), (2, False, 0.9, 0.400, "\r")], 1),
    ("失败 + 跳过", [(1, False, 1.2, 0.500, "\r"), (2, True, 0.9, 0.001, " ")], 1),
    ("全跳过", [(1, True, 1.2, 0.001, " ")], 0),
]


def layer2(target, tally, created_dir):
    print("② main() 的归类 / 汇总 / 退出码 / 落盘")
    for label, plan, want_rc in SCENARIOS:
        mod = load_module(target)
        motors, keys = [], []
        for mid, zero_ok, before, after, key in plan:
            motors.append({"motor": FakeMotor(zero_ok, before, after),
                           "motor_id": mid, "interface": "can0", "index": mid})
            keys.append(key)

        mod.load_config = lambda _p: {
            "motor_id": [m["motor_id"] for m in motors],
            "motor_type": ["DM"] * len(motors),
            "motor_interface_type": ["can"] * len(motors),
            "motor_interface": ["can0"] * len(motors),
            "motor_model": [2] * len(motors),
        }
        mod.create_motors = lambda _c: motors
        mod.read_key_nonblocking = lambda _t=0.05: keys.pop(0) if keys else None
        mod.time.sleep = lambda _s: None  # 免掉 0.3+0.2 秒/台
        saved_argv = sys.argv
        sys.argv = ["set_zero.py", "--config", "/tmp/does-not-exist.yaml", "-y"]
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = mod.main()
        except Exception as e:
            # 指向的如果不是当前那版（比如忘了加 --yes），别抛堆栈 —— 记一条失败就好
            tally(False, f"{label:<12} main() 抛异常: {type(e).__name__}: {e}")
            continue
        finally:
            sys.argv = saved_argv
        out = buf.getvalue()

        # 落盘内容也要对：三态必须逐台写清楚
        record, rec_path = None, None
        for ln in out.splitlines():
            if "记录已落盘" in ln:
                rec_path = ln.split("记录已落盘:")[-1].strip()
        if rec_path and os.path.exists(rec_path):
            created_dir.add(os.path.dirname(rec_path))
            with open(rec_path, encoding="utf-8") as f:
                record = json.load(f)
            os.remove(rec_path)

        want_results = []
        for mid, zero_ok, _b, _a, key in plan:
            want_results.append((mid, "skipped" if key == " " else ("ok" if zero_ok else "failed")))
        got_results = [(r["motor_id"], r["result"]) for r in record["results"]] if record else None

        ok = rc == want_rc and got_results == want_results and record is not None
        summary = " | ".join(
            ln.strip() for ln in out.splitlines()
            if ln.strip().startswith(("✅ 已标零", "❌ 失败", "⏭"))
        )
        tally(ok, f"{label:<12} 退出码 {rc}(期望 {want_rc})  落盘 {got_results}")
        if summary:
            print(f"        ↳ {summary}")
    print()


def main():
    target = os.environ.get("SET_ZERO_PATH") or (sys.argv[1] if len(sys.argv) > 1 else None)
    target = os.path.expanduser(target or DEFAULT_TARGET)
    if not os.path.isfile(target):
        print(f"⏭ SKIP：找不到被测文件 {target}")
        print("   用法: python3 tests/test_set_zero.py [路径]   或设 $SET_ZERO_PATH")
        print("   需要一个 roboparty_deploy 检出（本测试验的是它的 scripts/set_zero.py）")
        return 2

    print("=== set_zero.py · 回归测试 ===")
    print(f"被测文件: {target}")
    print()

    state = {"pass": 0, "fail": 0}

    def tally(ok, line):
        if ok:
            state["pass"] += 1
            print(f"  ✅ {line}")
        else:
            state["fail"] += 1
            print(f"  ❌ {line}")

    mod = load_module(target)
    layer1(mod, tally)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(target))), "results")
    existed = os.path.isdir(results_dir)
    created_dir = set()
    layer2(target, tally, created_dir)

    # 测试产物不留痕：删除记录文件；如果 results/ 是本测试创建的，一并删掉
    for d in created_dir:
        if not existed and os.path.isdir(d) and not os.listdir(d):
            shutil.rmtree(d, ignore_errors=True)

    print(f"=== 结果：{state['pass']} 过 / {state['fail']} 失 ===")
    return 0 if state["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
