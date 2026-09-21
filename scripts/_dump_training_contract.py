#!/usr/bin/env python3
"""从训练 run_config.json 里抽出【部署必须一致】的全部常数。

结构：{ "run":…, "config": {完整 hydra 配置}, "contract_snapshot": {扁平契约} }
判据来源 = config（训练当时实际用的值），contract_snapshot 作为交叉印证。
"""
import json
from pathlib import Path

RUN = Path("/home/zhan/UniLab/logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-09_00-52-12_mujoco/run_config.json")
cfg = json.loads(RUN.read_text())
conf = cfg["config"]
snap = cfg["contract_snapshot"]

print("=" * 78)
print("训练 run:", cfg["run"])
print("=" * 78)

print("\n########## contract_snapshot['env.observations'] ##########")
print(json.dumps(snap["env.observations"], indent=2, ensure_ascii=False))

print("\n########## contract_snapshot['env.actions'] ##########")
print(json.dumps(snap["env.actions"], indent=2, ensure_ascii=False))

print("\n########## contract_snapshot 其它 ##########")
for k in ("algo.obs_groups", "env.policy_observation_group",
          "env.critic_observation_group", "algo.policy.actor_hidden_dims",
          "algo.policy.critic_hidden_dims", "algo.empirical_normalization", "env.ctrl_dt"):
    print(f"  {k} = {json.dumps(snap[k], ensure_ascii=False)}")

env = conf.get("env", {})
print("\n########## config.env ##########")
print("  键:", list(env.keys()))
for k in ("sim_dt", "ctrl_dt", "max_episode_seconds", "policy_observation_group"):
    print(f"  {k} = {env.get(k)}")

print("\n########## config.env.scene ##########")
print(json.dumps(env.get("scene"), indent=2, ensure_ascii=False)[:2000])

print("\n########## config.env.actions ##########")
print(json.dumps(env.get("actions"), indent=2, ensure_ascii=False))

print("\n########## config.env.commands ##########")
print(json.dumps(env.get("commands"), indent=2, ensure_ascii=False))

print("\n########## config.env.events ##########")
print("  事件名:", list(env.get("events", {}).keys()))

algo = conf.get("algo", {})

def dig(d, *ks, default=None):
    cur = d
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

print("\n########## algo ##########")
for k in ("num_envs", "max_iterations", "empirical_normalization"):
    print(f"  {k} = {dig(algo, k)}")
print("  algo 键:", list(algo.keys()))

seed = dig(conf, "algo", "seed", default=dig(conf, "seed"))
print(f"  seed = {seed}")
