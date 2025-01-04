# training/rllib_train.py
"""
Runs PPO training on the WerewolfMultiAgentEnv with day/night logic, special roles, etc.
"""

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv
from .rllib_config import build_policies, policy_mapping_fn

def werewolf_env_creator(config):
    """Factory function returning our multi-agent env."""
    return WerewolfMultiAgentEnv(config)

def main():
    ray.init()

    # 1) Register the environment with RLlib
    register_env("my_werewolf_env", werewolf_env_creator)

    # 2) Build multi-agent policies
    policies = build_policies()

    # 3) Construct PPO config
    config = (
        PPOConfig()
        .environment(
            env="my_werewolf_env",
            disable_env_checking=False  # or True if you want to skip checks
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .resources(num_gpus=0)
    )

    algo = config.build()

    for i in range(3):
        result = algo.train()
        print(f"Iteration {i} - episode_reward_mean: {result['episode_reward_mean']}")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
