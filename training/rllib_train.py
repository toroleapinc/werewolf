"""
Run multi-agent training with Ray 2.7+ using a custom MultiAgentEnv.
No PettingZoo needed, so no agent_selection issues or parallel_to_aec conversions.
"""

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv
from .rllib_config import build_policies, policy_mapping_fn


def my_multiagent_werewolf_env_creator(config):
    """Factory function that returns our native MultiAgentEnv."""
    return WerewolfMultiAgentEnv(config)


def main():
    ray.init()

    # 1) Register your environment with a unique ID
    register_env("my_multiagent_werewolf", my_multiagent_werewolf_env_creator)

    # 2) Build multi-agent policies (seat_0..3 => wolf, else villager, etc.)
    policies = build_policies()

    # 3) Create PPOConfig referencing that env
    config = (
        PPOConfig()
        .environment(env="my_multiagent_werewolf")  # no PettingZoo
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
