"""
Run multi-agent training via Ray RLlib PPO using the new config style in Ray 2.7+.
"""

import ray
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from werewolf_env import WerewolfParallelEnv
from .rllib_config import build_policies, policy_mapping_fn

def env_creator(_):
    """Factory to create our PettingZoo environment, then wrap it for RLlib."""
    env = WerewolfParallelEnv()
    return PettingZooEnv(env)

def main():
    # Initialize Ray
    ray.init()

    # Build policy dict
    policies = build_policies()

    # Construct new PPOConfig (Ray 2.7+ style)
    config = (
        PPOConfig()
        # 1) environment
        .environment(env=env_creator)
        # 2) multi-agent
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        # 3) rollout settings
        .rollouts(num_rollout_workers=1)
        # 4) framework
        .framework("torch")
        # 5) resources
        .resources(num_gpus=0)
    )

    # Build algo
    algo = config.build()

    # Train for a few iterations
    for i in range(3):
        result = algo.train()
        print(f"Iteration {i} - episode_reward_mean: {result['episode_reward_mean']}")

    # Cleanup
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
