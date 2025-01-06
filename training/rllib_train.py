# training/rllib_train.py
"""
Runs PPO training on the WerewolfMultiAgentEnv with:
- seat-based or multi-agent approach
- a custom callback (MyPolicyRewardCallback) to log each policy's total reward
- printing each seat (policy) reward after each training iteration

Usage:
    python -m training.rllib_train
"""

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv
from .rllib_config import build_policies, policy_mapping_fn


class MyPolicyRewardCallback(DefaultCallbacks):
    """
    Custom callback to record each policy's total reward per episode.
    Then RLlib logs the average of those totals under custom_metrics.
    """

    def on_episode_start(self, worker, base_env, policies, episode, env_index, **kwargs):
        # Track a dictionary of { policy_id: cumulative_reward } in user_data
        episode.user_data["policy_rewards"] = {}

    def on_episode_step(self, worker, base_env, policies, episode, env_index, **kwargs):
        # For each agent, get the last incremental reward in _agent_reward_history
        for agent_id, rewards_list in episode._agent_reward_history.items():
            if not rewards_list:
                continue  # no rewards recorded yet
            step_reward = rewards_list[-1]  # the latest reward

            # Figure out this agent's policy
            policy_id = worker.policy_mapping_fn(agent_id, episode, worker)

            if policy_id not in episode.user_data["policy_rewards"]:
                episode.user_data["policy_rewards"][policy_id] = 0.0

            episode.user_data["policy_rewards"][policy_id] += step_reward

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        # Now store each policy's final total in custom_metrics
        for policy_id, total_r in episode.user_data["policy_rewards"].items():
            metric_name = f"{policy_id}_reward"
            episode.custom_metrics[metric_name] = total_r


def werewolf_env_creator(config):
    """Factory returning our multi-agent Werewolf environment."""
    return WerewolfMultiAgentEnv(config)


def main():
    ray.init()

    # 1) Register environment
    register_env("my_werewolf_env", werewolf_env_creator)

    # 2) Build multi-agent seat-based or role-based policies
    policies = build_policies()

    # 3) Construct PPO config with multi-agent
    config = (
        PPOConfig()
        .environment(
            env="my_werewolf_env",
            disable_env_checking=False
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .resources(num_gpus=0)
        .callbacks(MyPolicyRewardCallback)  # log per-policy rewards
    )

    # 4) Build the PPO algorithm
    algo = config.build()

    # 5) Train and print results each iteration
    for i in range(5):
        result = algo.train()

        # Default aggregate
        print(f"Iteration {i} - episode_reward_mean: {result['episode_reward_mean']}")

        # Also print each policy's reward
        # result["custom_metrics"] is a dict, e.g. { "seat_0_policy_reward_mean": X, ... }
        # We'll iterate over keys ending in "_mean"
        if "custom_metrics" in result:
            for metric_name, value in result["custom_metrics"].items():
                # RLlib automatically appends "_mean" to the metric's average over episodes
                if metric_name.endswith("_mean"):
                    print(f"  {metric_name}: {value}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
