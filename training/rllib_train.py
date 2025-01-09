# File name: rllib_train.py
"""
Trains a multi-agent PPO model on WerewolfMultiAgentEnv under Ray 2.x,
with seat-based multi-agent policies, a custom TorchModelV2 (ActionMaskModel),
and a custom Exploration class (SafeStochasticSampling) that never returns None.

Usage:
    python -m training.rllib_train
"""

import os
import numpy as np
import torch
import torch.nn as nn

import ray
from ray import tune
from ray.rllib.utils import override
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.policy.sample_batch import SampleBatch

# 1) Your environment code
from werewolf_env import WerewolfMultiAgentEnv, NUM_SEATS


######################################
# 2) Seat-based multi-agent policies
######################################
def build_seat_policies():
    """
    Returns a dict of 12 seat-based policy IDs for seat_0..seat_11.
    """
    policies = {}
    for seat_id in range(NUM_SEATS):
        pid = f"seat_{seat_id}_policy"
        # (None, None, None, {}) => RLlib picks defaults for obs/action spaces
        policies[pid] = (None, None, None, {})
    return policies

def seat_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # agent_id is "seat_0" => return "seat_0_policy"
    return f"{agent_id}_policy"


######################################
# 3) Env registration
######################################
def werewolf_env_creator(env_config):
    return WerewolfMultiAgentEnv(env_config)


######################################
# 4) Custom Callbacks
######################################
class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["policy_rewards"] = {}

    def on_episode_step(self, worker, base_env, policies, episode, env_index, **kwargs):
        for agent_id, rewards_list in episode._agent_reward_history.items():
            if not rewards_list:
                continue
            step_reward = rewards_list[-1]
            policy_id = worker.policy_mapping_fn(agent_id, episode, worker)
            if policy_id not in episode.user_data["policy_rewards"]:
                episode.user_data["policy_rewards"][policy_id] = 0.0
            episode.user_data["policy_rewards"][policy_id] += step_reward

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        for policy_id, total_r in episode.user_data["policy_rewards"].items():
            metric_name = f"{policy_id}_reward"
            episode.custom_metrics[metric_name] = total_r


##############################################
# 5) Custom Exploration: SafeStochasticSampling
##############################################
class SafeStochasticSampling(Exploration):
    """
    Custom exploration that never returns None. 
    Importantly, we do NOT pass 'config' to super().__init__().
    """

    def __init__(self, action_space, **kwargs):
        # In your RLlib version, Exploration.__init__() typically is:
        #     def __init__(self, action_space, *, **kwargs)
        # So we do NOT pass config=...
        super().__init__(action_space=action_space, **kwargs)

    @override(Exploration)
    def get_exploration_action(self,
                               action_distribution,
                               timestep: int,
                               explore: bool = True,
                               is_tensorzied=False,
                               **kwargs):
        """
        Must return (action, logp).
        """
        if not explore:
            # Deterministic
            action = action_distribution.deterministic_sample()
            logp = action_distribution.logp(action)
            return action, logp
        else:
            # Stochastic
            action = action_distribution.sample()
            logp = action_distribution.logp(action)
            return action, logp


##################################################
# 6) Custom TorchModelV2 for "obs" + "action_mask"
##################################################
class ActionMaskModel(TorchModelV2, nn.Module):
    """
    Merges "obs" (shape=10) + "action_mask" (shape=num_actions) => masked logits.
    Fallback if entire mask=0 => we set action=0 => valid.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.n
        hidden_dim = 64
        self.obs_size = 10

        self.fc1 = nn.Linear(self.obs_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_layer = nn.Linear(hidden_dim, self.action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.logits_layer.weight)
        nn.init.zeros_(self.logits_layer.bias)

    def forward(self, input_dict, state, seq_lens):
        raw_obs = input_dict["obs"]
        if isinstance(raw_obs, dict):
            obs_tensor = raw_obs["obs"]
            mask_tensor = raw_obs.get("action_mask", None)
        else:
            obs_tensor = raw_obs
            mask_tensor = None

        # If no mask => all ones
        if mask_tensor is None:
            if torch.is_tensor(obs_tensor):
                bsz = obs_tensor.shape[0]
                mask_tensor = torch.ones(bsz, self.action_dim, device=obs_tensor.device)
            else:
                mask_tensor = torch.ones(1, self.action_dim)

        # Ensure obs_tensor is a torch Tensor
        if not torch.is_tensor(obs_tensor):
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32)

        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        x = torch.relu(self.fc1(obs_tensor))
        x = torch.relu(self.fc2(x))
        logits = self.logits_layer(x)

        # Ensure mask_tensor is shaped (B, action_dim)
        if not torch.is_tensor(mask_tensor):
            mask_tensor = torch.as_tensor(mask_tensor, dtype=torch.float32)
        if len(mask_tensor.shape) == 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        # If entire row is zero => action=0 => valid
        sum_mask = mask_tensor.sum(dim=1)
        zero_mask_rows = (sum_mask < 1e-8)
        if zero_mask_rows.any():
            mask_tensor[zero_mask_rows, 0] = 1.0

        inf_mask = (1.0 - mask_tensor) * -1e20
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        return torch.zeros(1, device=self.fc1.weight.device)


##################################################
# 7) The main training script
##################################################
def main():
    ray.init()

    tune.register_env("my_werewolf_env", werewolf_env_creator)
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    # Build seat-based policies
    policies = build_seat_policies()

    # Construct a PPO config
    config = (
        PPOConfig()
        .environment("my_werewolf_env", disable_env_checking=False)
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=seat_policy_mapping_fn,
        )
        .callbacks(MyCallbacks)
    )

    # Force older RLlib (ModelV2) code path
    config._enable_rl_module_api = False
    config._enable_learner_api = False

    # Provide custom exploration => SafeStochasticSampling
    config.exploration_config = {
        "type": SafeStochasticSampling,  # Our custom class
        # no "config": ... because your version’s Exploration base doesn't accept it
    }

    # Provide custom model
    config.training(
        model={
            "custom_model": "action_mask_model",
        }
    )

    algo = config.build()

    num_iterations = 10
    save_freq = 2
    checkpoint_root = "rllib_checkpoints"

    for i in range(num_iterations):
        result = algo.train()
        print(f"Iteration {i} => episode_reward_mean: {result['episode_reward_mean']}")

        if "custom_metrics" in result:
            for k, v in result["custom_metrics"].items():
                if k.endswith("_mean"):
                    print(f"  {k}: {v}")

        if i % save_freq == 0:
            ckpt = algo.save(checkpoint_dir=checkpoint_root)
            print(f"Checkpoint saved at: {ckpt}")

    final_ckpt = algo.save(checkpoint_dir=checkpoint_root)
    print(f"Final checkpoint saved at: {final_ckpt}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
