# File name: rllib_restore.py
"""
Restore a multi-agent PPO model from `rllib_checkpoints` on WerewolfMultiAgentEnv.
- We disable exploration (NoOpExploration) after restore
- We force-cast observation arrays to float32 to fix "mat1 and mat2 must have the same dtype" errors.

Usage:
  python rllib_restore.py
"""

import os
import numpy as np
import torch
import torch.nn as nn

import ray
from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog

# Your environment
from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv, NUM_SEATS


# ---------------------------------------------------------------------
# 1) Build seat-based policy definitions
# ---------------------------------------------------------------------
def build_seat_policies():
    """
    Returns seat_0_policy..seat_11_policy, each: (None, None, None, {})
    """
    policies = {}
    for seat_id in range(NUM_SEATS):
        pid = f"seat_{seat_id}_policy"
        policies[pid] = (None, None, None, {})
    return policies

def seat_policy_mapping_fn(agent_id, *args, **kwargs):
    # "seat_0" => "seat_0_policy"
    return f"{agent_id}_policy"


# ---------------------------------------------------------------------
# 2) A no-op exploration that calls deterministic_sample
# ---------------------------------------------------------------------
class NoOpExploration(Exploration):
    def __init__(self,
                 action_space,
                 *,
                 framework,
                 policy_config,
                 model,
                 num_workers,
                 worker_index,
                 **kwargs):
        """
        We must call super() with all of these keyword arguments. 
        """
        super().__init__(
            action_space=action_space,
            framework=framework,
            policy_config=policy_config,
            model=model,
            num_workers=num_workers,
            worker_index=worker_index,
            **kwargs
        )

    def before_compute_actions(self, explore: bool = True, timestep: int = None, **kwargs):
        pass

    def get_exploration_action(self,
                               action_distribution,
                               timestep: int = None,
                               explore: bool = True,
                               is_tensorized=False,
                               **kwargs):
        """
        Return (action, logp). We'll do a purely deterministic pick from the distribution.
        """
        action = action_distribution.deterministic_sample()
        logp = action_distribution.logp(action)
        return action, logp

    def get_exploration_action_dist(self, action_distribution, timestep=None, explore=True):
        """Return the same distribution plus None for dist inputs."""
        return action_distribution, None


# ---------------------------------------------------------------------
# 3) The same ActionMaskModel from training, with forced float32
# ---------------------------------------------------------------------
class ActionMaskModel(TorchModelV2, nn.Module):
    """
    Applies a mask to the output logits to respect valid/invalid actions.
    We force-cast obs + mask => float32, so PyTorch doesn't complain about dtype.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.n
        self.obs_size = 10
        hidden_dim = 64

        self.fc1 = nn.Linear(self.obs_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_layer = nn.Linear(hidden_dim, self.action_dim)

        # Weight init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.logits_layer.weight)
        nn.init.zeros_(self.logits_layer.bias)

    def forward(self, input_dict, state, seq_lens):
        raw_input = input_dict["obs"]  # typically a dict => {"obs", "action_mask"}

        if isinstance(raw_input, dict):
            obs_array = raw_input["obs"]
            mask_array = raw_input.get("action_mask", None)
        else:
            obs_array = raw_input
            mask_array = None

        # -------------------------------------------------------
        # Force-cast obs to float32
        # -------------------------------------------------------
        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array)
        obs_array = obs_array.astype(np.float32, copy=False)
        obs_tensor = torch.from_numpy(obs_array)

        # Also handle mask => float32
        if mask_array is None:
            # no mask => all ones
            mask_array = np.ones((obs_tensor.shape[0], self.action_dim), dtype=np.float32) \
                         if len(obs_tensor.shape) > 1 else \
                         np.ones((1, self.action_dim), dtype=np.float32)
        else:
            if not isinstance(mask_array, np.ndarray):
                mask_array = np.array(mask_array)
            mask_array = mask_array.astype(np.float32, copy=False)
        mask_tensor = torch.from_numpy(mask_array)

        # Expand dims if needed
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if len(mask_tensor.shape) == 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        # optional debug
        # print("DEBUG dtypes: obs=", obs_tensor.dtype, "mask=", mask_tensor.dtype,
        #       "fc1.weight=", self.fc1.weight.dtype)

        x = torch.relu(self.fc1(obs_tensor))
        x = torch.relu(self.fc2(x))
        logits = self.logits_layer(x)

        # If entire row of mask=0 => seat=0 => valid
        sum_mask = mask_tensor.sum(dim=1)
        zero_mask_rows = (sum_mask < 1e-8)
        if zero_mask_rows.any():
            mask_tensor[zero_mask_rows, 0] = 1.0

        inf_mask = (1.0 - mask_tensor) * -1e20
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        """If you had a separate value net, you'd do that here; we'll just return zero."""
        return torch.zeros(1, device=self.fc1.weight.device)


# ---------------------------------------------------------------------
# 4) Minimal callbacks (if needed)
# ---------------------------------------------------------------------
class MyCallbacks(DefaultCallbacks):
    pass


# ---------------------------------------------------------------------
# 5) Env registration
# ---------------------------------------------------------------------
def werewolf_env_creator(env_config):
    return WerewolfMultiAgentEnv(env_config)


# ---------------------------------------------------------------------
# 6) The main restore + demo function
# ---------------------------------------------------------------------
def main():
    ray.init()

    # Register env + model
    tune.register_env("my_werewolf_env", werewolf_env_creator)
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    # Build seat-based policies
    seat_policies = build_seat_policies()

    # Construct the EXACT same PPOConfig from training
    config = (
        PPOConfig()
        .environment("my_werewolf_env", disable_env_checking=False)
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
        .multi_agent(
            policies=seat_policies,
            policy_mapping_fn=seat_policy_mapping_fn,
        )
        .callbacks(MyCallbacks)
    )
    # Must set these flags to disable RLModule/learner API
    config._enable_rl_module_api = False
    config._enable_learner_api   = False

    # Same exploration config as training
    config.exploration_config = {
        "type": "StochasticSampling",
    }

    # Provide same custom model
    config.training(
        model={
            "custom_model": "action_mask_model",
        }
    )

    # Build PPO algo
    algo = config.build()

    # Attempt restore
    checkpoint_dir = "rllib_checkpoints"
    print(f"\n[INFO] Attempting to restore from: {checkpoint_dir}")
    algo.restore(checkpoint_dir)
    print("[INFO] Restore successful!")

    # Overwrite exploration with a NoOpExploration
    for pid, pol in algo.workers.local_worker().policy_map.items():
        pol.exploration = NoOpExploration(
            action_space=pol.action_space,
            framework=pol.config.get("framework", "torch"),
            policy_config=pol.config,
            model=pol.model,
            num_workers=pol.config.get("num_workers", 0),
            worker_index=0
        )

    # Create env, step once
    env = WerewolfMultiAgentEnv()
    obs_dict, _ = env.reset()

    done_flags = {agent_id: False for agent_id in env.agents}
    step_count = 0

    print("\n=== Starting demonstration with the restored model ===")

    while not all(done_flags.values()):
        step_count += 1
        action_dict = {}

        for agent_id in env.agents:
            if done_flags[agent_id]:
                # seat is done => action=0
                action_dict[agent_id] = 0
                continue

            pol_id = seat_policy_mapping_fn(agent_id)
            pol = algo.get_policy(pol_id)
            seat_obs = obs_dict[agent_id]

            print(f"\n[DEBUG] step={step_count}, agent={agent_id}, obs={seat_obs}")
            # compute_single_action(..., explore=False) => deterministic
            action, _, extra = pol.compute_single_action(
                seat_obs,
                explore=False
            )
            print(f"[DEBUG] step={step_count}, agent={agent_id}, action={action}")
            action_dict[agent_id] = action

        # Step env
        next_obs, rewards, terminated, truncated, infos = env.step(action_dict)
        print(f"\n--- STEP {step_count} => Phase={env.phase}, DayCount={env.day_count}")

        # Print non-zero rewards
        for agent_id, rew in rewards.items():
            if abs(rew) > 1e-8:
                print(f"  Agent {agent_id} => reward={rew}")

        # Update done flags
        for agent_id in env.agents:
            if terminated[agent_id] or truncated[agent_id]:
                done_flags[agent_id] = True

        obs_dict = next_obs

        if all(done_flags.values()):
            print("\n=== END OF GAME ===\nFinal rewards:")
            for agent_id in env.agents:
                print(f"  {agent_id} => {rewards[agent_id]}")
            break

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
