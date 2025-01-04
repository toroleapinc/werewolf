# werewolf_multiagent.py

import random
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box

from .roles import ROLE_DISTRIBUTION

NUM_SEATS = 12

def assign_roles():
    roles = ROLE_DISTRIBUTION[:]
    random.shuffle(roles)
    return roles

class WerewolfMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        # 1) Create your list of agent IDs
        self.num_seats = NUM_SEATS
        self.agents = [f"seat_{i}" for i in range(self.num_seats)]

        # 2) RLlib wants a private set _agent_ids that matches your agent IDs
        self._agent_ids = set(self.agents)

        # Action + obs space
        self.action_space = Discrete(15)
        self.observation_space = Box(low=0, high=100, shape=(5,), dtype=np.int32)

        # Internal state
        self.role_assignment = []
        self.alive = []
        self.episode_terminated = False
        self.episode_truncated = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.role_assignment = assign_roles()
        self.alive = [True] * self.num_seats
        self.episode_terminated = False
        self.episode_truncated = False

        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = self._get_obs(i)

        info_dict = {}
        return obs_dict, info_dict

    def step(self, action_dict):
        if self.episode_terminated or self.episode_truncated:
            obs = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
            rewards = {agent: 0.0 for agent in self.agents}
            terminateds = {agent: True for agent in self.agents}
            truncateds = {agent: False for agent in self.agents}
            terminateds["__all__"] = True
            truncateds["__all__"] = self.episode_truncated
            info = {agent: {} for agent in self.agents}
            return obs, rewards, terminateds, truncateds, info

        # Process actions (placeholder)
        werewolf_alive = 0
        for i in range(self.num_seats):
            if self.alive[i] and self.role_assignment[i] == "werewolf":
                werewolf_alive += 1

        if werewolf_alive == 4:
            self.episode_terminated = True

        rewards = {agent: 0.0 for agent in self.agents}
        if self.episode_terminated:
            for i, agent in enumerate(self.agents):
                if self.role_assignment[i] == "werewolf":
                    rewards[agent] = 1.0
                else:
                    rewards[agent] = -1.0

        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = self._get_obs(i)

        terminateds = {}
        truncateds = {}
        if self.episode_terminated:
            for agent in self.agents:
                terminateds[agent] = True
                truncateds[agent] = False
            terminateds["__all__"] = True
            truncateds["__all__"] = False
        else:
            for agent in self.agents:
                terminateds[agent] = False
                truncateds[agent] = False
            terminateds["__all__"] = False
            truncateds["__all__"] = False

        info = {agent: {} for agent in self.agents}
        return obs, rewards, terminateds, truncateds, info

    def _get_obs(self, seat_index):
        seat_id = seat_index
        is_wolf = 1 if (self.role_assignment[seat_index] == "werewolf") else 0
        is_alive = 1 if self.alive[seat_index] else 0
        return np.array([seat_id, is_wolf, is_alive, 0, 0], dtype=np.int32)
