"""
A minimal PettingZoo ParallelEnv for a "Werewolf" game.
This version:
- Assigns roles on reset.
- Ends the episode if all 4 werewolves remain alive (placeholder "Wolf win" condition).
- Rewards +1 if your team "won," -1 if not.
"""

import random
import numpy as np

from gymnasium.spaces import Discrete, Box
from pettingzoo.utils import ParallelEnv

from .roles import ROLE_DISTRIBUTION

NUM_SEATS = 12

def assign_roles():
    """
    Shuffle the predefined distribution of roles:
    [4 werewolves, 4 villagers, 1 seer, 1 witch, 1 hunter, 1 idiot].
    Returns a list of length 12.
    """
    roles = ROLE_DISTRIBUTION[:]
    random.shuffle(roles)
    return roles

class WerewolfParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "werewolf_env_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = [f"seat_{i}" for i in range(NUM_SEATS)]
        self.agents = []

        # Action space: Discrete(15) for demonstration
        self.action_spaces = {
            agent: Discrete(15) for agent in self.possible_agents
        }

        # Observation space: minimal vector of 5 ints
        self.observation_spaces = {
            agent: Box(low=0, high=100, shape=(5,), dtype=np.int32)
            for agent in self.possible_agents
        }

        # Internal state
        self.role_assignment = []
        self.alive = [True]*NUM_SEATS
        self.game_done = False

    def reset(self, seed=None, return_info=False, options=None):
        """
        Shuffle roles, set everyone alive, no game_done, return initial observations.
        """
        super().reset(seed=seed)
        self.agents = self.possible_agents[:]
        self.role_assignment = assign_roles()
        self.alive = [True]*NUM_SEATS
        self.game_done = False

        obs = self._generate_observations()
        if return_info:
            return obs, {}
        else:
            return obs

    def step(self, actions):
        """
        Process actions (placeholder), check if game ends, assign final rewards, return step data.
        """
        if self.game_done:
            # Game already over => no changes
            obs = self._generate_observations()
            dones = {agent: True for agent in self.agents}
            rewards = {agent: 0.0 for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
            self.agents = []
            return obs, rewards, dones, infos

        # 1. Process the actions (do nothing in this minimal example)
        # 2. Check if all 4 werewolves are still alive => end game
        werewolf_alive_count = 0
        total_wolves = 4
        for i in range(NUM_SEATS):
            if self.role_assignment[i] == "werewolf" and self.alive[i]:
                werewolf_alive_count += 1
        if werewolf_alive_count == total_wolves:
            self.game_done = True

        # 3. Rewards
        rewards = {agent: 0.0 for agent in self.agents}
        if self.game_done:
            # werewolves alive => werewolves "win" in this simplistic scenario
            for i, agent in enumerate(self.possible_agents):
                role = self.role_assignment[i]
                if role == "werewolf":
                    rewards[agent] = +1.0
                else:
                    rewards[agent] = -1.0

        # 4. Observations, dones, infos
        obs = self._generate_observations()
        dones = {agent: self.game_done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.game_done:
            self.agents = []

        return obs, rewards, dones, infos

    def _generate_observations(self):
        """
        Return a dict of seat_i -> small vector of int data.
        For example: [seat_id, isWolf(0/1), isAlive(1/0), 0, 0].
        """
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            if not self.alive[i]:
                obs[agent] = np.array([0,0,0,0,0], dtype=np.int32)
            else:
                seat_id = i
                is_wolf = 1 if (self.role_assignment[i] == "werewolf") else 0
                is_alive = 1 if self.alive[i] else 0
                obs[agent] = np.array([seat_id, is_wolf, is_alive, 0, 0], dtype=np.int32)
        return obs

    def render(self):
        """
        Print simple info if in "human" mode.
        """
        if self.render_mode == "human":
            print("===== RENDERING WEREWOLF ENV =====")
            for i, agent in enumerate(self.possible_agents):
                role = self.role_assignment[i]
                status = "alive" if self.alive[i] else "dead"
                print(f"{agent} - {role} - {status}")

    def close(self):
        pass
