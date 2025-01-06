# werewolf_env/werewolf_multiagent.py
"""
A seat-based multi-agent Werewolf environment for RLlib, with logging of each seat's decisions:
 - 12 seats: seat_0.. seat_11
 - Each seat can be Wolf, Seer, Witch, Hunter, Idiot, or Villager (randomly assigned)
 - We embed 'role_id' in the observation so the policy knows its role
 - We have night kills (Wolves collectively pick, Witch can heal/poison) and day voting
 - We print seat-by-seat decisions so you can see who kills/votes for whom.
"""

import random
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box

ALL_ROLES = ["werewolf", "villager", "seer", "witch", "hunter", "idiot"]
def role_to_int(role_name):
    """ 
    Map role strings to integer IDs, so seat-based policy knows current role:
      werewolf->0, villager->1, seer->2, witch->3, hunter->4, idiot->5
    """
    return ALL_ROLES.index(role_name)

# By default: 4 wolves, 4 villagers, 1 seer, 1 witch, 1 hunter, 1 idiot
ROLE_DISTRIBUTION = (
    ["werewolf"]*4 +
    ["villager"]*4 +
    ["seer"] +
    ["witch"] +
    ["hunter"] +
    ["idiot"]
)
NUM_SEATS = 12

def assign_roles():
    """Shuffle the 12-seat distribution."""
    roles = ROLE_DISTRIBUTION[:]
    random.shuffle(roles)
    return roles

class WerewolfMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent environment for seat-based approach.
    Each seat sees an observation with role_id, day_count, etc. 
    We log each seat's decision during night/day for clarity.
    """

    def __init__(self, config=None):
        super().__init__()
        self.num_seats = NUM_SEATS
        self.agents = [f"seat_{i}" for i in range(self.num_seats)]
        self._agent_ids = set(self.agents)

        # Each seat picks a seat (0..11) to kill/vote
        self.action_space = Discrete(NUM_SEATS)

        # We'll store a 9-int observation:
        # [0: seat_idx,
        #  1: alive(1/0),
        #  2: day_count,
        #  3: phase(1=day, 0=night),
        #  4: role_id(0..5),
        #  5: witch_heal_used(0/1),
        #  6: witch_poison_used(0/1),
        #  7: extra=0,
        #  8: isWolf(1/0) for debugging or remove if you prefer]
        self.observation_space = Box(low=0, high=100, shape=(9,), dtype=np.int32)

        # Internal state
        self.role_assignment = []
        self.alive = []
        self.phase = "NIGHT"  # or "DAY"
        self.day_count = 0
        self.episode_terminated = False
        self.episode_truncated = False

        # Witch usage
        self.witch_heal_used = False
        self.witch_poison_used = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Randomly assign roles
        self.role_assignment = assign_roles()
        self.alive = [True]*self.num_seats
        self.phase = "NIGHT"
        self.day_count = 0
        self.episode_terminated = False
        self.episode_truncated = False

        self.witch_heal_used = False
        self.witch_poison_used = False

        obs_dict = {}
        for i, agent_id in enumerate(self.agents):
            obs_dict[agent_id] = self._get_obs(i)
        info_dict = {}
        return obs_dict, info_dict

    def step(self, action_dict):
        if self.episode_terminated or self.episode_truncated:
            obs = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
            rew = {agent: 0.0 for agent in self.agents}
            ter = {agent: True for agent in self.agents}
            tru = {agent: False for agent in self.agents}
            ter["__all__"] = True
            tru["__all__"] = self.episode_truncated
            info = {agent: {} for agent in self.agents}
            return obs, rew, ter, tru, info

        # Day->Night->Day toggle
        if self.phase=="NIGHT":
            self._night_phase(action_dict)
            self.phase = "DAY"
        else:
            self._day_phase(action_dict)
            self.day_count += 1
            self.phase = "NIGHT"

        if self.day_count>10:
            self.episode_truncated = True

        self._check_winner()

        obs = {}
        for i, agent_id in enumerate(self.agents):
            obs[agent_id] = self._get_obs(i)

        rew = self._build_rewards()

        ter = {}
        tru = {}
        if self.episode_terminated or self.episode_truncated:
            for agent_id in self.agents:
                ter[agent_id] = self.episode_terminated
                tru[agent_id] = self.episode_truncated
            ter["__all__"] = self.episode_terminated
            tru["__all__"] = self.episode_truncated
        else:
            for agent_id in self.agents:
                ter[agent_id] = False
                tru[agent_id] = False
            ter["__all__"] = False
            tru["__all__"] = False

        info = {agent: {} for agent in self.agents}
        return obs, rew, ter, tru, info

    def _night_phase(self, action_dict):
        """
        Wolves collectively kill => final kill_target by majority.
        Witch can heal/poison if not used.
        We also log each Wolf's choice, the final kill target, and witch actions.
        """
        # 1) Wolves pick
        wolf_targets = []
        # print("\nNight: Wolves' kill votes:")
        for i in range(self.num_seats):
            if self.alive[i] and (self.role_assignment[i]=="werewolf"):
                agent_id = f"seat_{i}"
                kill_idx = action_dict.get(agent_id, i)
                kill_idx = self._validate_seat(kill_idx)
                wolf_targets.append(kill_idx)
                # print(f"  Wolf seat_{i} => seat_{kill_idx}")

        kill_target = None
        if wolf_targets:
            # majority
            kill_target = max(set(wolf_targets), key=wolf_targets.count)
            # print(f"Night: final Wolf kill target => seat_{kill_target}")

        # 2) Witch sees kill_target
        if kill_target is not None:
            witch_idx = None
            for seat_i in range(self.num_seats):
                if self.alive[seat_i] and (self.role_assignment[seat_i]=="witch"):
                    witch_idx = seat_i
                    break
            if witch_idx is not None:
                agent_id = f"seat_{witch_idx}"
                w_action = action_dict.get(agent_id, 0)
                # 0 => do nothing, 1 => heal kill_target, 2.. => poison seat
                if (w_action==1) and (not self.witch_heal_used):
                    # print(f"Night: Witch seat_{witch_idx} => HEAL seat_{kill_target} => kill canceled")
                    kill_target = None
                    self.witch_heal_used = True
                elif (w_action>=2) and (not self.witch_poison_used):
                    poison_seat = w_action-2
                    poison_seat = self._validate_seat(poison_seat)
                    # print(f"Night: Witch seat_{witch_idx} => POISON seat_{poison_seat}")
                    self.witch_poison_used = True
                    self._kill_seat(poison_seat, "witch_poison")

        if kill_target is not None:
            # print(f"Night: Wolf kill => seat_{kill_target}")
            self._kill_seat(kill_target, "wolf_kill")

    def _day_phase(self, action_dict):
        """
        Each alive seat votes => top seat is killed unless tie => skip.
        We log each seat's vote, final kill target, etc.
        """
        votes = {}
        # print("\nDay: Voting")
        for i in range(self.num_seats):
            if self.alive[i]:
                agent_id = f"seat_{i}"
                vote_idx = action_dict.get(agent_id, i)
                vote_idx = self._validate_seat(vote_idx)
                votes[vote_idx] = votes.get(vote_idx, 0)+1
                # print(f"  seat_{i} ({self.role_assignment[i]}) => seat_{vote_idx}")

        if not votes:
            return

        # find top
        kill_target, topcount = None, -1
        for seat_i, vcount in votes.items():
            if vcount>topcount:
                kill_target = seat_i
                topcount = vcount

        # tie?
        count_with_top = sum(v==topcount for v in votes.values())
        if count_with_top>1:
            # print("Day: tie => no elimination")
            return

        if kill_target is not None:
            # print(f"Day: kill => seat_{kill_target}")
            self._kill_seat(kill_target, "day_vote")

    def _kill_seat(self, seat_idx, reason):
        """Eliminate seat_idx. If Hunter => immediate shot. If Idiot => maybe survive once, etc."""
        if seat_idx<0 or seat_idx>=self.num_seats:
            return
        if not self.alive[seat_idx]:
            return
        role = self.role_assignment[seat_idx]
        # print(f"  seat_{seat_idx} ({role}) => KILLED by {reason}")

        self.alive[seat_idx] = False

        # If it's Hunter => revenge shot
        if role=="hunter":
            possible_targets = [i for i in range(self.num_seats) if self.alive[i] and i!=seat_idx]
            if possible_targets:
                victim = random.choice(possible_targets)
                # print(f"  Hunter seat_{seat_idx} => SHOOTS seat_{victim}")
                self.alive[victim] = False

        # If it's Idiot and reason=="day_vote", you could let them survive once if you want, etc.

    def _check_winner(self):
        """If wolves=0 => village wins, if wolves>=others => wolf wins."""
        wolves_alive = sum(self.alive[i] and self.role_assignment[i]=="werewolf" for i in range(self.num_seats))
        vill_alive = sum(self.alive[i] and self.role_assignment[i]!="werewolf" for i in range(self.num_seats))
        if wolves_alive==0:
            self.episode_terminated = True
        elif wolves_alive>=vill_alive:
            self.episode_terminated = True

    def _build_rewards(self):
        """
        +1 for winners, -1 for losers if ended, else 0 mid-game
        """
        rew = {}
        if not self.episode_terminated and not self.episode_truncated:
            for agent_id in self.agents:
                rew[agent_id] = 0.0
            return rew

        # If ended
        wolves_alive = sum(self.alive[i] and self.role_assignment[i]=="werewolf" for i in range(self.num_seats))
        if wolves_alive==0:
            # Village wins
            for i, agent_id in enumerate(self.agents):
                if self.role_assignment[i]=="werewolf":
                    rew[agent_id] = -1.0
                else:
                    rew[agent_id] = +1.0
        else:
            # Wolf >= others => wolf wins
            for i, agent_id in enumerate(self.agents):
                if self.role_assignment[i]=="werewolf":
                    rew[agent_id] = +1.0
                else:
                    rew[agent_id] = -1.0
        return rew

    def _get_obs(self, seat_idx):
        """
        9-int obs: 
          0 seat_idx
          1 alive(1/0)
          2 day_count
          3 phase(1=day,0=night)
          4 role_id(0..5)
          5 witch_heal_used(0/1)
          6 witch_poison_used(0/1)
          7 extra=0
          8 isWolf(1/0) for debugging 
        """
        seat_id = seat_idx
        alive_flag = 1 if self.alive[seat_idx] else 0
        day_val = self.day_count
        phase_val = 1 if (self.phase=="DAY") else 0
        role_name = self.role_assignment[seat_idx]
        rid = role_to_int(role_name)
        wh_used = 1 if self.witch_heal_used else 0
        wp_used = 1 if self.witch_poison_used else 0
        extra = 0
        is_wolf = 1 if (role_name=="werewolf") else 0

        return np.array([
            seat_id,
            alive_flag,
            day_val,
            phase_val,
            rid,
            wh_used,
            wp_used,
            extra,
            is_wolf
        ], dtype=np.int32)

    def _validate_seat(self, seat_idx):
        """Clamp seat_idx to [0..11] so we never go out-of-range."""
        return max(0, min(seat_idx, self.num_seats-1))
