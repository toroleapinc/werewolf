"""
A simple script to run one environment episode with random actions for debugging.
"""

import numpy as np
from werewolf_env import WerewolfParallelEnv

def main():
    env = WerewolfParallelEnv()
    obs = env.reset()
    done_flags = {agent: False for agent in env.agents}

    while any(not done for done in done_flags.values()):
        actions = {}
        for agent in env.agents:
            if not done_flags[agent]:
                actions[agent] = env.action_spaces[agent].sample()
        obs, rewards, dones, infos = env.step(actions)
        done_flags.update(dones)

    print("Episode finished!")
    for agent, rew in rewards.items():
        print(f"{agent} => reward {rew}")

if __name__ == "__main__":
    main()
