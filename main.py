# File name: main.py

"""
Demonstration script for WerewolfMultiAgentEnv using random, mask-based actions.
"""

import random
import numpy as np
from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv, NUM_SEATS

def simulate_one_game():
    env = WerewolfMultiAgentEnv()
    obs, info = env.reset()

    done_flags = {agent_id: False for agent_id in env.agents}
    step_count = 0

    print("\n--- START OF GAME ---")
    for seat_idx, agent_id in enumerate(env.agents):
        role = env.role_assignment[seat_idx]
        print(f"{agent_id}: {role}")
    print()

    while not all(done_flags.values()):
        step_count += 1
        action_dict = {}

        for seat_idx, agent_id in enumerate(env.agents):
            if done_flags[agent_id]:
                action_dict[agent_id] = 0
                continue

            agent_obs = obs[agent_id]
            # agent_obs is { "obs": <ndarray shape=(10,)>, "action_mask": <ndarray shape=(NUM_SEATS+3,)> }
            action_mask = agent_obs["action_mask"]
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
            chosen_action = random.choice(valid_actions)
            action_dict[agent_id] = chosen_action

        next_obs, rewards, terminated, truncated, infos = env.step(action_dict)

        print(f"\n--- STEP {step_count} ---")
        print(f"Phase: {env.phase}, DayCount: {env.day_count}")
        alive_list = [f"seat_{i}" for i in range(env.num_seats) if env.alive[i]]
        print("Alive seats:", alive_list)

        for agent_id in env.agents:
            if terminated[agent_id] or truncated[agent_id]:
                done_flags[agent_id] = True

        obs = next_obs

        if all(done_flags.values()):
            print("\n--- END OF GAME ---")
            print("Final rewards:")
            for agent_id in env.agents:
                print(f"{agent_id} => {rewards[agent_id]}")
            break

def main():
    simulate_one_game()

if __name__ == "__main__":
    main()
