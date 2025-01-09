"""
Name of script: main.py

Driver script for WerewolfMultiAgentEnv. 
We create the environment and run one game in which each seat's action is chosen
with a "mask-based random policy" (meaning: we gather the valid actions from
the environment-provided mask, then randomly pick among them with uniform probability).

Key points:
 - No seat picks a dead seat or picks invalid seats, thanks to the environment's mask.
 - The environment no longer does random seat picking for actions such as Wolf kill,
   Seer check, or Witch poison. It only does random tie-breaking or random events
   (like hunter shots) that happen as "environment resolution."
 - This code is purely for demonstration of how to use the environment with
   a random, mask-based policy. In an RL scenario, you'd feed `obs["obs"]` and `obs["action_mask"]`
   into your RL policy or trainer.
"""

import random
import numpy as np

# Make sure you have the environment script accessible or installed as a module:
from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv, NUM_SEATS

def simulate_one_game():
    """
    Create the WerewolfMultiAgentEnv and run one game with random seat actions
    that only select from valid actions as indicated by the action_mask.
    """
    env = WerewolfMultiAgentEnv()
    obs, info = env.reset()

    done_flags = {agent_id: False for agent_id in env.agents}
    step_count = 0

    # Print initial seat roles (for debugging)
    print("\n--- START OF GAME ---")
    for seat_idx, agent_id in enumerate(env.agents):
        role = env.role_assignment[seat_idx]
        print(f"{agent_id}: {role}")
    print()

    while not all(done_flags.values()):
        step_count += 1
        action_dict = {}

        # Build actions for each alive agent using the action_mask
        for seat_idx, agent_id in enumerate(env.agents):
            if done_flags[agent_id]:
                # If we're done for this agent, it picks no action
                action_dict[agent_id] = 0
                continue

            # Extract the agent's observation
            agent_obs = obs[agent_id]
            action_mask = agent_obs["action_mask"]
            # Collect valid actions
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
            # Randomly pick among valid actions
            chosen_action = random.choice(valid_actions)
            action_dict[agent_id] = chosen_action

        # Step environment
        next_obs, rewards, terminated, truncated, infos = env.step(action_dict)

        print(f"\n--- STEP {step_count} ---")
        print(f"Phase: {env.phase}, DayCount: {env.day_count}")
        alive_list = [f"seat_{i}" for i in range(env.num_seats) if env.alive[i]]
        print("Alive seats:", alive_list)

        # Update done flags
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
