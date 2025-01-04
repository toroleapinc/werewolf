# main.py
"""
Observe a single random Werewolf game for debugging/inspection.
We fix the KeyError by skipping the "__all__" key in the terminated dict.
"""

import numpy as np
from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv

def simulate_one_game():
    """
    Create the environment, run one random-play game,
    and log the state after each step (NIGHT or DAY).
    """
    # 1) Create the environment
    env = WerewolfMultiAgentEnv()
    # 2) Reset (new game)
    obs, info = env.reset()

    # Track which agents are done
    done_flags = {agent_id: False for agent_id in env.agents}
    step_count = 0

    # Show initial roles for clarity
    print("\n--- START OF GAME ---")
    for seat_idx, agent_id in enumerate(env.agents):
        role = env.role_assignment[seat_idx]
        print(f"{agent_id}: {role}")
    print()

    while not all(done_flags.values()):
        step_count += 1

        # Create random seat-based action for each "alive" agent
        action_dict = {}
        for seat_idx, agent_id in enumerate(env.agents):
            if not done_flags[agent_id]:
                # pick a random seat in [0..11]
                action_dict[agent_id] = np.random.randint(0, env.num_seats)

        # Step the environment
        next_obs, rewards, terminated, truncated, infos = env.step(action_dict)

        # Print some info about the step
        print(f"\n--- STEP {step_count} ---")
        print(f"Phase: {env.phase}, DayCount: {env.day_count}")

        # Show which seats are still alive
        alive_list = [f"seat_{i}" for i in range(env.num_seats) if env.alive[i]]
        print("Alive seats:", alive_list)

        # Avoid the KeyError by skipping the "__all__" key
        for agent_id, term in terminated.items():
            # RLlib adds "__all__" => skip or handle separately
            if agent_id == "__all__":
                continue
            if term and not done_flags[agent_id]:
                # This agent is newly done
                pass  # we won't do anything special here

        # Update done_flags
        for agent_id in env.agents:
            if terminated[agent_id] or truncated[agent_id]:
                done_flags[agent_id] = True

        # If all agents are done, we have a final outcome
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
