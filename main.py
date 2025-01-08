"""
Name of script: main.py

Example driver script for the WerewolfMultiAgentEnv, where:
 - We create the environment from werewolf_env.werewolf_multiagent.
 - On each step, we supply valid actions depending on the current phase:
    ELECTION => seats pick {0,1,2}
    DAY => seats pick from [0..NUM_SEATS]
    NIGHT => Wolf seats pick [1..NUM_SEATS], Seer picks [1..NUM_SEATS], Witch picks [0..2], etc.

We do random seat actions for demonstration. Kills from pending lists
(after ELECTION) will be applied at the start of the next Day.
"""

import numpy as np
import random
from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv

def simulate_one_game():
    """
    Create the WerewolfMultiAgentEnv, run one game with random actions for demonstration.
    """
    env = WerewolfMultiAgentEnv()
    obs, info = env.reset()

    done_flags = {agent_id: False for agent_id in env.agents}
    step_count = 0

    # Print initial seat roles (for debugging only)
    print("\n--- START OF GAME ---")
    for seat_idx, agent_id in enumerate(env.agents):
        role = env.role_assignment[seat_idx]
        print(f"{agent_id}: {role}")
    print()

    while not all(done_flags.values()):
        step_count += 1
        action_dict = {}

        if env.phase == "ELECTION":
            # Each alive seat picks from {0,1,2}
            for seat_idx, agent_id in enumerate(env.agents):
                if done_flags[agent_id] or not env.alive[seat_idx]:
                    action_dict[agent_id] = 0
                else:
                    action_dict[agent_id] = random.randint(0, 2)

        elif env.phase == "DAY":
            # Each alive seat picks from [0..NUM_SEATS]
            for seat_idx, agent_id in enumerate(env.agents):
                if done_flags[agent_id] or not env.alive[seat_idx]:
                    action_dict[agent_id] = 0
                else:
                    # 25% => no vote, else pick random alive seat
                    if random.random() < 0.25:
                        action_dict[agent_id] = 0
                    else:
                        alive_targets = [
                            s for s in range(env.num_seats)
                            if s != seat_idx and env.alive[s]
                        ]
                        if not alive_targets:
                            action_dict[agent_id] = 0
                        else:
                            chosen = random.choice(alive_targets)
                            action_dict[agent_id] = chosen + 1

        else:  # NIGHT
            # Wolf => pick [1..NUM_SEATS], Seer => pick [1..NUM_SEATS], Witch => [0..2], etc.
            for seat_idx, agent_id in enumerate(env.agents):
                if done_flags[agent_id] or not env.alive[seat_idx]:
                    action_dict[agent_id] = 0
                    continue

                role = env.role_assignment[seat_idx]
                if role == "werewolf":
                    alive_targets = [
                        s for s in range(env.num_seats)
                        if s != seat_idx and env.alive[s]
                    ]
                    if not alive_targets:
                        action_dict[agent_id] = 0
                    else:
                        chosen = random.choice(alive_targets)
                        action_dict[agent_id] = chosen + 1
                elif role == "seer":
                    alive_targets = [
                        s for s in range(env.num_seats)
                        if s != seat_idx and env.alive[s]
                    ]
                    if not alive_targets:
                        action_dict[agent_id] = 0
                    else:
                        chosen = random.choice(alive_targets)
                        action_dict[agent_id] = chosen + 1
                elif role == "witch":
                    action_dict[agent_id] = random.randint(0, 2)
                else:
                    if random.random() < 0.25:
                        action_dict[agent_id] = 0
                    else:
                        alive_targets = [
                            s for s in range(env.num_seats)
                            if s != seat_idx and env.alive[s]
                        ]
                        if not alive_targets:
                            action_dict[agent_id] = 0
                        else:
                            chosen = random.choice(alive_targets)
                            action_dict[agent_id] = chosen + 1

        # Step environment
        next_obs, rewards, terminated, truncated, infos = env.step(action_dict)

        print(f"\n--- STEP {step_count} ---")
        print(f"Phase: {env.phase}, DayCount: {env.day_count}")

        # Show which seats are alive
        alive_list = [f"seat_{i}" for i in range(env.num_seats) if env.alive[i]]
        print("Alive seats:", alive_list)

        # Update done flags
        for agent_id, term in terminated.items():
            if agent_id == "__all__":
                continue
            if term and not done_flags[agent_id]:
                pass

        for agent_id in env.agents:
            if terminated[agent_id] or truncated[agent_id]:
                done_flags[agent_id] = True

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
