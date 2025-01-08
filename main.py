"""
Name of script: main.py

Driver script for the WerewolfMultiAgentEnv, ensuring:
 - No seat picks a dead seat or picks themselves if they are dead.
 - We use a "mask" approach for seat actions so RL or random code never chooses an invalid seat.

No changes here regarding the badge logic (that is in werewolf_multiagent.py).
We just ensure all seat picks reference valid, alive seats or 0 => no vote/no kill.
"""

import numpy as np
import random
from werewolf_env.werewolf_multiagent import WerewolfMultiAgentEnv

def simulate_one_game():
    """
    Create the WerewolfMultiAgentEnv and run one game with random seat actions
    that only select from valid, alive seats (excluding self).
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

        if env.phase == "ELECTION":
            # Each alive seat picks from {0,1,2} => 0=no-run,1=run,2=quit
            for seat_idx, agent_id in enumerate(env.agents):
                if not env.alive[seat_idx] or done_flags[agent_id]:
                    action_dict[agent_id] = 0
                else:
                    action_dict[agent_id] = random.randint(0, 2)

        elif env.phase == "DAY":
            # Each alive seat picks {0 => no vote} or a valid alive seat
            for seat_idx, agent_id in enumerate(env.agents):
                if not env.alive[seat_idx] or done_flags[agent_id]:
                    action_dict[agent_id] = 0
                else:
                    # 25% => no vote
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
            # Wolf => 0 => no kill, else pick from alive seats
            # Seer => pick from alive seats not self/known, else 0
            # Witch => if used heal => no 1, if used poison => no 2
            # Others => pick 0 or from alive seats
            for seat_idx, agent_id in enumerate(env.agents):
                if not env.alive[seat_idx] or done_flags[agent_id]:
                    action_dict[agent_id] = 0
                    continue

                role = env.role_assignment[seat_idx]

                if role == "werewolf":
                    alive_targets = [
                        s for s in range(env.num_seats)
                        if s != seat_idx and env.alive[s]
                    ]
                    options = [0] + [t+1 for t in alive_targets]
                    action_dict[agent_id] = random.choice(options)

                elif role == "seer":
                    known_dict = env.seer_knowledge.get(seat_idx, {})
                    valid_candidates = [
                        s for s in range(env.num_seats)
                        if s != seat_idx and env.alive[s] and (s not in known_dict)
                    ]
                    if not valid_candidates:
                        action_dict[agent_id] = 0
                    else:
                        chosen = random.choice(valid_candidates)
                        action_dict[agent_id] = chosen + 1

                elif role == "witch":
                    possible_actions = [0]
                    if not env.witch_heal_used:
                        possible_actions.append(1)
                    if not env.witch_poison_used:
                        possible_actions.append(2)
                    action_dict[agent_id] = random.choice(possible_actions)

                else:
                    # Other roles => 0 or pick from alive seats
                    alive_targets = [
                        s for s in range(env.num_seats)
                        if s != seat_idx and env.alive[s]
                    ]
                    picks = [0] + [t+1 for t in alive_targets]
                    action_dict[agent_id] = random.choice(picks)

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
