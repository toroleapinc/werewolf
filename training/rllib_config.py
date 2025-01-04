# training/rllib_config.py

NUM_SEATS = 12

def seat_role_to_policy_name(seat_id, role_name):
    return f"policy_seat{seat_id}_{role_name}"

def build_policies():
    policies = {}
    for seat_id in range(NUM_SEATS):
        for role_name in ["werewolf", "villager"]:
            policy_name = seat_role_to_policy_name(seat_id, role_name)
            policies[policy_name] = (None, None, None, {})
    return policies

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    seat_id = int(agent_id.split("_")[1])
    if seat_id < 4:
        role = "werewolf"
    else:
        role = "villager"
    return seat_role_to_policy_name(seat_id, role)
