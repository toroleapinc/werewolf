# training/rllib_config.py
"""
Build seat-based policies for seat_i in {werewolf, villager} or full roles if you like.
For demonstration, we only do 'werewolf' vs. 'villager' to keep it simpler.
"""

NUM_SEATS = 12

def seat_role_to_policy_name(seat_id, role_name):
    return f"policy_seat{seat_id}_{role_name}"

def build_policies():
    """
    Minimal example: seat_i => werewolf or villager.
    If you want all 6 roles, you can expand the loop or do a role-check in the environment's reset.
    """
    policies = {}
    for seat_id in range(NUM_SEATS):
        # Suppose seat_id < 4 => Wolf, else villager
        # but you can do a 12x6 approach if you want 72 policies
        if seat_id < 4:
            role_name = "werewolf"
        else:
            role_name = "villager"
        policy_name = seat_role_to_policy_name(seat_id, role_name)
        policies[policy_name] = (None, None, None, {})
    return policies

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """
    Simple approach: seat_0..3 => 'werewolf', else 'villager'.
    Or you can read from the environment's role assignment if you stored that externally.
    """
    seat_id = int(agent_id.split("_")[1])
    if seat_id < 4:
        return f"policy_seat{seat_id}_werewolf"
    else:
        return f"policy_seat{seat_id}_villager"
