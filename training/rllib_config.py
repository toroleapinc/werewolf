"""
Example multi-agent config for seat-based role mapping.
We define policies for seat_i in {werewolf, villager} for simplicity,
but you can expand to all 6 roles if desired.
"""

NUM_SEATS = 12

def seat_role_to_policy_name(seat_id, role_name):
    """Generate a unique policy name for seat/role."""
    return f"policy_seat{seat_id}_{role_name}"

def build_policies():
    """
    Build a dict of policy specs. 
    Here we do seat i => {werewolf, villager} for demonstration.
    In a real scenario, you might do seat i => 6 roles => 72 policies.
    """
    policies = {}
    for seat_id in range(NUM_SEATS):
        for role_name in ["werewolf", "villager"]:
            policy_name = seat_role_to_policy_name(seat_id, role_name)
            policies[policy_name] = (
                None,  # Policy class (None => default)
                None,  # Obs space (inferred)
                None,  # Action space (inferred)
                {}     # Extra config
            )
    return policies

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """
    Return which policy each seat uses.
    For demonstration: seat_0..3 => "werewolf" policy, else "villager".
    In real usage, you'd query the environment's role assignment for each seat.
    """
    seat_id = int(agent_id.split("_")[1])
    if seat_id < 4:
        role = "werewolf"
    else:
        role = "villager"
    return seat_role_to_policy_name(seat_id, role)
