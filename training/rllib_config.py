# training/rllib_config.py
"""
Config utilities for our seat-based multi-agent Werewolf setup.

We define:
- build_policies(): Creates 12 seat-based policy IDs (seat_0_policy..seat_11_policy)
- policy_mapping_fn(): Returns the seat_N_policy for agent_id "seat_N"

No mention of "werewolf" or "villager" in the policy name; the environment does random roles
under the hood, and the seat sees `role_id` in its observation to adapt as Wolf, Witch, etc.
"""

def build_policies():
    """
    Build 12 seat-based policies:
      seat_0_policy, seat_1_policy, ... seat_11_policy
    Each seat uses exactly one policy across all episodes,
    even if the seat's role changes.
    """
    policies = {}
    for seat_id in range(12):
        policy_id = f"seat_{seat_id}_policy"
        # RLlib expects a tuple: (policy_cls, obs_space, act_space, config)
        # We can do (None, None, None, {}) so RLlib uses defaults. 
        policies[policy_id] = (None, None, None, {})
    return policies


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """
    Called each time RLlib needs to figure out which policy to use for a given agent.
    agent_id will be like "seat_0", "seat_1", etc.
    We just return "seat_0_policy" for "seat_0", etc.
    """
    # agent_id is "seat_X"
    return f"{agent_id}_policy"
