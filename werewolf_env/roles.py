"""
Defines role distributions and constants for the Werewolf game.
"""

ALL_ROLES = [
    "werewolf",
    "villager",
    "seer",
    "witch",
    "hunter",
    "idiot"
]

# Example distribution: 4 werewolves, 4 villagers, 1 seer, 1 witch, 1 hunter, 1 idiot.
ROLE_DISTRIBUTION = (
    ["werewolf"] * 4 +
    ["villager"] * 4 +
    ["seer"] +
    ["witch"] +
    ["hunter"] +
    ["idiot"]
)
