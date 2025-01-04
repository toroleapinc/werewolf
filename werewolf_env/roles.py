# werewolf_env/roles.py
"""
Defines role distribution for the 12 seats. 
By default:
- 4 werewolves
- 4 villagers
- 1 seer
- 1 witch
- 1 hunter
- 1 idiot
"""

ALL_ROLES = [
    "werewolf",
    "villager",
    "seer",
    "witch",
    "hunter",
    "idiot"
]

# Exactly 4 werewolves, 4 villagers, 1 seer, 1 witch, 1 hunter, 1 idiot
ROLE_DISTRIBUTION = (
    ["werewolf"] * 4 +
    ["villager"] * 4 +
    ["seer"] +
    ["witch"] +
    ["hunter"] +
    ["idiot"]
)
