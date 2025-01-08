"""
Name of script: werewolf_multiagent.py

A seat-based multi-agent Werewolf environment with:
 - NIGHT -> ELECTION -> DAY -> (repeat) phases.
 - Wolf majority kill, Seer checks, Witch potions, Hunter shot, Idiot reveal.
 - Badge mechanic for day tie-break.
 - Kills from pending lists after ELECTION are now applied at the *start* of the next Day,
   so that lines like seat_9 => KILLED are printed in the Day logs.
 - We do NOT reveal the cause of death in the logs.

Implementation:
 1) If ELECTION ends, we do NOT immediately apply self.pending_kills_first_night (or any kills).
 2) We wait until the next Day phase (i.e. in _day_phase_two_round) to do so.
"""

import random
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box

ALL_ROLES = ["werewolf", "villager", "seer", "witch", "hunter", "idiot"]

def role_to_int(role_name):
    """Map role string -> int: werewolf->0, villager->1, seer->2, witch->3, hunter->4, idiot->5."""
    return ALL_ROLES.index(role_name)

ROLE_DISTRIBUTION = (
    ["werewolf"]*4 +
    ["villager"]*4 +
    ["seer"] +
    ["witch"] +
    ["hunter"] +
    ["idiot"]
)
NUM_SEATS = 12

def assign_roles():
    roles = ROLE_DISTRIBUTION[:]
    random.shuffle(roles)
    return roles

class WerewolfMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.num_seats = NUM_SEATS
        self.agents = [f"seat_{i}" for i in range(self.num_seats)]
        self._agent_ids = set(self.agents)

        self.phase = "NIGHT"

        self.action_space = Discrete(NUM_SEATS + 1)
        self.observation_space = Box(low=0, high=255, shape=(10,), dtype=np.int32)

        self.role_assignment = []
        self.alive = []
        self.day_count = 0
        self.episode_terminated = False
        self.episode_truncated = False

        # Witch potions
        self.witch_heal_used = False
        self.witch_poison_used = False

        self.badge_holder = -1
        self.election_done = False

        # Seer knowledge => {seer_idx: {seat_idx: bool}}
        self.seer_knowledge = {}

        # For kills if day_count==0 & not election_done
        self.pending_kills_first_night = []

        # Track idiot reveal
        self.idiot_revealed = [False]*self.num_seats

        # Track kills that happen at night
        self.last_night_deaths = []

        # NEW: We'll also track kills that happen immediately after ELECTION
        # but want them to appear in the Day logs, so we'll store them in
        # self.pending_post_election_kills and apply them at the start of Day.
        self.pending_post_election_kills = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.role_assignment = assign_roles()
        self.alive = [True]*self.num_seats

        self.phase = "NIGHT"
        self.day_count = 0
        self.episode_terminated = False
        self.episode_truncated = False

        self.witch_heal_used = False
        self.witch_poison_used = False
        self.badge_holder = -1
        self.election_done = False

        self.seer_knowledge.clear()
        self.pending_kills_first_night = []
        self.pending_post_election_kills = []
        self.idiot_revealed = [False]*self.num_seats
        self.last_night_deaths = []

        for i, role in enumerate(self.role_assignment):
            if role == "seer":
                self.seer_knowledge[i] = {}

        obs_dict = {}
        for i, agent_id in enumerate(self.agents):
            obs_dict[agent_id] = self._get_obs(i)
        return obs_dict, {}

    def step(self, action_dict):
        if self.episode_terminated or self.episode_truncated:
            obs = {a: self._get_obs(i) for i,a in enumerate(self.agents)}
            rew = {a: 0.0 for a in self.agents}
            ter = {a: True for a in self.agents}
            tru = {a: False for a in self.agents}
            ter["__all__"] = True
            tru["__all__"] = self.episode_truncated
            return obs, rew, ter, tru, {a:{} for a in self.agents}

        if self.phase == "NIGHT":
            self._night_phase(action_dict)
            if self.day_count == 0 and not self.election_done:
                self.phase = "ELECTION"
            else:
                # DO NOT apply kills here if the environment just finished ELECTION
                # Instead, we'll apply them at the start of the next day.
                self.phase = "DAY"

        elif self.phase == "ELECTION":
            self._election_phase(action_dict)
            # We store kills in self.pending_kills_first_night (or other kills)
            # but do not apply them here. We'll do it in the next Day phase.
            self.election_done = True
            self.phase = "DAY"

        elif self.phase == "DAY":
            # (1) Apply any leftover kills from either first-night or post-election
            self._apply_kills(self.pending_kills_first_night, nightkill=True)
            self._apply_kills(self.pending_post_election_kills, nightkill=True)

            # (2) Now do the day phase logic
            self._day_phase_two_round(action_dict)
            self.day_count += 1
            self.phase = "NIGHT"

        if self.day_count > 10:
            self.episode_truncated = True

        self._check_winner()

        obs = {}
        for i,a in enumerate(self.agents):
            obs[a] = self._get_obs(i)

        rew = self._build_rewards()

        ter = {}
        tru = {}
        if self.episode_terminated or self.episode_truncated:
            for a in self.agents:
                ter[a] = self.episode_terminated
                tru[a] = self.episode_truncated
            ter["__all__"] = self.episode_terminated
            tru["__all__"] = self.episode_truncated
        else:
            for a in self.agents:
                ter[a] = False
                tru[a] = False
            ter["__all__"] = False
            tru["__all__"] = False

        return obs, rew, ter, tru, {a:{} for a in self.agents}

    # ---------------- NIGHT PHASE ---------------
    def _night_phase(self, action_dict):
        print(f"\nNight: Wolf kills => majority. day_count={self.day_count}")
        wolf_targets = []
        for i in range(self.num_seats):
            if self.alive[i] and self.role_assignment[i] == "werewolf":
                pick = action_dict.get(f"seat_{i}", 0)
                seat_cand = i
                if pick > 0:
                    seat_cand = (pick - 1) % self.num_seats
                wolf_targets.append(seat_cand)
                print(f"  Wolf seat_{i} => seat_{seat_cand}")

        kill_target = None
        if wolf_targets:
            kill_target = max(set(wolf_targets), key=wolf_targets.count)
            print(f"Night: final kill target => seat_{kill_target}")

        # Seer
        for i, ro in enumerate(self.role_assignment):
            if self.alive[i] and ro == "seer":
                spick = action_dict.get(f"seat_{i}", 0)
                cidx = i
                if spick > 0:
                    cidx = (spick - 1) % self.num_seats
                iswolf = (self.role_assignment[cidx] == "werewolf")
                print(f"  Seer seat_{i} => checks seat_{cidx} => Wolf? {iswolf}")
                if i not in self.seer_knowledge:
                    self.seer_knowledge[i] = {}
                self.seer_knowledge[i][cidx] = iswolf

        # Witch
        if kill_target is not None:
            wseat = None
            for si, rr in enumerate(self.role_assignment):
                if rr == "witch" and self.alive[si]:
                    wseat = si
                    break
            if wseat is not None:
                wact = action_dict.get(f"seat_{wseat}", 0)
                if wact < 0 or wact > 2:
                    wact = 0
                if wact == 1:  # Heal
                    if not self.witch_heal_used:
                        print(f"  Witch seat_{wseat} => HEAL seat_{kill_target}, kill canceled")
                        kill_target = None
                        self.witch_heal_used = True
                    else:
                        print(f"  Witch seat_{wseat} => tries HEAL but can't => do nothing")
                elif wact == 2:  # Poison
                    if not self.witch_poison_used:
                        possible = [
                            x for x in range(self.num_seats)
                            if x != wseat and self.alive[x] and x != kill_target
                        ]
                        if possible:
                            victim = random.choice(possible)
                            print(f"  Witch seat_{wseat} => POISON seat_{victim}")
                            self.witch_poison_used = True
                            if self.day_count == 0 and not self.election_done:
                                print(f"Night: first-night poison is pending => seat_{victim}")
                                self.pending_kills_first_night.append(("poison", victim))
                            else:
                                # If we were after the ELECTION, store in post_election kills
                                if self.election_done:
                                    self.pending_post_election_kills.append(("poison", victim))
                                else:
                                    self.last_night_deaths.append({
                                        "seat_idx": victim,
                                        "reason": "poison"
                                    })
                                    self._kill_seat(victim, reason="poison", nightkill=True)
                        else:
                            print(f"  Witch seat_{wseat} => no valid poison target => do nothing")
                    else:
                        print("  Witch => poison used => do nothing")
                else:
                    print("  Witch => does NOTHING")

        if kill_target is not None:
            # If day_count==0 & not election_done => store in pending_kills_first_night
            if self.day_count == 0 and not self.election_done:
                print(f"Night: first-night kill is pending => seat_{kill_target}")
                self.pending_kills_first_night.append(("wolf", kill_target))
            else:
                # If we are after ELECTION => store in pending_post_election_kills
                if self.election_done:
                    print(f"Night: kill target seat_{kill_target} is pending post-election")
                    self.pending_post_election_kills.append(("wolf", kill_target))
                else:
                    print(f"Night: seat_{kill_target} is chosen for elimination")
                    self.last_night_deaths.append({
                        "seat_idx": kill_target,
                        "reason": "wolf"
                    })
                    self._kill_seat(kill_target, reason="wolf", nightkill=True)

    # ---------------- ELECTION PHASE ---------------
    def _election_phase(self, action_dict):
        print("\nELECTION PHASE with PK-speech tie resolution")

        runner_set = set()
        quitter = set()

        for i in range(self.num_seats):
            if self.alive[i]:
                a = action_dict.get(f"seat_{i}", 0)
                if a == 1:
                    runner_set.add(i)
                elif a == 2:
                    quitter.add(i)

        round1_runners = sorted(list(runner_set | quitter))
        round1_quitter = sorted(list(quitter))
        final_runners = sorted(list(runner_set))
        print(f"  Round1 runners => {round1_runners}")
        print(f"  Round1 quitter => {round1_quitter}")
        print(f"  Round1 final runners => {final_runners}")

        if not final_runners:
            print("  Round1 => no final_runners => no badge assigned.")
            return

        alive_set = set(i for i in range(self.num_seats) if self.alive[i])
        votes_r1 = {}

        # Round1 sub-actions
        for i in alive_set:
            if i not in final_runners and i not in quitter:
                sub = random.randint(0, 1)
                if sub == 0:
                    print(f"    seat_{i} => no vote (sub=0)")
                    continue
                choice = random.choice(final_runners)
                print(f"    seat_{i} => votes runner seat_{choice} (sub=1)")
                votes_r1[choice] = votes_r1.get(choice, 0) + 1

        print(f"  Round1 votes => {dict(votes_r1)}")
        if not votes_r1:
            print("  Round1 => no valid votes => treat as tie among all final_runners => Round2")
            tie_seats = final_runners[:]
            print(f"  All tie seats => PK speech => {tie_seats}")
            round2_votes = {}
            tie_set = set(tie_seats)
            for j in alive_set:
                if j not in tie_set:
                    s2 = random.randint(0, 1)
                    if s2 == 0:
                        print(f"    seat_{j} => no vote (sub=0)")
                        continue
                    c2 = random.choice(tie_seats)
                    print(f"    seat_{j} => votes runner seat_{c2} (sub=1, tie_set=final_runners)")
                    round2_votes[c2] = round2_votes.get(c2, 0) + 1

            print(f"  Round2 votes => {dict(round2_votes)}")
            if not round2_votes:
                print("  Round2 => no valid votes => no badge assigned.")
                return
            tv = max(round2_votes.values())
            tcs = [k for k,v in round2_votes.items() if v == tv]
            if len(tcs) == 1:
                w2 = tcs[0]
                print(f"  Round2 => seat_{w2} singled out => badge assigned.")
                self.badge_holder = w2
            else:
                print(f"  Round2 => tie => no badge => seats {tcs}")
            return

        best_val = max(votes_r1.values())
        top_cands = [k for k,v in votes_r1.items() if v == best_val]
        if len(top_cands) == 1:
            w = top_cands[0]
            print(f"  Round1 => seat_{w} singled out => badge assigned.")
            self.badge_holder = w
            return

        # tie => Round2
        print(f"  Round1 => tie => seats {top_cands} => PK speech => Round2")
        round2_votes = {}
        tie_set = set(top_cands)
        for i in alive_set:
            if i not in tie_set:
                sub2 = random.randint(0, 1)
                if sub2 == 0:
                    print(f"    seat_{i} => no vote (sub=0)")
                    continue
                c2 = random.choice(top_cands)
                print(f"    seat_{i} => votes seat_{c2} in Round2 tie")
                round2_votes[c2] = round2_votes.get(c2, 0) + 1

        print(f"  Round2 votes => {dict(round2_votes)}")
        if not round2_votes:
            print("  Round2 => no valid votes => no badge assigned.")
            return
        mx = max(round2_votes.values())
        candz = [xx for xx,vv in round2_votes.items() if vv == mx]
        if len(candz) == 1:
            w2 = candz[0]
            print(f"  Round2 => seat_{w2} singled out => badge assigned.")
            self.badge_holder = w2
        else:
            print(f"  Round2 => tie => no badge => seats {candz}")

    # ---------------- DAY PHASE ---------------
    def _day_phase_two_round(self, action_dict):
        """
        2-round day vote => if tie => Round2 => if tie => no elimination.
        - We apply leftover kills from pending lists at the start.
        - Announce kills from last night (and from these pending lists).
        - If a Hunter died => can shoot in the morning if they wish.
        - Then proceed with normal day voting, Idiot reveal, etc.
        """

        # 1) Announce kills from last night that were just applied
        if self.last_night_deaths:
            print("\nAnnouncing kills from last night:")
            for entry in self.last_night_deaths:
                seat_idx = entry["seat_idx"]
                print(f"  seat_{seat_idx} died last night.")
                # If seat_idx was a Hunter => random sub=0 => no shot, sub=1 => shot
                if self.role_assignment[seat_idx] == "hunter":
                    sub = random.randint(0,1)
                    if sub == 1:
                        possible = [
                            i for i in range(self.num_seats)
                            if self.alive[i] and i != seat_idx
                        ]
                        if possible:
                            victim = random.choice(possible)
                            print(f"  seat_{seat_idx} (Hunter) opens fire on seat_{victim}!")
                            self._kill_seat(victim, reason="hunter_shot_morning", nightkill=False)
                        else:
                            print(f"  seat_{seat_idx} (Hunter) tries to shoot but no valid targets alive.")
                    else:
                        print(f"  seat_{seat_idx} (Hunter) chooses not to open fire.")

            # Clear last_night_deaths
            self.last_night_deaths.clear()

        print(f"\nDAY with PK-speech tie resolution, day_count={self.day_count}, badge_holder={self.badge_holder}")

        alive_list = [i for i in range(self.num_seats) if self.alive[i]]
        votes1 = {}
        # Round1
        for i in alive_list:
            # If revealed idiot => skip
            if self.role_assignment[i] == "idiot" and self.idiot_revealed[i]:
                print(f"  seat_{i} => cannot vote (revealed idiot)")
                continue

            pick = action_dict.get(f"seat_{i}", 0)
            if pick == 0:
                print(f"  seat_{i} => no vote (action=0)")
                continue
            seat_cand = (pick - 1) % self.num_seats
            if not self.alive[seat_cand]:
                print(f"  seat_{i} => no vote (target seat_{seat_cand} is dead)")
                continue
            w = 1.5 if i == self.badge_holder else 1.0
            votes1[seat_cand] = votes1.get(seat_cand, 0) + w
            print(f"  seat_{i} => seat_{seat_cand} (weight={w})")

        print(f"  Round1 votes => {dict(votes1)}")
        if not votes1:
            print("  Round1 => no votes => no elimination.")
            return

        top_val = max(votes1.values())
        top_cands = [s for s,v in votes1.items() if v == top_val]
        if len(top_cands) == 1:
            kill_target = top_cands[0]
            print(f"  Round1 => seat_{kill_target} singled out => ", end="")
            # If seat is an unrevealed idiot => reveal but not kill
            if self.role_assignment[kill_target] == "idiot" and not self.idiot_revealed[kill_target]:
                print(f"seat_{kill_target} (idiot) => REVEALED but not killed.")
                self.idiot_revealed[kill_target] = True
            else:
                print(f"seat_{kill_target} is eliminated.")
                self._kill_seat(kill_target, reason="day_vote", nightkill=False)
            return

        # tie => Round2
        print(f"  Round1 => tie => seats {top_cands} => PK speech => Round2")
        votes2 = {}
        tie_set = set(top_cands)
        for i in alive_list:
            if i in tie_set:
                continue
            if self.role_assignment[i] == "idiot" and self.idiot_revealed[i]:
                print(f"    seat_{i} => cannot vote (revealed idiot)")
                continue

            sub2 = random.randint(0, 1)
            if sub2 == 0:
                print(f"    seat_{i} => no vote (sub=0)")
                continue
            c2 = random.choice(list(tie_set))
            if not self.alive[c2]:
                print(f"    seat_{i} => no vote => tie target seat_{c2} dead??")
                continue
            if c2 == i:
                print(f"    seat_{i} => no vote => tried to vote self.")
                continue
            w2 = 1.5 if i == self.badge_holder else 1.0
            votes2[c2] = votes2.get(c2, 0) + w2
            print(f"    seat_{i} => seat_{c2} (weight={w2})")

        print(f"  Round2 votes => {dict(votes2)}")
        if not votes2:
            print("  Round2 => no valid votes => no elimination.")
            return

        mx2 = max(votes2.values())
        cand2 = [xx for xx,vv in votes2.items() if vv == mx2]
        if len(cand2) == 1:
            k2 = cand2[0]
            print(f"  Round2 => seat_{k2} singled out => ", end="")
            if self.role_assignment[k2] == "idiot" and not self.idiot_revealed[k2]:
                print(f"seat_{k2} (idiot) => REVEALED but not killed.")
                self.idiot_revealed[k2] = True
            else:
                print(f"seat_{k2} is eliminated.")
                self._kill_seat(k2, reason="day_vote", nightkill=False)
        else:
            print(f"  Round2 => tie again => no elimination => seats {cand2}")

    def _apply_kills(self, kill_list, nightkill=True):
        """Applies kills from a pending list. We do NOT print the cause; we just kill them."""
        if not kill_list:
            return
        print("Applying pending kills now:")
        for (reason, seat_idx) in kill_list:
            self._kill_seat(seat_idx, reason=reason, nightkill=nightkill)
        kill_list.clear()

    def _kill_seat(self, seat_idx, reason, nightkill=False):
        """
        Kills seat_idx for real, no cause displayed to logs.
        If seat_idx=Hunter & killed at night => final shot in morning.
        If seat_idx=Hunter & day-killed => immediate shot if not poison.
        """
        if seat_idx < 0 or seat_idx >= self.num_seats:
            return
        if not self.alive[seat_idx]:
            return

        rname = self.role_assignment[seat_idx]
        print(f"    seat_{seat_idx} ({rname}) => KILLED")
        self.alive[seat_idx] = False

        # If had the badge, discard
        if seat_idx == self.badge_holder:
            print(f"    Badge-holder seat_{seat_idx} is dying => discard badge.")
            self.badge_holder = -1

        if nightkill:
            # If night, store for morning announcement
            self.last_night_deaths.append({
                "seat_idx": seat_idx,
                "reason": reason
            })
        else:
            # If day kill => check hunter final shot
            if rname == "hunter" and reason not in ("poison", "witch_poison"):
                possible = [x for x in range(self.num_seats) if self.alive[x] and x != seat_idx]
                if possible:
                    shot_victim = random.choice(possible)
                    print(f"    Hunter seat_{seat_idx} => SHOOTS seat_{shot_victim}")
                    self._kill_seat(shot_victim, reason="hunter_shot_day", nightkill=False)
                else:
                    print("    No alive targets for Hunter final shot.")

    def _check_winner(self):
        wolves_alive = sum(
            self.alive[i] and self.role_assignment[i] == "werewolf"
            for i in range(self.num_seats)
        )
        vill_alive = sum(
            self.alive[i] and self.role_assignment[i] != "werewolf"
            for i in range(self.num_seats)
        )
        if wolves_alive == 0:
            self.episode_terminated = True
        elif wolves_alive >= vill_alive:
            self.episode_terminated = True

    def _build_rewards(self):
        rew = {}
        if not (self.episode_terminated or self.episode_truncated):
            for a in self.agents:
                rew[a] = 0.0
            return rew

        w_alive = sum(
            self.alive[i] and self.role_assignment[i] == "werewolf"
            for i in range(self.num_seats)
        )
        if w_alive == 0:
            # village wins
            for i,a in enumerate(self.agents):
                if self.role_assignment[i] == "werewolf":
                    rew[a] = -1.0
                else:
                    rew[a] = +1.0
        else:
            # wolves >= others => wolves side wins
            for i,a in enumerate(self.agents):
                if self.role_assignment[i] == "werewolf":
                    rew[a] = +1.0
                else:
                    rew[a] = -1.0
        return rew

    def _get_obs(self, seat_idx):
        alive_flag = 1 if self.alive[seat_idx] else 0
        ph = 0
        if self.phase == "ELECTION":
            ph = 1
        elif self.phase == "DAY":
            ph = 2

        role_name = self.role_assignment[seat_idx]
        rid = role_to_int(role_name)
        wh = 1 if self.witch_heal_used else 0
        wp = 1 if self.witch_poison_used else 0
        bh = self.badge_holder if self.badge_holder >= 0 else 255
        iw = 1 if role_name == "werewolf" else 0

        return np.array([
            seat_idx,
            alive_flag,
            self.day_count,
            ph,
            rid,
            wh,
            wp,
            bh,
            iw,
            0
        ], dtype=np.int32)
