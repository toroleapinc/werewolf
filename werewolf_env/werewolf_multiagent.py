"""
Name of script: werewolf_multiagent.py

A seat-based multi-agent Werewolf environment with:
 - NIGHT -> ELECTION -> DAY -> (repeat)
 - No seat picks a dead seat behind the scenes; we assume main.py ensures that.
 - Badge logic:
    - If the badge-holder is killed at night: pass the badge to a random other alive seat.
    - If the badge-holder is killed during the day: the badge is discarded.
    - If the seat is an Idiot singled out the first time in day voting => do NOT kill => 
        if they had the badge, they keep it (since they remain alive).
    - Second time singled out => truly day-killed => badge is discarded if they had it.
 - Wolf => all 0 => no kill, else majority among non-zero. tie => random
 - Seer => checks seat if alive
 - Witch => 0,1,2 if potions not used
 - If day_count>10 => truncated
 - If wolves=0 => village wins
 - If vill=0 or gods=0 => werewolves win
 - If wolves>=(vill+gods) => werewolves early-win
 - No skipping or reassigning picks is done in the environment.

We implement the new badge logic in _kill_seat for night/day kills, 
and preserve the badge if the seat is the idiot singled out first time (not actually killed).
"""

import random
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box

ALL_ROLES = ["werewolf", "villager", "seer", "witch", "hunter", "idiot"]

def role_to_int(role_name):
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

        # Witch potions usage
        self.witch_heal_used = False
        self.witch_poison_used = False

        # Badge holder seat index or -1 if none
        self.badge_holder = -1
        self.election_done = False

        self.seer_knowledge = {}
        self.pending_kills_first_night = []
        self.pending_post_election_kills = []
        self.last_night_deaths = []

        # Track how many times the idiot seat has been singled out in day votes
        self.idiot_singled_out_count = [0]*NUM_SEATS

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
        self.pending_kills_first_night.clear()
        self.pending_post_election_kills.clear()
        self.last_night_deaths.clear()
        self.idiot_singled_out_count = [0]*NUM_SEATS

        for i, role in enumerate(self.role_assignment):
            if role == "seer":
                self.seer_knowledge[i] = {}

        obs_dict = {}
        for i, agent_id in enumerate(self.agents):
            obs_dict[agent_id] = self._get_obs(i)
        return obs_dict, {}

    def step(self, action_dict):
        # If ended or truncated => return final
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
                self.phase = "DAY"

        elif self.phase == "ELECTION":
            self._election_phase(action_dict)
            self.election_done = True
            self.phase = "DAY"

        elif self.phase == "DAY":
            # Apply leftover kills
            self._apply_kills(self.pending_kills_first_night, nightkill=True)
            self._apply_kills(self.pending_post_election_kills, nightkill=True)

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

    # -------------- NIGHT PHASE ---------------
    def _night_phase(self, action_dict):
        print(f"\nNight: Wolf kills => majority. day_count={self.day_count}")

        wolf_votes = []
        for i in range(self.num_seats):
            if self.alive[i] and self.role_assignment[i] == "werewolf":
                pick = action_dict.get(f"seat_{i}", 0)
                seat_cand = None if (pick == 0) else ((pick-1) % self.num_seats)
                if seat_cand is not None:
                    print(f"  Wolf seat_{i} => seat_{seat_cand}")
                wolf_votes.append(seat_cand)

        if all(x is None for x in wolf_votes):
            print("Night: all wolves pick no kill => no one is killed.")
            kill_target = None
        else:
            valid = [v for v in wolf_votes if v is not None]
            freq = {}
            for v in valid:
                freq[v] = freq.get(v, 0)+1
            if freq:
                mx = max(freq.values())
                top_cands = [k for k,v in freq.items() if v==mx]
                kill_target = random.choice(top_cands) if len(top_cands)>1 else top_cands[0]
                print(f"Night: final kill target => seat_{kill_target}")
            else:
                kill_target = None

        # Seer checks
        for i,ro in enumerate(self.role_assignment):
            if self.alive[i] and ro=="seer":
                pick = action_dict.get(f"seat_{i}",0)
                if pick>0:
                    seat_cand = (pick-1)%self.num_seats
                    known = self.seer_knowledge.setdefault(i,{})
                    if seat_cand not in known:
                        iswolf = (self.role_assignment[seat_cand]=="werewolf")
                        print(f"  Seer seat_{i} => checks seat_{seat_cand} => Wolf? {iswolf}")
                        known[seat_cand]= iswolf

        # Witch
        if kill_target is not None:
            wseat=None
            for si, rr in enumerate(self.role_assignment):
                if rr=="witch" and self.alive[si]:
                    wseat= si
                    break
            if wseat is not None:
                wact= action_dict.get(f"seat_{wseat}",0)
                if wact==0:
                    print("  Witch => does NOTHING")
                elif wact==1:
                    if not self.witch_heal_used:
                        print(f"  Witch seat_{wseat} => HEAL seat_{kill_target}, kill canceled")
                        kill_target=None
                        self.witch_heal_used=True
                elif wact==2:
                    if not self.witch_poison_used:
                        possible = [
                            x for x in range(self.num_seats)
                            if x!=wseat and self.alive[x] and x!=kill_target
                        ]
                        if possible:
                            victim= random.choice(possible)
                            print(f"  Witch seat_{wseat} => POISON seat_{victim}")
                            self.witch_poison_used=True
                            if self.day_count==0 and not self.election_done:
                                print(f"Night: first-night poison is pending => seat_{victim}")
                                self.pending_kills_first_night.append(("poison",victim))
                            else:
                                if self.election_done:
                                    self.pending_post_election_kills.append(("poison",victim))
                                else:
                                    self.last_night_deaths.append({
                                        "seat_idx": victim,
                                        "reason":"poison"
                                    })
                                    self._kill_seat(victim,"poison",nightkill=True)

        if kill_target is not None:
            if self.day_count==0 and not self.election_done:
                print(f"Night: first-night kill is pending => seat_{kill_target}")
                self.pending_kills_first_night.append(("wolf",kill_target))
            else:
                if self.election_done:
                    print(f"Night: kill target seat_{kill_target} is pending post-election")
                    self.pending_post_election_kills.append(("wolf",kill_target))
                else:
                    print(f"Night: seat_{kill_target} is chosen for elimination")
                    self.last_night_deaths.append({
                        "seat_idx":kill_target,
                        "reason":"wolf"
                    })
                    self._kill_seat(kill_target, "wolf", nightkill=True)

    # -------------- ELECTION PHASE ---------------
    def _election_phase(self, action_dict):
        print("\nELECTION PHASE with PK-speech tie resolution")
        runner_set=set()
        quitter=set()
        for i in range(self.num_seats):
            if self.alive[i]:
                a= action_dict.get(f"seat_{i}",0)
                if a==1:
                    runner_set.add(i)
                elif a==2:
                    quitter.add(i)

        r1= sorted(runner_set|quitter)
        q1= sorted(quitter)
        final_r= sorted(runner_set)
        print(f"  Round1 runners => {r1}")
        print(f"  Round1 quitter => {q1}")
        print(f"  Round1 final runners => {final_r}")

        if not final_r:
            print("  Round1 => no final_runners => no badge assigned.")
            return

        alive_set= set(i for i in range(self.num_seats) if self.alive[i])
        votes={}
        for i in alive_set:
            if i not in final_r and i not in quitter:
                sub= random.randint(0,1)
                if sub==0:
                    print(f"    seat_{i} => no vote (sub=0)")
                    continue
                c= random.choice(final_r)
                print(f"    seat_{i} => votes runner seat_{c} (sub=1)")
                votes[c]= votes.get(c,0)+1

        print(f"  Round1 votes => {dict(votes)}")
        if not votes:
            print("  Round1 => no valid votes => tie among all => Round2")
            tie_seats= final_r[:]
            print(f"  All tie seats => {tie_seats}")
            round2={}
            tset=set(tie_seats)
            for j in alive_set:
                if j not in tset:
                    s2= random.randint(0,1)
                    if s2==0:
                        print(f"    seat_{j} => no vote (sub=0)")
                        continue
                    c2= random.choice(tie_seats)
                    print(f"    seat_{j} => votes runner seat_{c2} (sub=1)")
                    round2[c2]= round2.get(c2,0)+1
            print(f"  Round2 votes => {dict(round2)}")
            if not round2:
                print("  Round2 => no valid votes => no badge assigned.")
                return
            tv= max(round2.values())
            tcs= [k for k,v in round2.items() if v==tv]
            if len(tcs)==1:
                w2= tcs[0]
                print(f"  Round2 => seat_{w2} singled out => badge assigned.")
                self.badge_holder= w2
            else:
                print(f"  Round2 => tie => no badge => seats {tcs}")
            return

        best_val= max(votes.values())
        top_c=[k for k,v in votes.items() if v==best_val]
        if len(top_c)==1:
            w= top_c[0]
            print(f"  Round1 => seat_{w} singled out => badge assigned.")
            self.badge_holder=w
            return

        print(f"  Round1 => tie => seats {top_c} => PK speech => Round2")
        round2={}
        tset= set(top_c)
        for i in alive_set:
            if i not in tset and i not in final_r:
                sub2= random.randint(0,1)
                if sub2==0:
                    print(f"    seat_{i} => no vote (sub=0)")
                    continue
                c2= random.choice(top_c)
                print(f"    seat_{i} => votes seat_{c2} in Round2 tie")
                round2[c2]= round2.get(c2,0)+1

        print(f"  Round2 votes => {dict(round2)}")
        if not round2:
            print("  Round2 => no valid votes => no badge assigned.")
            return
        mx= max(round2.values())
        candz=[xx for xx,vv in round2.items() if vv==mx]
        if len(candz)==1:
            w2=candz[0]
            print(f"  Round2 => seat_{w2} singled out => badge assigned.")
            self.badge_holder=w2
        else:
            print(f"  Round2 => tie => no badge => seats {candz}")

    # -------------- DAY PHASE ---------------
    def _day_phase_two_round(self, action_dict):
        # Announce kills from last night
        if self.last_night_deaths:
            print("\nAnnouncing kills from last night:")
            for entry in self.last_night_deaths:
                sidx= entry["seat_idx"]
                print(f"  seat_{sidx} died last night.")
                if self.role_assignment[sidx]=="hunter":
                    sub= random.randint(0,1)
                    if sub==1:
                        possible= [x for x in range(self.num_seats) if self.alive[x] and x!=sidx]
                        if possible:
                            victim= random.choice(possible)
                            print(f"  seat_{sidx} (Hunter) opens fire on seat_{victim}!")
                            self._kill_seat(victim, "hunter_shot_morning",nightkill=False)
                        else:
                            print(f"  seat_{sidx} (Hunter) tries to shoot but no valid targets alive.")
            self.last_night_deaths.clear()

        print(f"\nDAY with PK-speech tie resolution, day_count={self.day_count}, badge_holder={self.badge_holder}")
        alive_list= [i for i in range(self.num_seats) if self.alive[i]]
        votes1={}

        # Round1
        for i in alive_list:
            # If seat is idiot & singled_out_count>=1 => skip
            if self.role_assignment[i]=="idiot" and self.idiot_singled_out_count[i]>=1:
                print(f"  seat_{i} => cannot vote (revealed idiot)")
                continue

            pick= action_dict.get(f"seat_{i}",0)
            if pick==0:
                print(f"  seat_{i} => no vote (action=0)")
                continue

            seat_cand= (pick-1)%self.num_seats
            w= 1.5 if i==self.badge_holder else 1.0
            votes1[seat_cand]= votes1.get(seat_cand,0)+ w
            print(f"  seat_{i} => seat_{seat_cand} (weight={w})")

        print(f"  Round1 votes => {dict(votes1)}")
        if not votes1:
            print("  Round1 => no votes => no elimination.")
            return

        top_val= max(votes1.values())
        top_cands= [xx for xx,vv in votes1.items() if vv==top_val]
        if len(top_cands)==1:
            kill_target= top_cands[0]
            print(f"  Round1 => seat_{kill_target} singled out => ",end="")
            # if seat is idiot singled out first time => no kill => keep badge
            if self.role_assignment[kill_target]=="idiot":
                if self.idiot_singled_out_count[kill_target]==0:
                    print(f"seat_{kill_target} (idiot) => REVEALED but not killed. Keeps the badge if had it.")
                    self.idiot_singled_out_count[kill_target]=1
                else:
                    print(f"seat_{kill_target} (idiot) => singled out again => now killed.")
                    self._kill_seat(kill_target, "day_vote", nightkill=False)
            else:
                print(f"seat_{kill_target} is eliminated.")
                self._kill_seat(kill_target, "day_vote", nightkill=False)
            return

        print(f"  Round1 => tie => seats {top_cands} => PK speech => Round2")
        votes2={}
        tie_set= set(top_cands)
        for i in alive_list:
            if i in tie_set:
                continue
            if self.role_assignment[i]=="idiot" and self.idiot_singled_out_count[i]>=1:
                print(f"    seat_{i} => cannot vote (revealed idiot)")
                continue

            sub2= random.randint(0,1)
            if sub2==0:
                print(f"    seat_{i} => no vote (sub=0)")
                continue
            c2= random.choice(list(tie_set))
            w2= 1.5 if i==self.badge_holder else 1.0
            votes2[c2]= votes2.get(c2,0)+ w2
            print(f"    seat_{i} => seat_{c2} (weight={w2})")

        print(f"  Round2 votes => {dict(votes2)}")
        if not votes2:
            print("  Round2 => no valid votes => no elimination.")
            return
        mx2= max(votes2.values())
        cand2= [xx for xx,vv in votes2.items() if vv==mx2]
        if len(cand2)==1:
            k2= cand2[0]
            print(f"  Round2 => seat_{k2} singled out => ",end="")
            if self.role_assignment[k2]=="idiot":
                if self.idiot_singled_out_count[k2]==0:
                    print(f"seat_{k2} (idiot) => REVEALED but not killed. Keeps the badge if had it.")
                    self.idiot_singled_out_count[k2]=1
                else:
                    print(f"seat_{k2} (idiot) => singled out again => now killed.")
                    self._kill_seat(k2,"day_vote",nightkill=False)
            else:
                print(f"seat_{k2} is eliminated.")
                self._kill_seat(k2,"day_vote",nightkill=False)
        else:
            print(f"  Round2 => tie again => no elimination => seats {cand2}")

    def _apply_kills(self, kill_list, nightkill=True):
        if not kill_list:
            return
        print("Applying pending kills now:")
        for (reason, seat_idx) in kill_list:
            self._kill_seat(seat_idx, reason, nightkill)
        kill_list.clear()

    def _kill_seat(self, seat_idx, reason, nightkill=False):
        if seat_idx<0 or seat_idx>=self.num_seats:
            return
        if not self.alive[seat_idx]:
            return

        rname= self.role_assignment[seat_idx]
        print(f"    seat_{seat_idx} ({rname}) => KILLED")
        self.alive[seat_idx]= False

        # Implement updated badge logic
        if seat_idx==self.badge_holder:
            if nightkill:
                # If night kill => pass badge to random other alive seat
                possible = [s for s in range(self.num_seats) if self.alive[s] and s!=seat_idx]
                if possible:
                    new_holder = random.choice(possible)
                    print(f"    Badge-holder seat_{seat_idx} was killed at night => pass badge to seat_{new_holder}.")
                    self.badge_holder= new_holder
                else:
                    print(f"    Badge-holder seat_{seat_idx} was killed at night => no one else alive => no badge.")
                    self.badge_holder= -1
            else:
                # day kill => badge is discarded
                print(f"    Badge-holder seat_{seat_idx} is dying by day => discard badge.")
                self.badge_holder= -1

        # if day-kill => check if seat is hunter => final shot (unless poison)
        if not nightkill:
            if rname=="hunter" and reason not in ("poison","witch_poison"):
                possible=[x for x in range(self.num_seats) if self.alive[x] and x!=seat_idx]
                if possible:
                    victim= random.choice(possible)
                    print(f"    Hunter seat_{seat_idx} => SHOOTS seat_{victim}")
                    self._kill_seat(victim, "hunter_shot_day",nightkill=False)
                else:
                    print("    No alive targets for Hunter final shot.")

    def _check_winner(self):
        # werewolf early-win if:
        #   - wolves=0 => village wins
        #   - all villagers dead => werewolves win
        #   - all gods (seer,witch,hunter,idiot) dead => werewolves win
        #   - wolves>=(villagers+gods) => werewolves early-win
        w=0
        vill=0
        gods=0
        for i in range(self.num_seats):
            if not self.alive[i]:
                continue
            ro=self.role_assignment[i]
            if ro=="werewolf":
                w+=1
            elif ro=="villager":
                vill+=1
            else:
                gods+=1

        if w==0:
            self.episode_terminated=True
            return
        if vill==0 or gods==0:
            self.episode_terminated=True
            return
        if w>=(vill+gods):
            self.episode_terminated=True
            return

    def _build_rewards(self):
        if not (self.episode_terminated or self.episode_truncated):
            return {a:0.0 for a in self.agents}

        w=0
        vill=0
        gods=0
        for i in range(self.num_seats):
            if not self.alive[i]:
                continue
            ro=self.role_assignment[i]
            if ro=="werewolf":
                w+=1
            elif ro=="villager":
                vill+=1
            else:
                gods+=1

        rew={}
        if w==0:
            for i,a in enumerate(self.agents):
                if self.role_assignment[i]=="werewolf":
                    rew[a]= -1.0
                else:
                    rew[a]= +1.0
        else:
            for i,a in enumerate(self.agents):
                if self.role_assignment[i]=="werewolf":
                    rew[a]= +1.0
                else:
                    rew[a]= -1.0
        return rew

    def _get_obs(self, seat_idx):
        alive_flag= 1 if self.alive[seat_idx] else 0
        ph=0
        if self.phase=="ELECTION":
            ph=1
        elif self.phase=="DAY":
            ph=2

        role_name= self.role_assignment[seat_idx]
        rid= role_to_int(role_name)
        wh= 1 if self.witch_heal_used else 0
        wp= 1 if self.witch_poison_used else 0
        bh= self.badge_holder if self.badge_holder>=0 else 255
        iw= 1 if role_name=="werewolf" else 0

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
