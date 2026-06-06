import traci
import numpy as np
from gymnasium import spaces
import xml.etree.ElementTree as ET
from pettingzoo import ParallelEnv


class SumoMultiAgentEnv(ParallelEnv):
    """
    Multi-agent SUMO env (headless by default).

    Observation (per agent):
      one-hot(current_edge) + one-hot(possible_next_edges) + one-hot(goal_edge)
      + density_per_next_edge(max_branching values)
      + [speed_norm, pos_on_edge, dist_to_goal_norm, density_cur_edge,
         halted_cur_edge, at_decision_point]
      -> size 3 * num_edges + max_branching + 6

    v10 reward function redesign:
      - Goal reward increased to 200 (dominates any shaping accumulation)
      - Shaping rewards reduced (speed_coef /5, distance_factor /2)
      - Stronger penalties for pathological behavior (halt, backtrack, invalid)
      - Explicit cooperation: penalty for blocking other agents behind
      - Cleaner decomposition: each reward component as a separate method
    """
    metadata = {"render_modes": ["human"], "name": "sumo_traffic_v0"}

    NUM_CONTINUOUS = 6  # speed_norm, pos_on_edge, dist_to_goal, density, halted, at_dp

    def __init__(self, sumo_config, rou_file, use_gui=False, max_steps=1000,
                 reward_params=None, min_turn_buffer_m: float = 5.0,
                 decision_zone_m: float = 30.0):
        super().__init__()
        self.sumo_config = sumo_config
        self.rou_file = rou_file
        self.use_gui = bool(use_gui)
        self.max_steps = int(max_steps)
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.step_count = 0
        self.min_turn_buffer_m = float(min_turn_buffer_m)
        self.decision_zone_m = float(decision_zone_m)
        self._committed_edge = {}

        # v10 reward parameters: balanced shaping with dominant goal reward.
        # Design principle: sum of all shaping rewards over an episode must be
        # smaller than a single goal_reward, so the policy cannot exploit
        # shaping to outperform goal-reaching strategies.
        self.reward_params = reward_params or {
            # ── Core objective ──
            "goal_reward": 200.0,            # dominant positive signal
            "time_penalty": -0.02,           # urgency; 2200 steps -> -44 floor

            # ── Progress shaping (potential-based, policy-invariant) ──
            "distance_factor": 0.005,        # halved: max ~5 pts/episode
            "edge_switch_bonus": 0.05,       # bonus for significant progress
            "progress_big_drop": 70.0,       # threshold for progress bonus

            # ── Efficiency incentive ──
            "speed_coef": 0.001,             # 5x smaller: max ~2 pts per ep

            # ── Anti-exploit penalties ──
            "halt_penalty": -0.1,            # 5x stronger
            "backtrack_penalty": -0.5,       # 10x stronger
            "backtrack_margin": 20.0,        # detects backtrack earlier
            "invalid_action_penalty": -0.2,  # 4x stronger

            # ── Cooperation ──
            "congestion_route_penalty": -0.1,  # implicit: avoid congested routes
            "congestion_penalty": 0.001,       # global traffic queue pressure
            "blocking_penalty": -0.05,         # explicit: blocking others behind
            "blocking_threshold": 0.1,         # speed below this = considered blocking
            "teleport_penalty": -5.0,          # severe cooperation failure
        }

        self.flow_info = self._load_flow_info()
        self.goal_positions = {}
        self.previous_distances = {}
        self.initial_distances = {}
        self.last_edge = {}
        self.active_agents = set()
        self.possible_agents = []

        self.all_edges = []
        self.edge_index_map = {}
        self.num_edges = 0
        self.max_branching = 1

        self._start_sumo_temp()
        self.all_edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
        self.edge_index_map = {edge: idx for idx, edge in enumerate(self.all_edges)}
        self.num_edges = len(self.all_edges)
        self.max_branching = self._compute_max_branching()
        traci.close()
        self._define_spaces()

    def kpis(self):
        edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
        if not edges:
            return {"avg_speed": 0.0, "halted": 0, "arrived": 0, "queue": 0, "teleports": 0,
                    "speed_norm": 0.0, "halts_per_vehstep": 0.0, "arrivals_per_100vehsteps": 0.0,
                    "mean_waiting_time": 0.0, "mean_travel_time": 0.0, "throughput": 0,
                    "goals_reached": 0}
        speeds, halts, waits, travels = [], 0, [], []
        vehs = set(traci.vehicle.getIDList())
        for e in edges:
            try:
                speeds.append(traci.edge.getLastStepMeanSpeed(e))
                halts += traci.edge.getLastStepHaltingNumber(e)
            except: pass
        for v in vehs:
            try: waits.append(traci.vehicle.getAccumulatedWaitingTime(v))
            except: pass
            try: travels.append(traci.simulation.getTime() - traci.vehicle.getDeparture(v))
            except: pass
        arrived = traci.simulation.getArrivedNumber()
        teleports = traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber()
        veh_count = len(vehs)
        goals_reached = int(getattr(self, "goals_reached", 0))
        return {
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "halted": int(halts), "arrived": int(arrived), "queue": int(halts),
            "teleports": int(teleports),
            "speed_norm": float(np.mean(speeds)) / 13.9 if speeds else 0.0,
            "halts_per_vehstep": halts / max(1, veh_count),
            "arrivals_per_100vehsteps": 100.0 * arrived / max(1, veh_count * self.step_count) if self.step_count > 0 else 0.0,
            "mean_waiting_time": float(np.mean(waits)) if waits else 0.0,
            "mean_travel_time": float(np.mean(travels)) if travels else 0.0,
            "throughput": int(arrived), "goals_reached": goals_reached,
        }

    def _define_spaces(self):
        obs_size = 3 * self.num_edges + self.max_branching + self.NUM_CONTINUOUS
        act_size = self.max_branching
        self._default_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self._default_action_space = spaces.Discrete(act_size)
        self.action_spaces = {}
        self.observation_spaces = {}

    def _is_at_decision_point(self, veh_id):
        try:
            lane = traci.vehicle.getLaneID(veh_id)
            if lane.startswith(':'): return False
            m = self._meters_to_lane_end(veh_id)
            return m <= self.decision_zone_m
        except: return False

    def _compute_max_branching(self):
        mx = 1
        for lane in traci.lane.getIDList():
            if lane.startswith(':'): continue
            try:
                links = traci.lane.getLinks(lane)
                mx = max(mx, len({l[0].split("_")[0] for l in links if l and l[0]}))
            except: pass
        return max(1, mx)

    def _next_edges_from_lane(self, lane_id):
        try:
            return list({l[0].split("_")[0] for l in traci.lane.getLinks(lane_id) if l and l[0]})
        except: return []

    def _get_next_edges_from_vehicle(self, veh_id):
        try:
            lane = traci.vehicle.getLaneID(veh_id)
            if lane.startswith(':'): return []
            return self._next_edges_from_lane(lane)
        except: return []

    def _meters_to_lane_end(self, veh_id):
        try:
            lane = traci.vehicle.getLaneID(veh_id)
            if lane.startswith(':'): return 0.0
            lane_len = traci.lane.getLength(lane)
            pos = traci.vehicle.getLanePosition(veh_id)
            return max(0.0, lane_len - pos)
        except: return 0.0

    def _get_edge_density(self, edge_id):
        """Get normalized vehicle density on an edge (0-1)."""
        try:
            if edge_id in self.edge_index_map:
                return min(1.0, traci.edge.getLastStepVehicleNumber(edge_id) / 5.0)
        except: pass
        return 0.0

    def _get_agents_behind(self, aid):
        """
        Count agents on the same edge, behind `aid`, that are halted.
        Used for the blocking_penalty (explicit cooperation term).
        """
        try:
            edge = traci.vehicle.getRoadID(aid)
            if edge.startswith(':') or edge not in self.edge_index_map:
                return 0
            current_ids = set(traci.vehicle.getIDList())
            my_pos = traci.vehicle.getLanePosition(aid)
            blocked = 0
            for other in self.active_agents:
                if other == aid or other not in current_ids:
                    continue
                try:
                    if traci.vehicle.getRoadID(other) != edge:
                        continue
                    other_pos = traci.vehicle.getLanePosition(other)
                    # Behind = smaller lane position on same edge
                    if other_pos < my_pos:
                        other_speed = traci.vehicle.getSpeed(other)
                        if other_speed < self.reward_params.get("blocking_threshold", 0.1):
                            blocked += 1
                except: continue
            return blocked
        except: return 0

    def _load_flow_info(self):
        root = ET.parse(self.rou_file).getroot()
        info = {}
        for flow in root.findall("flow"):
            fid = flow.get("id")
            info[fid] = {"source": flow.get("from"), "target": flow.get("to"),
                         "begin": float(flow.get("begin", 0)), "end": float(flow.get("end", self.max_steps))}
        return info

    def _start_sumo_temp(self):
        traci.start([self.sumo_binary, "-c", self.sumo_config, "--no-step-log", "true"])
        try: traci.simulationStep()
        except: pass

    def _start_sumo(self):
        try:
            traci.start([self.sumo_binary, "-c", self.sumo_config,
                         "--no-step-log", "true", "--no-warnings", "true"])
            self.step_count = 0; return True
        except traci.TraCIException as e:
            print(f"Error starting SUMO: {e}"); return False

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        try: traci.close()
        except: pass
        if not self._start_sumo(): raise RuntimeError("Failed to start SUMO")
        self.step_count = 0; self.active_agents = set(); self.goal_positions = {}
        self.previous_distances = {}; self.initial_distances = {}; self.last_edge = {}
        self.goals_reached = 0; self._agents_that_reached_goal = set(); self._committed_edge = {}
        self._last_action_edge = {}  # track which edge each agent chose last
        # Reset per-episode agent registries. Without this, possible_agents /
        # action_spaces / observation_spaces grow every episode, which slows the
        # per-step termination/truncation loops and leaks memory over a long run.
        self.possible_agents = []
        self.action_spaces = {}
        self.observation_spaces = {}
        traci.simulationStep(); self._update_agents()
        obs = self._get_observations(); self.agents = list(self.active_agents)
        return obs, {aid: {} for aid in self.agents}

    def get_local_action_mask(self, agent_id):
        mask = np.zeros(self.max_branching, dtype=np.float32)
        nxt = self._get_next_edges_from_vehicle(agent_id)
        k = min(len(nxt), self.max_branching)
        if k > 0: mask[:k] = 1.0
        return mask

    def step(self, actions):
        self.invalid_action_penalties = {}
        self._last_action_edge = {}
        self._removed_goal_agents = set()

        for aid, a_idx in actions.items():
            if aid not in self.active_agents or aid not in traci.vehicle.getIDList(): continue
            try:
                cur = traci.vehicle.getRoadID(aid)
                if not cur or cur.startswith(':'): continue

                # Skip action for vehicles on goal edge — they will be removed after simStep
                ge = self.goal_positions.get(aid)
                if ge and cur == ge:
                    continue

                if not self._is_at_decision_point(aid): continue
                nxt = self._get_next_edges_from_vehicle(aid)
                if not nxt:
                    self.invalid_action_penalties[aid] = self.invalid_action_penalties.get(aid, 0) + 1; continue
                a = int(a_idx)
                if 0 <= a < len(nxt):
                    if self._meters_to_lane_end(aid) <= 0.0:
                        self.invalid_action_penalties[aid] = self.invalid_action_penalties.get(aid, 0) + 0.5; continue
                    try:
                        traci.vehicle.setRoute(aid, [cur, nxt[a]]); self._committed_edge[aid] = (cur, nxt[a])
                        self._last_action_edge[aid] = nxt[a]  # track chosen edge
                    except: self.invalid_action_penalties[aid] = self.invalid_action_penalties.get(aid, 0) + 1
                else:
                    self.invalid_action_penalties[aid] = self.invalid_action_penalties.get(aid, 0) + 1
            except: continue

        for aid in list(self._committed_edge.keys()):
            try:
                if aid in traci.vehicle.getIDList():
                    c = traci.vehicle.getRoadID(aid)
                    if c != self._committed_edge[aid][0] and not c.startswith(':'): del self._committed_edge[aid]
                else: del self._committed_edge[aid]
            except: pass

        traci.simulationStep()
        try: arrived = set(traci.simulation.getArrivedIDList())
        except: arrived = set()
        self._arrived_this_step = arrived
        self._goal_arrivals_this_step = set()
        for vid in arrived:
            if vid in self.active_agents and vid in self.goal_positions and vid not in self._agents_that_reached_goal:
                self._goal_arrivals_this_step.add(vid); self._agents_that_reached_goal.add(vid); self.goals_reached += 1
        try: self._teleported_this_step = set(traci.simulation.getStartingTeleportIDList())
        except: self._teleported_this_step = set()
        self._new_teleports_this_step = len(self._teleported_this_step)

        # Phase: remove vehicles on goal edge AFTER simulationStep (avoids TraCI errors)
        for aid in list(self.active_agents):
            if aid in self._agents_that_reached_goal: continue
            if aid not in traci.vehicle.getIDList(): continue
            try:
                cur = traci.vehicle.getRoadID(aid)
                if cur.startswith(':'): continue
                ge = self.goal_positions.get(aid)
                if ge and cur == ge:
                    self._agents_that_reached_goal.add(aid)
                    self.goals_reached += 1
                    self._removed_goal_agents.add(aid)
                    try: traci.vehicle.remove(aid, reason=2)
                    except: pass
            except: continue

        self.step_count += 1; self._update_agents()
        obs = self._get_observations(); rewards = self._compute_rewards()
        terms = self._get_terminations(); truncs = self._get_truncations()
        return obs, rewards, terms, truncs, {aid: {} for aid in self.possible_agents}

    def _update_agents(self):
        current = set(traci.vehicle.getIDList())
        for vid in current:
            if vid not in self.active_agents:
                fid = vid.rsplit(".", 1)[0] if "." in vid else (vid.split("_")[0] if "_" in vid else None)
                if fid in self.flow_info:
                    self.goal_positions[vid] = self.flow_info[fid]["target"]
                    try:
                        ce = traci.vehicle.getRoadID(vid)
                        if ce.startswith(":"): r = traci.vehicle.getRoute(vid); ce = r[0] if r else ce
                        ro = traci.simulation.findRoute(ce, self.goal_positions[vid])
                        self.previous_distances[vid] = ro.length if ro.edges else 1000.0
                    except: self.previous_distances[vid] = 1000.0
                    self.initial_distances[vid] = self.previous_distances[vid]; self.last_edge[vid] = ce
                self.active_agents.add(vid)
                if vid not in self.possible_agents: self.possible_agents.append(vid)
                self.action_spaces[vid] = self._default_action_space
                self.observation_spaces[vid] = self._default_observation_space
        ga = getattr(self, "_goal_arrivals_this_step", set())
        rga = getattr(self, "_removed_goal_agents", set())
        self.active_agents = self.active_agents.intersection(current | ga | rga)
        self.agents = list(self.active_agents)

    def _get_observations(self):
        obs = {}
        NE = self.num_edges
        MB = self.max_branching
        cv = set(traci.vehicle.getIDList())
        removed_goals = getattr(self, "_removed_goal_agents", set())
        for aid in self.active_agents:
            # For removed goal agents, return a zero obs (they get terminal reward)
            if aid in removed_goals and aid not in cv:
                obs[aid] = np.zeros(3 * NE + MB + self.NUM_CONTINUOUS, dtype=np.float32)
                # Set goal edge one-hot so the obs is identifiable
                ge = self.goal_positions.get(aid)
                if ge and ge in self.edge_index_map:
                    obs[aid][2 * NE + self.edge_index_map[ge]] = 1.0
                continue
            if aid not in cv:
                continue
            try:
                cur = traci.vehicle.getRoadID(aid)
                vec = np.zeros(3 * NE + MB + self.NUM_CONTINUOUS, dtype=np.float32)
                if cur.startswith(':'):
                    r = traci.vehicle.getRoute(aid); ri = traci.vehicle.getRouteIndex(aid)
                    if ri > 0: cur = r[ri - 1]
                    nxt = [r[ri + 1]] if r and ri < len(r) - 1 else []
                else:
                    nxt = []
                    try:
                        lid = traci.vehicle.getLaneID(aid)
                        if lid and not lid.startswith(':'): nxt = self._next_edges_from_lane(lid)
                        if not nxt:
                            r = traci.vehicle.getRoute(aid); ri = traci.vehicle.getRouteIndex(aid)
                            if r and ri < len(r) - 1: nxt = [r[ri + 1]]
                    except: pass

                # Block 1: current edge one-hot
                if cur in self.edge_index_map: vec[self.edge_index_map[cur]] = 1.0
                # Block 2: next edges one-hot
                for e in nxt:
                    if e in self.edge_index_map: vec[NE + self.edge_index_map[e]] = 1.0
                # Block 3: goal edge one-hot
                ge = self.goal_positions.get(aid)
                if ge and ge in self.edge_index_map: vec[2 * NE + self.edge_index_map[ge]] = 1.0

                # Block 4: density of each candidate next edge (NEW - congestion awareness)
                for i, e in enumerate(nxt[:MB]):
                    vec[3 * NE + i] = self._get_edge_density(e)

                # Block 5: continuous features
                idx = 3 * NE + MB
                try: vec[idx] = min(1.0, traci.vehicle.getSpeed(aid) / 13.9)
                except: pass
                idx += 1
                try:
                    lid = traci.vehicle.getLaneID(aid)
                    if lid and not lid.startswith(':'): vec[idx] = min(1.0, traci.vehicle.getLanePosition(aid) / max(1.0, traci.lane.getLength(lid)))
                except: pass
                idx += 1
                id_ = self.initial_distances.get(aid, 1000.0); cd = self.previous_distances.get(aid, 1000.0)
                if id_ > 0: vec[idx] = min(1.0, max(0.0, cd / id_))
                idx += 1
                try:
                    if cur in self.edge_index_map: vec[idx] = min(1.0, traci.edge.getLastStepVehicleNumber(cur) / 5.0)
                except: pass
                idx += 1
                try:
                    if cur in self.edge_index_map: vec[idx] = min(1.0, traci.edge.getLastStepHaltingNumber(cur) / 5.0)
                except: pass
                idx += 1
                if self._is_at_decision_point(aid): vec[idx] = 1.0
                obs[aid] = vec
            except: continue
        return obs

    # ── Reward components (v10 redesign) ──────────────────────────────────

    def _r_goal(self, aid, rewards):
        """Terminal goal reward. Dominates all shaping."""
        rp = self.reward_params
        removed_goals = getattr(self, "_removed_goal_agents", set())
        goal_arrivals = getattr(self, "_goal_arrivals_this_step", set())
        if aid in removed_goals or aid in goal_arrivals:
            rewards[aid] = rp.get("goal_reward", 200.0)
            return True
        return False

    def _r_time_urgency(self):
        """Constant per-step penalty: creates urgency to reach goal fast."""
        return self.reward_params.get("time_penalty", -0.02)

    def _r_speed_efficiency(self, aid):
        """Small bonus proportional to speed; stronger halt penalty."""
        rp = self.reward_params
        try:
            v = traci.vehicle.getSpeed(aid)
            if v < 0.1:
                return rp.get("halt_penalty", -0.1)
            return rp.get("speed_coef", 0.001) * (v / 13.9)
        except:
            return 0.0

    def _r_progress_shaping(self, aid, current_edge):
        """
        Potential-based distance shaping (Ng et al. 1999).
        Reward proportional to decrease in distance-to-goal;
        large bonus/penalty for big jumps (edge switches) and backtracking.
        Policy-invariant under gamma discount.
        """
        rp = self.reward_params
        gamma = 0.99
        goal_edge = self.goal_positions.get(aid)
        if not goal_edge or current_edge.startswith(':'):
            return 0.0, None

        try:
            ro = traci.simulation.findRoute(current_edge, goal_edge)
            cd = ro.length if ro.edges else 1000.0
        except:
            cd = 1000.0

        prev = self.previous_distances.get(aid, 1000.0)
        r = rp.get("distance_factor", 0.005) * (prev - gamma * cd)

        if cd > prev + rp.get("backtrack_margin", 20.0):
            r += rp.get("backtrack_penalty", -0.5)
        elif prev - cd > rp.get("progress_big_drop", 70.0):
            r += rp.get("edge_switch_bonus", 0.05)

        return r, cd

    def _r_congestion_awareness(self, aid):
        """Penalty for entering highly congested edges (implicit cooperation)."""
        rp = self.reward_params
        chosen_edge = self._last_action_edge.get(aid)
        if not chosen_edge:
            return 0.0
        density = self._get_edge_density(chosen_edge)
        if density > 0.4:
            return rp.get("congestion_route_penalty", -0.1) * density
        return 0.0

    def _r_cooperation_blocking(self, aid):
        """
        Explicit cooperation term: penalty when this agent is halted and
        has other agents stopped behind it on the same edge.
        Internalizes the social cost of blocking others.
        """
        rp = self.reward_params
        try:
            my_speed = traci.vehicle.getSpeed(aid)
            if my_speed >= rp.get("blocking_threshold", 0.1):
                return 0.0
            n_blocked = self._get_agents_behind(aid)
            if n_blocked > 0:
                return rp.get("blocking_penalty", -0.05) * n_blocked
        except: pass
        return 0.0

    def _r_invalid_action(self, aid):
        """Penalty accumulated from invalid action attempts."""
        rp = self.reward_params
        if not hasattr(self, "invalid_action_penalties"):
            return 0.0
        cnt = min(1.0, float(self.invalid_action_penalties.get(aid, 0.0)))
        if cnt > 0:
            self.invalid_action_penalties[aid] = 0.0
            return cnt * rp.get("invalid_action_penalty", -0.2)
        return 0.0

    def _r_global_pressure(self, total_queue, alive):
        """Small global team-reward-like pressure from network queue length."""
        return -self.reward_params.get("congestion_penalty", 0.001) * (float(total_queue) / alive)

    def _r_teleport_tax(self, alive):
        """Severe penalty per teleport event (cooperation failure indicator)."""
        new_tp = float(getattr(self, "_new_teleports_this_step", 0))
        return self.reward_params.get("teleport_penalty", -5.0) * (new_tp / alive)

    # ── Main reward aggregator ────────────────────────────────────────────

    def _compute_rewards(self):
        """
        Aggregates reward components per agent.

        For agents that reached goal: only goal_reward is applied (terminal).
        For active agents: full decomposition with individual + cooperation.
        """
        rewards = {}

        # Compute global signals once
        total_queue = 0
        try:
            for e in traci.edge.getIDList():
                if not e.startswith(":"):
                    total_queue += traci.edge.getLastStepHaltingNumber(e)
        except: pass
        alive = max(1, len(self.active_agents))

        teleport_tax = self._r_teleport_tax(alive)
        global_pressure = self._r_global_pressure(total_queue, alive)

        # Phase 1: goal rewards (terminal, override all shaping)
        for aid in list(self.active_agents):
            self._r_goal(aid, rewards)

        # Phase 2: active agents accumulate shaping + penalties
        current_vehicles = set(traci.vehicle.getIDList())
        for aid in self.active_agents:
            if aid in rewards:        # already has goal reward
                continue
            if aid not in current_vehicles:
                rewards[aid] = 0.0
                continue

            r = 0.0
            r += self._r_time_urgency()
            r += self._r_speed_efficiency(aid)

            # Determine current edge (handle internal junction lanes)
            ce = traci.vehicle.getRoadID(aid)
            if ce.startswith(':'):
                try:
                    rt = traci.vehicle.getRoute(aid)
                    ri = traci.vehicle.getRouteIndex(aid)
                    if rt and 0 <= ri < len(rt): ce = rt[ri]
                    elif rt: ce = rt[-1]
                except: pass

            r_prog, cd = self._r_progress_shaping(aid, ce)
            r += r_prog
            if cd is not None:
                self.previous_distances[aid] = cd

            r += self._r_congestion_awareness(aid)
            r += self._r_cooperation_blocking(aid)
            r += self._r_invalid_action(aid)
            r += global_pressure
            r += teleport_tax

            rewards[aid] = r

        return rewards

    def _get_terminations(self):
        t = {}; ga = getattr(self, "_goal_arrivals_this_step", set())
        rga = getattr(self, "_removed_goal_agents", set())
        cv = set(traci.vehicle.getIDList())
        for aid in self.possible_agents:
            if aid in ga or aid in rga: t[aid] = True
            elif aid in self.active_agents and aid in cv:
                t[aid] = False
            else: t[aid] = True
        return t

    def _get_truncations(self):
        return {aid: self.step_count >= self.max_steps for aid in self.possible_agents}

    def close(self):
        try: traci.close()
        except: pass

    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]

    def state(self):
        if not self.active_agents: return np.zeros(self.num_edges * 3, dtype=np.float32)
        ed = np.zeros(self.num_edges, dtype=np.float32); es = np.zeros(self.num_edges, dtype=np.float32)
        ec = np.zeros(self.num_edges, dtype=np.float32)
        for vid in traci.vehicle.getIDList():
            e = traci.vehicle.getRoadID(vid)
            if e in self.edge_index_map:
                i = self.edge_index_map[e]; ed[i] += 1; es[i] += traci.vehicle.getSpeed(vid); ec[i] += 1
        es = np.divide(es, ec, out=np.zeros_like(es), where=ec > 0)
        ap = np.zeros(self.num_edges, dtype=np.float32)
        for aid in self.active_agents:
            if aid in traci.vehicle.getIDList():
                e = traci.vehicle.getRoadID(aid)
                if e in self.edge_index_map: ap[self.edge_index_map[e]] = 1.0
        return np.concatenate([ed, es, ap]).astype(np.float32)