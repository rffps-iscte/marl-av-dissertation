import os, argparse, time, csv, numpy as np, random, torch, sys, datetime

from main.QMIX.replay_buffer import ReplayBuffer
from main.QMIX.qmix import QMIX
from main.environment import SumoMultiAgentEnv

SUMO_CFG = os.path.abspath("main/sumo-simulation/my_simulation_dense.sumocfg")
ROUTES = os.path.abspath("main/sumo-simulation/my_routes_dense.rou.xml")


class _Tee:
    """Mirror a stream (stdout/stderr) to the console *and* a log file."""
    def __init__(self, stream, fh):
        self._stream = stream; self._fh = fh
    def write(self, data):
        self._stream.write(data); self._fh.write(data)
    def flush(self):
        self._stream.flush(); self._fh.flush()
    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set.")

    p = argparse.ArgumentParser(description="SUMO QMIX (dense scenario, aligned with MAPPO, SMDP transitions)")
    p.add_argument("--sumo_config_path", default=SUMO_CFG)
    p.add_argument("--rou_file", default=ROUTES)
    p.add_argument("--use_gui", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results/qmix")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_agents", type=int, default=240)

    p.add_argument("--no_train", action="store_true",
                   help="Disable QMIX updates and force epsilon=1 -> random baseline.")

    # ── QMIX hyperparameters (stabilised values) ─────────────────────────
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hard_update_interval", type=int, default=0)
    p.add_argument("--mixer_hidden", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--update_freq", type=int, default=8)
    p.add_argument("--end_ep_updates", type=int, default=2)
    p.add_argument("--capacity", type=int, default=50_000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--reward_scale", type=float, default=1.0)

    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_anneal_episodes", type=int, default=60)

    # ── Reward parameters: IDENTICAL to MAPPO so the comparison is fair ─
    p.add_argument("--goal_reward", type=float, default=60.0)
    p.add_argument("--time_penalty", type=float, default=-0.02)
    p.add_argument("--distance_factor", type=float, default=0.01)
    p.add_argument("--edge_switch_bonus", type=float, default=0.1)
    p.add_argument("--progress_big_drop", type=float, default=70.0)
    p.add_argument("--speed_coef", type=float, default=0.002)
    p.add_argument("--halt_penalty", type=float, default=-0.1)
    p.add_argument("--backtrack_penalty", type=float, default=-0.5)
    p.add_argument("--backtrack_margin", type=float, default=20.0)
    p.add_argument("--invalid_action_pen", type=float, default=-0.2)
    p.add_argument("--congestion_penalty", type=float, default=0.001)
    p.add_argument("--teleport_penalty", type=float, default=-2.0)

    p.add_argument("--print_every", type=int, default=200)
    args = p.parse_args()
    train = not args.no_train
    print(f"[QMIX] mode = {'TRAIN' if train else 'BASELINE (no updates, eps=1.0)'}")

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Console logging ──────────────────────────────────────────────────
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = os.path.join(args.results_dir, f"console_{_ts}.log")
    _log_fh = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"[QMIX] console log -> {_log_path}")

    set_seed(args.seed)

    reward_params = {
        "time_penalty": args.time_penalty, "goal_reward": args.goal_reward,
        "distance_factor": args.distance_factor, "invalid_action_penalty": args.invalid_action_pen,
        "halt_penalty": args.halt_penalty, "speed_coef": args.speed_coef,
        "edge_switch_bonus": args.edge_switch_bonus, "congestion_penalty": args.congestion_penalty,
        "teleport_penalty": args.teleport_penalty, "backtrack_margin": args.backtrack_margin,
        "backtrack_penalty": args.backtrack_penalty, "progress_big_drop": args.progress_big_drop,
    }

    env = SumoMultiAgentEnv(sumo_config=args.sumo_config_path, rou_file=args.rou_file,
                            use_gui=args.use_gui, max_steps=args.max_steps, reward_params=reward_params)
    obs, _ = env.reset(seed=args.seed)

    if getattr(env, "agents", None):
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        act_dim = env.action_space(env.agents[0]).n
    else:
        obs_dim = env._default_observation_space.shape[0]
        act_dim = env._default_action_space.n
    global_state_dim = int(np.asarray(env.state(), dtype=np.float32).shape[0])
    max_num_agents = int(args.max_agents)

    qmix = QMIX(obs_dim=obs_dim, act_dim=act_dim, num_agents=max_num_agents,
                global_state_dim=global_state_dim, gamma=0.99, lr=args.lr,
                device=args.device, mixer_hidden=args.mixer_hidden, grad_clip=args.grad_clip, tau=args.tau,
                hard_update_interval=args.hard_update_interval, reward_scale=args.reward_scale)

    buf = ReplayBuffer(capacity=args.capacity, batch_size=args.batch_size,
                       state_dim=obs_dim, action_dim=1, n_agents=max_num_agents)

    agent2slot, slot2agent, agents = {}, [], {}

    # ── SMDP transition bookkeeping (Option B) ───────────────────────────
    # A learning transition spans from the decision point where an agent
    # CHOOSES an edge to the next decision point where it chooses again.
    # The reward of that transition is the SUM of per-step rewards accrued
    # along the way, so reward is paired with the decision that caused it.
    #   pending[aid] = dict(state, action, gstate, cum_r, mask)
    pending = {}
    # Track whether the agent currently occupies a decision zone, so we open
    # a new decision only on RE-ENTRY (the decision zone spans several steps).
    in_zone = {}

    def reset_episode_state():
        agent2slot.clear(); slot2agent.clear(); agents.clear()
        pending.clear(); in_zone.clear()

    def get_slot(aid):
        if aid in agent2slot: return agent2slot[aid]
        if len(slot2agent) >= max_num_agents: return None
        s = len(slot2agent); agent2slot[aid] = s; slot2agent.append(aid); return s

    ep_rewards, ep_steps = [], []
    kpi_speed_norm, kpi_halts_per_vehstep, kpi_arrivals_per_100 = [], [], []
    kpi_mean_wait, kpi_mean_travel, kpi_teleports = [], [], []
    kpi_goal_rate = []
    mean_td_loss, updates_per_ep, eps_per_ep = [], [], []

    stats_csv = os.path.join(args.results_dir, "qmix_stats.csv")
    with open(stats_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "update_idx", "td_loss", "grad_norm",
                                "batch_size", "eps", "buf_n"])
    update_idx = 0

    def eps_for_episode(ep):
        if not train:
            return 1.0
        t = min(1.0, ep / max(1, args.eps_anneal_episodes))
        return max(args.eps_end, args.eps_start + (args.eps_end - args.eps_start) * t)

    def flush_transition(aid, next_state, next_mask, done, gstate_next):
        """Write a completed SMDP transition for one agent into the buffer
        as a single-agent joint sample (other slots empty -> masked out)."""
        if aid not in pending:
            return
        pend = pending.pop(aid)
        slot = agent2slot.get(aid)
        if slot is None:
            return
        s = np.zeros((max_num_agents, obs_dim), np.float32)
        a = np.zeros((max_num_agents, 1), np.int64)
        r = np.zeros((max_num_agents,), np.float32)
        ns = np.zeros((max_num_agents, obs_dim), np.float32)
        d = np.zeros((max_num_agents,), np.float32)
        nmk = np.zeros((max_num_agents, act_dim), np.float32)

        s[slot] = pend["state"]
        a[slot] = pend["action"]
        r[slot] = pend["cum_r"]
        if done:
            d[slot] = 1.0
            nmk[slot] = np.ones(act_dim, dtype=np.float32)
            # ns[slot] stays zero -> terminal, no bootstrap.
        else:
            ns[slot] = next_state
            nmk[slot] = next_mask if next_mask is not None else np.ones(act_dim, dtype=np.float32)
        buf.add(global_state=pend["gstate"], state=s, action=a, reward=r,
                next_state=ns, next_global_state=gstate_next, done=d,
                active_agents=1, next_mask=nmk)

    for ep in range(args.episodes):
        t0 = time.time()
        reset_episode_state()
        obs, _ = env.reset(seed=args.seed + ep)
        ep_rew, steps = 0.0, 0
        eps = eps_for_episode(ep); eps_per_ep.append(eps)
        seen_aids = set()

        k_acc = {"speed_norm": 0.0, "halts_pv": 0.0, "arr_100": 0.0,
                 "wait": 0.0, "travel": 0.0, "teleports": 0, "goals": 0, "steps": 0}
        loss_log, updates_this_ep = [], 0

        while steps < args.max_steps:
            active = list(getattr(env, "agents", []))
            if not active: break
            seen_aids.update(active)

            gstate = np.asarray(env.state(), dtype=np.float32)
            actions = {}

            # ── Action selection + open/refresh pending decisions ────────
            for aid in active:
                if aid not in obs: continue
                slot = get_slot(aid)
                if slot is None: continue
                if aid not in agents: agents[aid] = qmix.make_agent()
                mask = env.get_local_action_mask(aid)
                act = int(agents[aid].select_action(obs[aid], epsilon=eps, mask=mask))
                actions[aid] = act

                at_dp = env._is_at_decision_point(aid)
                was_in_zone = in_zone.get(aid, False)
                if at_dp and not was_in_zone:
                    # New decision: close the previous pending transition
                    # (its next-state is the current state at this decision),
                    # then open a fresh one.
                    if aid in pending:
                        flush_transition(aid, obs[aid], mask, done=False, gstate_next=gstate)
                    pending[aid] = {"state": obs[aid].copy(), "action": act,
                                    "gstate": gstate.copy(), "cum_r": 0.0, "mask": mask}
                in_zone[aid] = at_dp

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            next_gstate = np.asarray(env.state(), dtype=np.float32)

            # ── KPI accumulation ─────────────────────────────────────────
            try:
                k = env.kpis()
                k_acc["speed_norm"] += float(k.get("speed_norm", 0.0))
                k_acc["halts_pv"] += float(k.get("halts_per_vehstep", 0.0))
                k_acc["arr_100"] += float(k.get("arrivals_per_100vehsteps", 0.0))
                k_acc["wait"] += float(k.get("mean_waiting_time", 0.0))
                k_acc["travel"] += float(k.get("mean_travel_time", 0.0))
                k_acc["teleports"] += int(k.get("teleports", 0))
                k_acc["goals"] = int(k.get("goals_reached", 0))
            except Exception:
                pass
            k_acc["steps"] += 1

            # ── Accrue per-step reward into each agent's pending decision ─
            for aid in list(pending.keys()):
                pending[aid]["cum_r"] += float(rewards.get(aid, 0.0))

            # ── Close transitions for terminal agents (goal / teleport) ──
            goal_agents = (getattr(env, '_removed_goal_agents', set())
                           | getattr(env, '_goal_arrivals_this_step', set()))
            teleported = getattr(env, '_teleported_this_step', set())
            for aid in list(pending.keys()):
                if aid in goal_agents:
                    flush_transition(aid, None, None, done=True, gstate_next=next_gstate)
                    in_zone.pop(aid, None)
                elif aid in teleported:
                    # teleport_penalty already folded into rewards by env;
                    # close as terminal so it doesn't bootstrap from a jump.
                    flush_transition(aid, None, None, done=True, gstate_next=next_gstate)
                    in_zone.pop(aid, None)

            obs = next_obs
            ep_rew += sum(rewards.values()) if rewards else 0.0
            steps += 1

            if train and steps % args.update_freq == 0 and len(buf) >= args.batch_size:
                stats = qmix.update(buf)
                if stats:
                    updates_this_ep += 1
                    td = float(stats.get("td_loss", np.nan)); loss_log.append(td)
                    gn = float(stats.get("grad_norm", 0.0))
                    with open(stats_csv, "a", newline="") as f:
                        csv.writer(f).writerow([ep + 1, update_idx, f"{td:.6f}",
                                                f"{gn:.6f}", args.batch_size,
                                                f"{eps:.4f}", len(buf)])
                    update_idx += 1

            if all(terminations.values()) if terminations else False: break

            if steps % args.print_every == 0:
                sps = steps / max(1e-6, time.time() - t0)
                print(f"[QMIX] step={steps} | partial R={ep_rew:.1f} | "
                      f"alive={len(active)} | {sps:.1f} steps/s | eps={eps:.3f}")

        if train and len(buf) >= args.batch_size:
            for _ in range(args.end_ep_updates):
                stats = qmix.update(buf)
                if stats:
                    updates_this_ep += 1
                    td = float(stats.get("td_loss", np.nan)); loss_log.append(td)
                    gn = float(stats.get("grad_norm", 0.0))
                    with open(stats_csv, "a", newline="") as f:
                        csv.writer(f).writerow([ep + 1, update_idx, f"{td:.6f}",
                                                f"{gn:.6f}", args.batch_size,
                                                f"{eps:.4f}", len(buf)])
                    update_idx += 1

        ep_rewards.append(ep_rew); ep_steps.append(steps)
        steps_k = max(1, k_acc["steps"])
        kpi_speed_norm.append(k_acc["speed_norm"] / steps_k)
        kpi_halts_per_vehstep.append(k_acc["halts_pv"] / steps_k)
        kpi_arrivals_per_100.append(k_acc["arr_100"] / steps_k)
        kpi_mean_wait.append(k_acc["wait"] / steps_k)
        kpi_mean_travel.append(k_acc["travel"] / steps_k)
        kpi_teleports.append(k_acc["teleports"])
        kpi_goal_rate.append(k_acc["goals"] / max(1, len(seen_aids)) * 100.0)
        updates_per_ep.append(updates_this_ep)
        mean_td_loss.append(float(np.nanmean(loss_log)) if loss_log else np.nan)

        sps = steps / max(1e-6, time.time() - t0)
        print(f"[QMIX] Ep {ep+1}/{args.episodes} | R={ep_rew:.1f} | "
              f"steps={steps} | speed_norm={kpi_speed_norm[-1]:.3f} | "
              f"halts/veh={kpi_halts_per_vehstep[-1]:.3f} | "
              f"goal_rate={kpi_goal_rate[-1]:.1f}% | "
              f"wait={kpi_mean_wait[-1]:.1f}s | "
              f"goals={k_acc['goals']}/{len(seen_aids)} | "
              f"eps={eps:.3f} | updates={updates_this_ep} | "
              f"td_loss={mean_td_loss[-1]:.4f} | {sps:.1f} steps/s")

    rd = args.results_dir
    for name, arr in [("rewards", ep_rewards), ("steps", ep_steps), ("kpi_speed_norm", kpi_speed_norm),
                      ("kpi_halts_per_vehstep", kpi_halts_per_vehstep),
                      ("kpi_goal_rate", kpi_goal_rate),
                      ("kpi_mean_waiting_time", kpi_mean_wait), ("kpi_mean_travel_time", kpi_mean_travel),
                      ("kpi_teleports", kpi_teleports), ("mean_td_loss", mean_td_loss),
                      ("updates_per_ep", updates_per_ep), ("eps_per_ep", eps_per_ep)]:
        np.save(os.path.join(rd, f"{name}.npy"), np.array(arr))

    env.close()
    print("[QMIX] Training complete.")

    if isinstance(sys.stdout, _Tee):
        sys.stdout.flush(); sys.stdout = sys.stdout._stream
    if isinstance(sys.stderr, _Tee):
        sys.stderr.flush(); sys.stderr = sys.stderr._stream
    _log_fh.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        for _s in (sys.stdout, sys.stderr):
            try:
                _s.flush()
            except Exception:
                pass