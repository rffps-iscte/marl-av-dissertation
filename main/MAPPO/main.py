import os, argparse, time, csv, random, sys, datetime
import numpy as np
import torch

from main.environment import SumoMultiAgentEnv
from main.MAPPO.agent import MAPPOAgent
from main.MAPPO.mappo import MAPPO
from main.MAPPO.replay_buffer import ReplayBuffer


class _Tee:
    """Mirror a stream (stdout/stderr) to the console *and* a log file.

    Used to capture the full console output of a run without having to
    redirect manually. The SUMO 'Vehicle ... is not known' lines go to
    stderr, so we tee stderr as well to keep everything in one file.
    """
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    # so libraries that probe the stream don't choke
    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_parser():
    p = argparse.ArgumentParser(description="SUMO MAPPO (dense-traffic scenario)")
    # NOTE: defaults below are the *validated* values from the 100-episode run.
    # The dense scenario files must be placed in main/sumo-simulation/.
    p.add_argument("--sumo_config_path",
                   default=os.path.abspath("main/sumo-simulation/my_simulation_dense.sumocfg"))
    p.add_argument("--rou_file",
                   default=os.path.abspath("main/sumo-simulation/my_routes_dense.rou.xml"))
    p.add_argument("--use_gui", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    # Dense scenario: flows inject over 0-400 s; with ~240 vehicles the
    # network drains well before step 1200, so 1200 covers the whole episode
    # without paying for a long empty-network tail.
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results/mappo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_agents", type=int, default=400)

    # Baseline mode: run the (untrained) policy WITHOUT calling mappo.update().
    # Use this to produce the random-policy reference curve for the thesis.
    p.add_argument("--no_train", action="store_true",
                   help="Disable PPO updates -> untrained-policy baseline.")

    # ── Network / PPO hyperparameters (validated values) ─────────────────
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mini_batch", type=int, default=64)
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument("--actor_lr", type=float, default=5e-4)
    p.add_argument("--critic_lr", type=float, default=1e-3) # 3e-4
    p.add_argument("--rollout_len", type=int, default=256)

    # ── Reward parameters (validated values) ─────────────────────────────
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
    return p


def build_critic_state(global_state: np.ndarray, agent_obs: np.ndarray) -> np.ndarray:
    """
    FIX P3b: critic input = concat(global_state, agent_obs).
    This gives the centralised critic enough signal to produce per-agent values
    instead of a single V(s) that's identical across all agents in a step.
    """
    return np.concatenate([global_state, agent_obs]).astype(np.float32)


def main():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set.")
    args = build_parser().parse_args()
    train = not args.no_train
    print(f"[MAPPO] mode = {'TRAIN' if train else 'BASELINE (no PPO updates)'}")

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Console logging ──────────────────────────────────────────────────
    # Mirror everything printed to the console into a timestamped log file
    # inside results_dir, so long runs are saved without manual copy-paste.
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = os.path.join(args.results_dir, f"console_{_ts}.log")
    _log_fh = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"[MAPPO] console log -> {_log_path}")

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    reward_params = {
        "time_penalty": args.time_penalty, "goal_reward": args.goal_reward,
        "distance_factor": args.distance_factor, "invalid_action_penalty": args.invalid_action_pen,
        "halt_penalty": args.halt_penalty, "speed_coef": args.speed_coef,
        "edge_switch_bonus": args.edge_switch_bonus, "congestion_penalty": args.congestion_penalty,
        "teleport_penalty": args.teleport_penalty, "backtrack_margin": args.backtrack_margin,
        "backtrack_penalty": args.backtrack_penalty, "progress_big_drop": args.progress_big_drop,
    }

    env = SumoMultiAgentEnv(sumo_config=args.sumo_config_path, rou_file=args.rou_file,
                            use_gui=args.use_gui, max_steps=args.max_steps,
                            reward_params=reward_params)
    obs, _ = env.reset(seed=args.seed)

    # Dimension discovery
    if getattr(env, "agents", None):
        first = env.agents[0]
        obs_dim = env.observation_space(first).shape[0]
        act_dim = env.action_space(first).n
    else:
        obs_dim = env._default_observation_space.shape[0]
        act_dim = env._default_action_space.n
    global_state_dim = int(np.asarray(env.state(), dtype=np.float32).shape[0])
    critic_state_dim = global_state_dim + obs_dim  # FIX P3b: concat dimension

    print(f"[MAPPO] obs_dim={obs_dim} act_dim={act_dim} "
          f"global_state_dim={global_state_dim} critic_state_dim={critic_state_dim}")

    mappo = MAPPO(obs_dim, act_dim, critic_state_dim, hidden_dim=args.hidden_dim,
                  actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                  ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch,
                  entropy_coef=args.entropy_coef, device=device)
    buf = ReplayBuffer(capacity=200_000)
    agents = {}

    # Per-agent last transition for terminal reward injection
    # (goal arrivals AND teleports — FIX P2)
    last_transition = {}  # aid -> (obs, action, logp, value, critic_state, mask)

    # ── CSV logging ──────────────────────────────────────────────────────
    stats_csv = os.path.join(args.results_dir, "ppo_stats.csv")
    if not os.path.exists(stats_csv):
        with open(stats_csv, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "update_idx", "actor_loss", "critic_loss",
                                    "entropy", "approx_kl", "clip_frac", "batch",
                                    "buf_n", "buf_term_pct",
                                    "explained_var", "ret_mean", "ret_std", "adv_std_raw"])

    ep_rewards, ep_steps = [], []
    kpi_speed_norm, kpi_halts_per_vehstep, kpi_arrivals_per_100 = [], [], []
    kpi_mean_wait, kpi_mean_travel, kpi_teleports = [], [], []
    kpi_goal_rate = []
    update_idx = 0

    # FIX P2: total teleports tracked per episode for diagnostic
    teleport_transitions_stored = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for aid in list(agents.keys()):
            agents[aid].reset(device=device)
        last_transition.clear()
        teleport_transitions_stored_ep = 0

        ep_reward, ep_step = 0.0, 0
        k_acc = {"speed_norm": 0.0, "halts_pv": 0.0, "arr_100": 0.0,
                 "wait": 0.0, "travel": 0.0, "teleports": 0, "goals": 0, "steps": 0}
        seen_aids = set()  # every distinct vehicle that appeared this episode
        t0 = time.time()

        while getattr(env, "agents", None) and len(env.agents) > 0 and ep_step < args.max_steps:
            seen_aids.update(env.agents)
            gstate = np.asarray(env.state(), dtype=np.float32)

            actions, acted = {}, []
            active = list(env.agents)[:args.max_agents]

            for aid in active:
                if obs is None or aid not in obs:
                    continue

                # FIX P3b: compute critic value per-agent using concatenated state
                critic_state = build_critic_state(gstate, obs[aid])
                with torch.no_grad():
                    v_pred, _ = mappo.critic(
                        torch.tensor(critic_state, device=device).unsqueeze(0), None
                    )
                    value_scalar = float(v_pred.squeeze(0).item())

                mask = env.get_local_action_mask(aid)
                if aid not in agents:
                    agents[aid] = MAPPOAgent(mappo.actor, mappo.hidden_dim)
                    agents[aid].reset(device=device)

                a, logp, _ = agents[aid].choose_action(
                    obs[aid], device=device, eval_mode=False, mask=mask
                )
                actions[aid] = a

                # Record transitions at decision points
                if env._is_at_decision_point(aid):
                    acted.append((aid, obs[aid], a, logp, value_scalar, critic_state, mask))
                    last_transition[aid] = (obs[aid], a, logp, value_scalar, critic_state, mask)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

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

            # ── FIX P2 (REVISED): unified terminal-transition handling ────────
            #
            # Critical bug identified from empirical analysis:
            # Previously, if an agent was at a decision point in step N (thus
            # registered in `acted`) AND reached the goal during the same step's
            # simulationStep, the code would skip the terminal injection (because
            # `aid in acted_aids`), and the regular `acted` loop would store the
            # transition with done=False and without the goal reward. Result:
            # ~0% terminal transitions in the buffer, which made the critic
            # unable to learn V(goal_state) correctly.
            #
            # Fix: determine terminal status FIRST for every acted agent, and
            # handle goals/teleports that weren't in `acted` separately.
            goal_agents = (getattr(env, '_removed_goal_agents', set())
                           | getattr(env, '_goal_arrivals_this_step', set()))
            teleported = getattr(env, '_teleported_this_step', set())
            acted_aids = {a[0] for a in acted}

            # Pass 1: agents that acted this step — use their recorded (obs, a, logp, v)
            # but override reward/done if they reached goal or teleported.
            for (aid, o_a, a, lp, v, cs, m) in acted:
                r = float(rewards.get(aid, 0.0))
                d = bool(terminations.get(aid, False) or truncations.get(aid, False))

                if aid in goal_agents:
                    # Agent reached goal this step — force terminal with goal reward.
                    # The env's _compute_rewards already put goal_reward into rewards[aid],
                    # but done was False at the time of the critic value estimate.
                    r = float(args.goal_reward) if aid in goal_agents else r
                    d = True
                elif aid in teleported:
                    r = float(rewards.get(aid, 0.0)) + float(args.teleport_penalty)
                    d = True
                    teleport_transitions_stored_ep += 1

                buf.store(o_a, a, lp, r, d, v, cs, mask=m)

                # Clean up last_transition bookkeeping for terminated agents
                if d and aid in last_transition:
                    del last_transition[aid]

            # Pass 2: agents that reached goal this step but were NOT at a decision
            # point (rarer case — agent was removed while cruising). Use the last
            # known (obs, a, logp, v) from a previous decision.
            for aid in goal_agents:
                if aid in acted_aids:
                    continue
                if aid in last_transition:
                    lt_obs, lt_a, lt_logp, lt_v, lt_cs, lt_mask = last_transition[aid]
                    buf.store(lt_obs, lt_a, lt_logp, float(args.goal_reward),
                              True, lt_v, lt_cs, mask=lt_mask)
                    del last_transition[aid]

            # Pass 3: teleports that weren't in acted nor goals
            for aid in teleported:
                if aid in acted_aids or aid in goal_agents:
                    continue
                if aid in last_transition:
                    lt_obs, lt_a, lt_logp, lt_v, lt_cs, lt_mask = last_transition[aid]
                    tp_r = float(rewards.get(aid, 0.0)) + float(args.teleport_penalty)
                    buf.store(lt_obs, lt_a, lt_logp, tp_r, True, lt_v, lt_cs, mask=lt_mask)
                    del last_transition[aid]
                    teleport_transitions_stored_ep += 1

            ep_reward += sum(rewards.values()) if rewards else 0.0
            ep_step += 1
            obs = next_obs

            # ── PPO update when rollout is full ───────────────────────────
            if len(buf) >= args.rollout_len:
                # Bootstrap value for the last (non-terminal) state of any live agent
                with torch.no_grad():
                    gs_now = np.asarray(env.state(), dtype=np.float32)
                    # Use a zero obs as placeholder (bootstrap is a rough estimate anyway)
                    boot_state = np.concatenate([gs_now, np.zeros(obs_dim, dtype=np.float32)])
                    v_boot, _ = mappo.critic(
                        torch.tensor(boot_state, device=device).unsqueeze(0).float(), None
                    )
                    last_val = float(v_boot.squeeze(0).item())

                bstats = buf.stats()
                buf.compute_returns_and_advantages(last_val, gamma=mappo.gamma, lam=mappo.lam)
                stats = (mappo.update(buf.get()) or {}) if train else {}
                buf.clear()

                with open(stats_csv, "a", newline="") as f:
                    csv.writer(f).writerow([ep + 1, update_idx,
                        f"{stats.get('actor_loss', 0.0):.6f}",
                        f"{stats.get('critic_loss', 0.0):.6f}",
                        f"{stats.get('entropy', 0.0):.6f}",
                        f"{stats.get('approx_kl', 0.0):.6f}",
                        f"{stats.get('clip_frac', 0.0):.6f}", args.mini_batch,
                        bstats["n"], f"{bstats['pct_terminal']:.4f}",
                        f"{stats.get('explained_var', 0.0):.4f}",
                        f"{stats.get('ret_mean', 0.0):.4f}",
                        f"{stats.get('ret_std', 0.0):.4f}",
                        f"{stats.get('adv_std_raw', 0.0):.4f}"])
                update_idx += 1

            if ep_step % args.print_every == 0:
                sps = ep_step / max(1e-6, (time.time() - t0))
                # FIX P1.2: normalise reward by number of alive agents for interpretability
                norm_r = ep_reward / max(1, len(active))
                print(f"[MAPPO] step={ep_step} | partial R={ep_reward:.1f} "
                      f"(per-agent {norm_r:.2f}) | alive={len(active)} | {sps:.1f} steps/s")

        # End-of-episode flush
        if len(buf) > 0:
            buf.compute_returns_and_advantages(0.0, gamma=mappo.gamma, lam=mappo.lam)
            bstats = buf.stats()
            stats = (mappo.update(buf.get()) or {}) if train else {}
            buf.clear()
            with open(stats_csv, "a", newline="") as f:
                csv.writer(f).writerow([ep + 1, update_idx,
                    f"{stats.get('actor_loss', 0.0):.6f}",
                    f"{stats.get('critic_loss', 0.0):.6f}",
                    f"{stats.get('entropy', 0.0):.6f}",
                    f"{stats.get('approx_kl', 0.0):.6f}",
                    f"{stats.get('clip_frac', 0.0):.6f}", args.mini_batch,
                    bstats["n"], f"{bstats['pct_terminal']:.4f}",
                    f"{stats.get('explained_var', 0.0):.4f}",
                    f"{stats.get('ret_mean', 0.0):.4f}",
                    f"{stats.get('ret_std', 0.0):.4f}",
                    f"{stats.get('adv_std_raw', 0.0):.4f}"])
            update_idx += 1

        ep_rewards.append(ep_reward); ep_steps.append(ep_step)
        steps_k = max(1, int(k_acc["steps"]))
        kpi_speed_norm.append(k_acc["speed_norm"] / steps_k)
        kpi_halts_per_vehstep.append(k_acc["halts_pv"] / steps_k)
        kpi_arrivals_per_100.append(k_acc["arr_100"] / steps_k)
        kpi_mean_wait.append(k_acc["wait"] / steps_k)
        kpi_mean_travel.append(k_acc["travel"] / steps_k)
        kpi_teleports.append(k_acc["teleports"])
        kpi_goal_rate.append(k_acc["goals"] / max(1, len(seen_aids)) * 100.0)
        teleport_transitions_stored += teleport_transitions_stored_ep

        print(f"[MAPPO] Ep {ep+1}/{args.episodes} | R={ep_reward:.1f} | "
              f"steps={ep_step} | speed_norm={kpi_speed_norm[-1]:.3f} | "
              f"halts/veh={kpi_halts_per_vehstep[-1]:.3f} | "
              f"goal_rate={kpi_goal_rate[-1]:.1f}% | "
              f"wait={kpi_mean_wait[-1]:.1f}s | "
              f"goals={k_acc['goals']}/{len(seen_aids)} | "
              f"tp_stored={teleport_transitions_stored_ep}")

    rd = args.results_dir
    np.save(os.path.join(rd, "rewards.npy"), np.array(ep_rewards))
    np.save(os.path.join(rd, "steps.npy"), np.array(ep_steps))
    np.save(os.path.join(rd, "kpi_speed_norm.npy"), np.array(kpi_speed_norm))
    np.save(os.path.join(rd, "kpi_halts_per_vehstep.npy"), np.array(kpi_halts_per_vehstep))
    np.save(os.path.join(rd, "kpi_goal_rate.npy"), np.array(kpi_goal_rate))
    np.save(os.path.join(rd, "kpi_mean_waiting_time.npy"), np.array(kpi_mean_wait))
    np.save(os.path.join(rd, "kpi_mean_travel_time.npy"), np.array(kpi_mean_travel))
    np.save(os.path.join(rd, "kpi_teleports.npy"), np.array(kpi_teleports))
    print(f"[MAPPO] Training done. Teleport transitions stored total: "
          f"{teleport_transitions_stored}")
    env.close()

    # Restore streams and close the log file cleanly.
    if isinstance(sys.stdout, _Tee):
        sys.stdout.flush(); sys.stdout = sys.stdout._stream
    if isinstance(sys.stderr, _Tee):
        sys.stderr.flush(); sys.stderr = sys.stderr._stream
    _log_fh.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure the log file captures crashes / Ctrl+C, not just clean exits.
        for _s in (sys.stdout, sys.stderr):
            try:
                _s.flush()
            except Exception:
                pass