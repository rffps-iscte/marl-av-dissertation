# MARL for Cooperative Routing of Autonomous Vehicles in SUMO

Implementation and comparison of two Multi-Agent Reinforcement Learning (MARL) algorithms under the CTDE (Centralised Training, Decentralised Execution) paradigm for cooperative routing of autonomous vehicles in an urban network simulated in SUMO:

- **MAPPO** - Multi-Agent PPO (policy-gradient with centralised critic)
- **QMIX** - value-based with monotonic mixing network

Both share the same environment (`environment.py`), the same traffic scenario and the same reward function, so that the comparison is fair.

---

## Requirements

- Python 3.10+
- [SUMO](https://www.eclipse.org/sumo/) installed, with the `SUMO_HOME` environment variable defined (the scripts abort if `SUMO_HOME` is not set).
- Python dependencies:

```bash
pip install -r main/requirements.txt
```

A virtual environment is recommended:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r main/requirements.txt
```

---

## Project structure

```
.
├── main/
│   ├── environment.py            # PettingZoo ParallelEnv on top of SUMO (shared)
│   ├── run_experiments.py        # Experiment orchestration
│   ├── requirements.txt
│   ├── MAPPO/
│   │   ├── main.py               # MAPPO entry point
│   │   ├── mappo.py              # Algorithm (PPO update, centralised critic)
│   │   ├── agent.py
│   │   ├── networks.py           # Actor / Critic (MLP)
│   │   └── replay_buffer.py      # On-policy buffer (rollouts)
│   └── QMIX/
│       ├── main.py               # QMIX entry point
│       ├── qmix.py               # Algorithm (mixing network, TD update)
│       ├── agent.py
│       ├── networks.py           # Q-network + MixingNetwork (hypernetworks)
│       └── replay_buffer.py      # Off-policy buffer (replay)
└── main/sumo-simulation/         # SUMO network, routes and configs
    ├── my_grid5x5.net.xml
    ├── my_routes_dense.rou.xml
    ├── my_simulation_dense.sumocfg
    └── ...
```

The algorithms run as modules from the **project root**:

```bash
python -m main.MAPPO.main [options]
python -m main.QMIX.main  [options]
```

---

## Traffic scenarios

The default scenario is the **dense** one, designed to create congestion in the centre and make routing relevant:

- 5×5 grid, 180 m edges, single lane, priority junctions (no traffic lights).
- 12 flows × 20 vehicles = **240 vehicles** injected between 0–400 s, all crossing the central 3×3 block.

Default files: `my_simulation_dense.sumocfg` and `my_routes_dense.rou.xml`. For a different scenario, use `--sumo_config_path` and `--rou_file`.

> **Density:** change the `number` attribute on the 12 flows in `my_routes_dense.rou.xml`. At the end of the 1st episode `speed_norm` should sit at 0.55–0.80.

---

## How to run MAPPO

Full training (validated configuration):
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --results_dir runs/mappo_trained_dense
```

Baseline (untrained policy):
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --no_train --results_dir runs/mappo_baseline_dense
```

Higher `critic_lr` variant (primary run):
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --critic_lr 1e-3 --results_dir runs/mappo_trained_dense_clr1e3
```

Quick smoke test:
```bash
python -m main.MAPPO.main --episodes 3 --seed 1 --no_train --results_dir runs/smoke
```

---

## How to run QMIX

Full training:
```bash
python -m main.QMIX.main --episodes 100 --seed 1 --results_dir runs/qmix_trained_dense
```

Baseline (constant epsilon=1, no updates):
```bash
python -m main.QMIX.main --episodes 100 --seed 1 --no_train --results_dir runs/qmix_baseline_dense
```

Diagnostic (fast epsilon annealing):
```bash
python -m main.QMIX.main --episodes 30 --seed 1 --eps_anneal_episodes 15 --results_dir runs/qmix_diag
```

---

## Training mode vs. baseline

Both accept `--no_train`:

| | Training (without the flag) | Baseline (`--no_train`) |
|---|---|---|
| **MAPPO** | Normal PPO updates. | No updates; policy stays at initialisation (≈ uniform). |
| **QMIX** | TD updates; epsilon anneals 1.0→0.05. | No updates **and** epsilon=1.0 (random actions with action mask). |

The baseline is the **comparison floor**: both in `--no_train` reduce to a random policy over the same environment, so `goal_rate` should match (~44–45%).

> **Important:** always use a fresh `--results_dir` per run. The CSV and the log are appended; reusing a folder mixes runs together.

---

## Where outputs go

Each run writes everything to `--results_dir`. For `runs/example`:

**Common to both:**

| File | Contents |
|---|---|
| `console_YYYYMMDD_HHMMSS.log` | Full console copy (includes stderr). Written even if the run fails. |
| `rewards.npy` | Total reward per episode. |
| `steps.npy` | Number of steps per episode. |
| `kpi_goal_rate.npy` | **Main metric:** % of vehicles that reached their destination. |
| `kpi_speed_norm.npy` | Mean normalised speed (congestion). |
| `kpi_halts_per_vehstep.npy` | Halts per vehicle-step. |
| `kpi_mean_waiting_time.npy` | Mean waiting time. |
| `kpi_mean_travel_time.npy` | Mean travel time. |
| `kpi_teleports.npy` | Number of teleports. |

**MAPPO only:**

| File | Contents |
|---|---|
| `ppo_stats.csv` | Per update: `actor_loss`, `critic_loss`, `entropy`, `approx_kl`, `clip_frac`, `explained_var`, `ret_mean`, `ret_std`, `adv_std_raw`. |

**QMIX only:**

| File | Contents |
|---|---|
| `qmix_stats.csv` | Per update: `td_loss`, `grad_norm`, `batch_size`, `eps`, `buf_n`. |
| `mean_td_loss.npy` | Mean TD loss per episode. |
| `updates_per_ep.npy` | Number of updates per episode. |
| `eps_per_ep.npy` | Epsilon per episode. |

> `runs/` and `results/` are in `.gitignore` and are not versioned. Keep results in a separate backup.

---

## Parameters - MAPPO

`python -m main.MAPPO.main [options]`

### Environment and execution

| Parameter | Default | Description |
|---|---|---|
| `--sumo_config_path` | `.../my_simulation_dense.sumocfg` | SUMO configuration. |
| `--rou_file` | `.../my_routes_dense.rou.xml` | Routes (defines agent goals). |
| `--use_gui` | (flag) | Run SUMO with the graphical interface. |
| `--episodes` | `100` | Number of episodes. |
| `--max_steps` | `1200` | Maximum steps per episode. |
| `--seed` | `42` | Seed. |
| `--results_dir` | `results/mappo` | Output folder. |
| `--device` | `cpu` | `cpu` or `cuda`. |
| `--max_agents` | `400` | Cap on simultaneous agents. |
| `--no_train` | (flag) | Disable PPO updates → baseline. |
| `--print_every` | `200` | Frequency of intermediate prints. |

### PPO hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--hidden_dim` | `128` | Hidden layer size. |
| `--ppo_epochs` | `4` | Epochs per rollout. |
| `--mini_batch` | `64` | Mini-batch size. |
| `--entropy_coef` | `0.02` | Entropy coefficient. |
| `--actor_lr` | `5e-4` | Actor learning rate. |
| `--critic_lr` | `3e-4` | Critic learning rate. |
| `--rollout_len` | `256` | Rollout length before each update. |

### Reward

| Parameter | Default |
|---|---|
| `--goal_reward` | `60.0` |
| `--time_penalty` | `-0.02` |
| `--distance_factor` | `0.01` |
| `--edge_switch_bonus` | `0.1` |
| `--progress_big_drop` | `70.0` |
| `--speed_coef` | `0.002` |
| `--halt_penalty` | `-0.1` |
| `--backtrack_penalty` | `-0.5` |
| `--backtrack_margin` | `20.0` |
| `--invalid_action_pen` | `-0.2` |
| `--congestion_penalty` | `0.001` |
| `--teleport_penalty` | `-2.0` |

---

## Parameters - QMIX

`python -m main.QMIX.main [options]`

### Environment and execution

| Parameter | Default | Description |
|---|---|---|
| `--sumo_config_path` | `.../my_simulation_dense.sumocfg` | SUMO configuration. |
| `--rou_file` | `.../my_routes_dense.rou.xml` | Routes. |
| `--use_gui` | (flag) | Run SUMO with the graphical interface. |
| `--episodes` | `100` | Number of episodes. |
| `--max_steps` | `1200` | Maximum steps per episode. |
| `--seed` | `42` | Seed. |
| `--results_dir` | `results/qmix` | Output folder. |
| `--device` | `cpu` | `cpu` or `cuda`. |
| `--max_agents` | `240` | Cap on agents (sizes the mixing network). |
| `--no_train` | (flag) | Disable updates and force epsilon=1 → baseline. |
| `--print_every` | `200` | Frequency of intermediate prints. |

### QMIX hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--lr` | `1e-4` | Learning rate (Q-network + mixer). |
| `--tau` | `0.005` | Target soft update rate. |
| `--hard_update_interval` | `0` | Hard update interval; `0` = soft only. |
| `--mixer_hidden` | `32` | Mixing network hidden size. |
| `--hidden_dim` | `128` | Q-network hidden size. |
| `--batch_size` | `64` | Replay batch size. |
| `--update_freq` | `8` | Update every N steps. |
| `--end_ep_updates` | `2` | Extra updates at the end of each episode. |
| `--capacity` | `50000` | Replay buffer capacity. |
| `--grad_clip` | `1.0` | Gradient norm clip. |
| `--reward_scale` | `1.0` | Extra divisor for the joint reward. |

### Exploration (epsilon-greedy)

| Parameter | Default | Description |
|---|---|---|
| `--eps_start` | `1.0` | Initial epsilon. |
| `--eps_end` | `0.05` | Final epsilon. |
| `--eps_anneal_episodes` | `60` | Episodes over which epsilon anneals from start to end. |

### Reward

Identical to MAPPO: `--goal_reward 60.0`, `--time_penalty -0.02`, `--distance_factor 0.01`, `--edge_switch_bonus 0.1`, `--progress_big_drop 70.0`, `--speed_coef 0.002`, `--halt_penalty -0.1`, `--backtrack_penalty -0.5`, `--backtrack_margin 20.0`, `--invalid_action_pen -0.2`, `--congestion_penalty 0.001`, `--teleport_penalty -2.0`.

---

## Reproducibility and seeds

Each configuration should be run with several seeds (e.g., 1, 2, 3):

```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --results_dir runs/mappo_trained_dense_seed1
python -m main.MAPPO.main --episodes 100 --seed 2 --results_dir runs/mappo_trained_dense_seed2
python -m main.MAPPO.main --episodes 100 --seed 3 --results_dir runs/mappo_trained_dense_seed3
```

Naming convention: `runs/<algorithm>_<scenario>_<what-changes>_seed<N>`. Run **sequentially, not in parallel**.