# MARL for Cooperative Routing of Autonomous Vehicles in SUMO

Implementation and comparison of two Multi-Agent Reinforcement Learning (MARL) algorithms under the CTDE (Centralised Training, Decentralised Execution) paradigm for cooperative routing of autonomous vehicles in an urban network simulated in SUMO:

- **MAPPO** - Multi-Agent PPO (policy-gradient with centralised critic)
- **QMIX** - value-based with monotonic mixing network

Both share the same environment (`environment.py`), the same traffic scenario and the same reward function, so that the comparison is fair.

This repository contains the code accompanying the MSc dissertation *Multi-Agent Reinforcement Learning for Autonomous Vehicles in SUMO* (ISCTE, 2026). The tag `v1.0-dissertation` marks the exact state of the code used to produce the reported results.

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

The reported runs were produced with SUMO 1.26.0 in headless mode, Python 3.13 and PyTorch 2.11 (CPU build).

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

- 5×5 grid, 180 m spacing between junctions, single lane per direction, priority junctions (no traffic lights).
- 80 directed edges, 165.6 m each (169.6 m for the edges meeting the corner junctions), speed limit 13.9 m/s.
- 12 flows × 20 vehicles = **240 vehicles** injected between 0–400 s, all crossing the central 3×3 block.

Default files: `my_simulation_dense.sumocfg` and `my_routes_dense.rou.xml`. For a different scenario, use `--sumo_config_path` and `--rou_file`.

> **Density:** change the `number` attribute on the 12 flows in `my_routes_dense.rou.xml`. At the end of the 1st episode `speed_norm` should sit at 0.55–0.80.

---

## Reproducing the results reported in the dissertation

Twelve runs: two algorithms × three seeds × (training, random baseline). All parameters that differ from the
defaults documented further below are given explicitly, so these commands reproduce the reported configuration
regardless of what the defaults happen to be.

**MAPPO - training:**
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --critic_lr 1e-3 --results_dir runs/mappo_trained_dense_seed1
python -m main.MAPPO.main --episodes 100 --seed 2 --critic_lr 1e-3 --results_dir runs/mappo_trained_dense_seed2
python -m main.MAPPO.main --episodes 100 --seed 3 --critic_lr 1e-3 --results_dir runs/mappo_trained_dense_seed3
```

**MAPPO - random baseline:**
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --no_train --results_dir runs/mappo_baseline_dense_seed1
python -m main.MAPPO.main --episodes 100 --seed 2 --no_train --results_dir runs/mappo_baseline_dense_seed2
python -m main.MAPPO.main --episodes 100 --seed 3 --no_train --results_dir runs/mappo_baseline_dense_seed3
```

**QMIX - training:**
```bash
python -m main.QMIX.main --episodes 100 --seed 1 --capacity 10000 --eps_anneal_episodes 30 --results_dir runs/qmix_trained_dense_seed1
python -m main.QMIX.main --episodes 100 --seed 2 --capacity 10000 --eps_anneal_episodes 30 --results_dir runs/qmix_trained_dense_seed2
python -m main.QMIX.main --episodes 100 --seed 3 --capacity 10000 --eps_anneal_episodes 30 --results_dir runs/qmix_trained_dense_seed3
```

**QMIX - random baseline:**
```bash
python -m main.QMIX.main --episodes 100 --seed 1 --no_train --results_dir runs/qmix_baseline_dense_seed1
python -m main.QMIX.main --episodes 100 --seed 2 --no_train --results_dir runs/qmix_baseline_dense_seed2
python -m main.QMIX.main --episodes 100 --seed 3 --no_train --results_dir runs/qmix_baseline_dense_seed3
```

Run **sequentially, not in parallel**, and always use a fresh `--results_dir` per run: the CSV and the log are
appended, so reusing a folder mixes runs together. Each training run takes roughly seven and a half hours for
MAPPO and slightly over four for QMIX on a personal computer.

Expected final values, averaged over the last ten episodes of each run: goal rate around 43.5% for the baselines,
74.5% for MAPPO and 98.0% for QMIX.

---

## How to run MAPPO

Full training (configuration reported in the dissertation):
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --critic_lr 1e-3 --results_dir runs/mappo_trained_dense_seed1
```

Baseline (untrained policy):
```bash
python -m main.MAPPO.main --episodes 100 --seed 1 --no_train --results_dir runs/mappo_baseline_dense_seed1
```

Quick smoke test:
```bash
python -m main.MAPPO.main --episodes 3 --seed 1 --no_train --results_dir runs/smoke
```

---

## How to run QMIX

Full training (configuration reported in the dissertation):
```bash
python -m main.QMIX.main --episodes 100 --seed 1 --capacity 10000 --eps_anneal_episodes 30 --results_dir runs/qmix_trained_dense_seed1
```

Baseline (constant epsilon=1, no updates):
```bash
python -m main.QMIX.main --episodes 100 --seed 1 --no_train --results_dir runs/qmix_baseline_dense_seed1
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

The baseline is the **comparison floor**: both in `--no_train` reduce to a random policy over the same environment, so `goal_rate` should match (~43–45%).

Note that all reported results are measured during training, not in a separate evaluation phase, so each algorithm
is observed under its own exploration regime: MAPPO samples from its masked policy throughout, while QMIX acts
greedily at the annealed floor of epsilon after episode 30.

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

| Parameter | Default | Used in the dissertation | Description |
|---|---|---|---|
| `--hidden_dim` | `128` | `128` | Hidden layer size. |
| `--ppo_epochs` | `4` | `4` | Epochs per rollout. |
| `--mini_batch` | `64` | `64` | Mini-batch size. |
| `--entropy_coef` | `0.02` | `0.02` | Entropy coefficient (base value; adapted at run time around a target entropy of 0.5 nats). |
| `--actor_lr` | `5e-4` | `5e-4` | Actor learning rate. |
| `--critic_lr` | `3e-4` | **`1e-3`** | Critic learning rate. |
| `--rollout_len` | `256` | `256` | Rollout length before each update. |

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

| Parameter | Default | Used in the dissertation | Description |
|---|---|---|---|
| `--lr` | `1e-4` | `1e-4` | Learning rate (Q-network + mixer). |
| `--tau` | `0.005` | `0.005` | Target soft update rate. |
| `--hard_update_interval` | `0` | `0` | Hard update interval; `0` = soft only. |
| `--mixer_hidden` | `32` | `32` | Mixing network hidden size. |
| `--hidden_dim` | `128` | `128` | Q-network hidden size. |
| `--batch_size` | `64` | `64` | Replay batch size. |
| `--update_freq` | `8` | `8` | Update every N steps. |
| `--end_ep_updates` | `2` | `2` | Extra updates at the end of each episode. |
| `--capacity` | `50000` | **`10000`** | Replay buffer capacity. |
| `--grad_clip` | `1.0` | `1.0` | Gradient norm clip. |
| `--reward_scale` | `1.0` | `1.0` | Extra divisor for the joint reward. |

A capacity larger than the reported 10 000 keeps the low-reward transitions of the early exploratory episodes in
the sampling pool long after the policy has improved, which visibly stalls learning.

### Exploration (epsilon-greedy)

| Parameter | Default | Used in the dissertation | Description |
|---|---|---|---|
| `--eps_start` | `1.0` | `1.0` | Initial epsilon. |
| `--eps_end` | `0.05` | `0.05` | Final epsilon. |
| `--eps_anneal_episodes` | `60` | **`30`** | Episodes over which epsilon anneals from start to end. |

### Reward

Identical to MAPPO: `--goal_reward 60.0`, `--time_penalty -0.02`, `--distance_factor 0.01`, `--edge_switch_bonus 0.1`, `--progress_big_drop 70.0`, `--speed_coef 0.002`, `--halt_penalty -0.1`, `--backtrack_penalty -0.5`, `--backtrack_margin 20.0`, `--invalid_action_pen -0.2`, `--congestion_penalty 0.001`, `--teleport_penalty -2.0`.

---

## Seeds

Each configuration is run with three seeds (1, 2 and 3). The seed is propagated to the Python, NumPy and PyTorch
generators and, through the per-episode reseeding of the environment, to the simulator itself, so that episode *k*
of seed *j* has the same traffic realisation across algorithms and across their paired baselines.

Naming convention: `runs/<algorithm>_<scenario>_<what-changes>_seed<N>`.