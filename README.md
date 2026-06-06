# MARL para Routing Cooperativo de Veículos Autónomos em SUMO

Implementação e comparação de dois algoritmos de Multi-Agent Reinforcement Learning (MARL) sob o paradigma CTDE (Centralised Training, Decentralised Execution) para routing cooperativo de veículos autónomos numa rede urbana simulada em SUMO:

- **MAPPO** - Multi-Agent PPO (policy-gradient com critic centralizado)
- **QMIX** - value-based com mixing network monotónica

Ambos partilham o mesmo ambiente (`environment.py`), o mesmo cenário de tráfego e a mesma função de recompensa, para que a comparação seja justa.

---

## Requisitos

- Python 3.10+
- [SUMO](https://www.eclipse.org/sumo/) instalado, com a variável de ambiente `SUMO_HOME` definida (os scripts abortam se `SUMO_HOME` não existir).
- Dependências Python:

\`\`\`bash
pip install -r main/requirements.txt
\`\`\`

Recomenda-se um ambiente virtual:

\`\`\`bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r main/requirements.txt
\`\`\`

---

## Estrutura do projeto

\`\`\`
.
├── main/
│   ├── environment.py            # Ambiente PettingZoo ParallelEnv sobre SUMO (partilhado)
│   ├── run_experiments.py        # Orquestração de experiências
│   ├── requirements.txt
│   ├── MAPPO/
│   │   ├── main.py               # Ponto de entrada do MAPPO
│   │   ├── mappo.py              # Algoritmo (update PPO, critic centralizado)
│   │   ├── agent.py
│   │   ├── networks.py           # Actor / Critic (MLP)
│   │   └── replay_buffer.py      # Buffer on-policy (rollouts)
│   └── QMIX/
│       ├── main.py               # Ponto de entrada do QMIX
│       ├── qmix.py               # Algoritmo (mixing network, update TD)
│       ├── agent.py
│       ├── networks.py           # Q-network + MixingNetwork (hypernetworks)
│       └── replay_buffer.py      # Buffer off-policy (replay)
└── main/sumo-simulation/         # rede, rotas e configs SUMO
    ├── my_grid5x5.net.xml
    ├── my_routes_dense.rou.xml
    ├── my_simulation_dense.sumocfg
    └── ...
\`\`\`

Os algoritmos correm como módulos a partir da **raiz do projeto**:

\`\`\`bash
python -m main.MAPPO.main [opções]
python -m main.QMIX.main  [opções]
\`\`\`

---

## Cenários de tráfego

O cenário por defeito é o **denso**, desenhado para criar congestionamento no centro e tornar o routing relevante:

- Rede 5×5, edges de 180 m, uma via, junções prioritárias (sem semáforos).
- 12 fluxos × 20 veículos = **240 veículos** injetados em 0–400 s, todos a atravessar o bloco central 3×3.

Ficheiros por defeito: `my_simulation_dense.sumocfg` e `my_routes_dense.rou.xml`. Para outro cenário, usar `--sumo_config_path` e `--rou_file`.

> **Densidade:** alterar o atributo `number` nos 12 fluxos do `my_routes_dense.rou.xml`. No fim do 1.º episódio o `speed_norm` deve ficar em 0.55–0.80.

---

## Como correr o MAPPO

Treino completo (configuração validada):
\`\`\`bash
python -m main.MAPPO.main --episodes 100 --seed 1 --results_dir runs/mappo_trained_dense
\`\`\`

Baseline (política não treinada):
\`\`\`bash
python -m main.MAPPO.main --episodes 100 --seed 1 --no_train --results_dir runs/mappo_baseline_dense
\`\`\`

Variante `critic_lr` mais alto (corrida principal):
\`\`\`bash
python -m main.MAPPO.main --episodes 100 --seed 1 --critic_lr 1e-3 --results_dir runs/mappo_trained_dense_clr1e3
\`\`\`

Smoke test rápido:
\`\`\`bash
python -m main.MAPPO.main --episodes 3 --seed 1 --no_train --results_dir runs/smoke
\`\`\`

---

## Como correr o QMIX

Treino completo:
\`\`\`bash
python -m main.QMIX.main --episodes 100 --seed 1 --results_dir runs/qmix_trained_dense
\`\`\`

Baseline (epsilon=1 constante, sem updates):
\`\`\`bash
python -m main.QMIX.main --episodes 100 --seed 1 --no_train --results_dir runs/qmix_baseline_dense
\`\`\`

Diagnóstico (annealing rápido do epsilon):
\`\`\`bash
python -m main.QMIX.main --episodes 30 --seed 1 --eps_anneal_episodes 15 --results_dir runs/qmix_diag
\`\`\`

---

## Modo treino vs. baseline

Ambos aceitam `--no_train`:

| | Treino (sem a flag) | Baseline (`--no_train`) |
|---|---|---|
| **MAPPO** | Updates PPO normais. | Sem updates; política fica na inicialização (≈ uniforme). |
| **QMIX** | Updates TD; epsilon anela 1.0→0.05. | Sem updates **e** epsilon=1.0 (ações aleatórias com action mask). |

A baseline é o **chão de comparação**: ambos em `--no_train` reduzem-se a política aleatória sobre o mesmo ambiente, logo o `goal_rate` deve coincidir (~44–45%).

> **Importante:** usar sempre um `--results_dir` novo por corrida. O CSV e o log são acrescentados; reutilizar pasta mistura corridas.

---

## Onde ficam os outputs

Cada corrida grava tudo em `--results_dir`. Para `runs/exemplo`:

**Comum aos dois:**

| Ficheiro | Conteúdo |
|---|---|
| `console_AAAAMMDD_HHMMSS.log` | Cópia integral da consola (inclui stderr). Gravado mesmo se a corrida falhar. |
| `rewards.npy` | Recompensa total por episódio. |
| `steps.npy` | Nº de passos por episódio. |
| `kpi_goal_rate.npy` | **Métrica principal:** % de veículos que chegaram ao destino. |
| `kpi_speed_norm.npy` | Velocidade média normalizada (congestionamento). |
| `kpi_halts_per_vehstep.npy` | Paragens por veículo-passo. |
| `kpi_mean_waiting_time.npy` | Tempo médio de espera. |
| `kpi_mean_travel_time.npy` | Tempo médio de viagem. |
| `kpi_teleports.npy` | Nº de teleports. |

**Só MAPPO:**

| Ficheiro | Conteúdo |
|---|---|
| `ppo_stats.csv` | Por update: `actor_loss`, `critic_loss`, `entropy`, `approx_kl`, `clip_frac`, `explained_var`, `ret_mean`, `ret_std`, `adv_std_raw`. |

**Só QMIX:**

| Ficheiro | Conteúdo |
|---|---|
| `qmix_stats.csv` | Por update: `td_loss`, `grad_norm`, `batch_size`, `eps`, `buf_n`. |
| `mean_td_loss.npy` | TD loss médio por episódio. |
| `updates_per_ep.npy` | Nº de updates por episódio. |
| `eps_per_ep.npy` | Epsilon por episódio. |

> `runs/` e `results/` estão no `.gitignore` e não são versionadas. Guardar os resultados num backup à parte.

---

## Parâmetros - MAPPO

`python -m main.MAPPO.main [opções]`

### Ambiente e execução

| Parâmetro | Default | Descrição |
|---|---|---|
| `--sumo_config_path` | `.../my_simulation_dense.sumocfg` | Configuração SUMO. |
| `--rou_file` | `.../my_routes_dense.rou.xml` | Rotas (define objetivos dos agentes). |
| `--use_gui` | (flag) | SUMO com interface gráfica. |
| `--episodes` | `100` | Número de episódios. |
| `--max_steps` | `1200` | Passos máximos por episódio. |
| `--seed` | `42` | Seed. |
| `--results_dir` | `results/mappo` | Pasta de output. |
| `--device` | `cpu` | `cpu` ou `cuda`. |
| `--max_agents` | `400` | Teto de agentes simultâneos. |
| `--no_train` | (flag) | Desliga updates PPO → baseline. |
| `--print_every` | `200` | Frequência dos prints intermédios. |

### Hiperparâmetros PPO

| Parâmetro | Default | Descrição |
|---|---|---|
| `--hidden_dim` | `128` | Dimensão das camadas escondidas. |
| `--ppo_epochs` | `4` | Épocas por rollout. |
| `--mini_batch` | `64` | Tamanho do mini-batch. |
| `--entropy_coef` | `0.02` | Coeficiente de entropia. |
| `--actor_lr` | `5e-4` | Learning rate do actor. |
| `--critic_lr` | `3e-4` | Learning rate do critic. |
| `--rollout_len` | `256` | Comprimento do rollout antes de cada update. |

### Recompensa

| Parâmetro | Default |
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

## Parâmetros - QMIX

`python -m main.QMIX.main [opções]`

### Ambiente e execução

| Parâmetro | Default | Descrição |
|---|---|---|
| `--sumo_config_path` | `.../my_simulation_dense.sumocfg` | Configuração SUMO. |
| `--rou_file` | `.../my_routes_dense.rou.xml` | Rotas. |
| `--use_gui` | (flag) | SUMO com interface gráfica. |
| `--episodes` | `100` | Número de episódios. |
| `--max_steps` | `1200` | Passos máximos por episódio. |
| `--seed` | `42` | Seed. |
| `--results_dir` | `results/qmix` | Pasta de output. |
| `--device` | `cpu` | `cpu` ou `cuda`. |
| `--max_agents` | `240` | Teto de agentes (dimensiona a mixing network). |
| `--no_train` | (flag) | Desliga updates e força epsilon=1 → baseline. |
| `--print_every` | `200` | Frequência dos prints intermédios. |

### Hiperparâmetros QMIX

| Parâmetro | Default | Descrição |
|---|---|---|
| `--lr` | `1e-4` | Learning rate (Q-network + mixer). |
| `--tau` | `0.005` | Taxa de soft update do target. |
| `--hard_update_interval` | `0` | Intervalo de hard update; `0` = só soft. |
| `--mixer_hidden` | `32` | Dimensão escondida da mixing network. |
| `--hidden_dim` | `128` | Dimensão escondida da Q-network. |
| `--batch_size` | `64` | Tamanho do batch do replay. |
| `--update_freq` | `8` | Update a cada N passos. |
| `--end_ep_updates` | `2` | Updates extra no fim de cada episódio. |
| `--capacity` | `50000` | Capacidade do replay buffer. |
| `--grad_clip` | `1.0` | Limite da norma do gradiente. |
| `--reward_scale` | `1.0` | Divisor extra da recompensa conjunta. |

### Exploração (epsilon-greedy)

| Parâmetro | Default | Descrição |
|---|---|---|
| `--eps_start` | `1.0` | Epsilon inicial. |
| `--eps_end` | `0.05` | Epsilon final. |
| `--eps_anneal_episodes` | `60` | Episódios para o epsilon descer de início a fim. |

### Recompensa

Idêntica ao MAPPO: `--goal_reward 60.0`, `--time_penalty -0.02`, `--distance_factor 0.01`, `--edge_switch_bonus 0.1`, `--progress_big_drop 70.0`, `--speed_coef 0.002`, `--halt_penalty -0.1`, `--backtrack_penalty -0.5`, `--backtrack_margin 20.0`, `--invalid_action_pen -0.2`, `--congestion_penalty 0.001`, `--teleport_penalty -2.0`.

---

## Reprodutibilidade e seeds

Cada configuração deve ser corrida com várias seeds (ex.: 1, 2, 3):

\`\`\`bash
python -m main.MAPPO.main --episodes 100 --seed 1 --results_dir runs/mappo_trained_dense_seed1
python -m main.MAPPO.main --episodes 100 --seed 2 --results_dir runs/mappo_trained_dense_seed2
python -m main.MAPPO.main --episodes 100 --seed 3 --results_dir runs/mappo_trained_dense_seed3
\`\`\`

Convenção de nomes: `runs/<algoritmo>_<cenario>_<o-que-muda>_seed<N>`. Correr **em sequência, não em paralelo**.