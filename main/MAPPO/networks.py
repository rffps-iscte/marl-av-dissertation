import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorMLP(nn.Module):
    """
    Feed-forward actor. Replaces the previous ActorRNN (GRU).
    The observation already contains enough information (current/next/goal one-hot,
    speed, position, densities) for a markovian policy — recurrence was adding
    complexity and a train/inference mismatch without empirical benefit.

    Kept the same forward signature (obs, hidden_state) -> (dist, new_hidden)
    so we don't have to touch agent.py / mappo.py heavily. `hidden_state` is
    accepted and returned as a dummy tensor to keep the interface stable.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, act_dim)
        # Orthogonal init is a standard PPO stability trick (Engstrom et al. 2020)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.out.weight, gain=0.01)  # small init for policy head
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, obs, hidden_state=None):
        x = self.net(obs)
        logits = self.out(x)
        dist = Categorical(logits=logits)
        # Return hidden_state unchanged (dummy) for interface compatibility
        return dist, hidden_state


class CriticMLP(nn.Module):
    """
    Feed-forward centralised critic.
    Input = concat(global_state, agent_obs).

    Rationale: the previous CriticRNN received only state() which is agent-agnostic,
    so it produced the same V(s) for every agent in the same step, collapsing
    advantages across agents. Concatenating the agent's local observation gives
    the critic the context it needs to produce per-agent values (this is the
    standard CTDE — Centralised Training, Decentralised Execution — formulation
    used in the original MAPPO paper, Yu et al. 2022).
    """
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, 1)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.out.weight, gain=1.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, state, hidden_state=None):
        x = self.net(state)
        value = self.out(x).squeeze(-1)
        return value, hidden_state


# Backwards-compatible aliases so existing imports keep working.
ActorRNN = ActorMLP
CriticRNN = CriticMLP
