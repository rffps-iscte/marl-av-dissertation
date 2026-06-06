from __future__ import annotations
import numpy as np
import torch
from torch.distributions import Categorical


class MAPPOAgent:
    """
    Thin per-agent wrapper around the shared actor network.
    After removing the GRU, we no longer carry a persistent hidden state,
    so this class is now almost stateless. Kept as a class to preserve the
    public interface used in main.py (init_hidden, reset, choose_action).
    """
    def __init__(self, actor, hidden_dim: int):
        self.actor = actor
        self.hidden_dim = hidden_dim  # kept for interface compatibility

    def init_hidden(self, device="cpu"):
        # No-op: MLP actor has no recurrent state.
        pass

    def reset(self, device="cpu"):
        self.init_hidden(device)

    @torch.no_grad()
    def choose_action(self, obs, device="cpu", eval_mode: bool = False, mask=None):
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).float().to(device)
        elif torch.is_tensor(obs):
            obs_t = obs.to(device).float()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        self.actor.eval()
        dist, _ = self.actor(obs_t, None)
        probs = dist.probs

        if mask is not None:
            m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            p = probs * m
            Z = p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            use_mask = (m.sum() > 0) and (Z.item() > 0)
            if use_mask:
                probs = p / Z
        masked_dist = Categorical(probs=probs)

        action = torch.argmax(masked_dist.probs, dim=-1) if eval_mode else masked_dist.sample()
        log_prob = masked_dist.log_prob(action).squeeze(0)

        return int(action.item()), float(log_prob.item()), masked_dist
