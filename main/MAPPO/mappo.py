from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from main.MAPPO.networks import ActorMLP, CriticMLP


class MAPPO:
    """
    MAPPO trainer (CTDE).

    Key design choices vs the previous version:
      * Feed-forward actor/critic (no GRU) — removes train/inference mismatch.
      * Critic input is concat(global_state, agent_obs) so V is per-agent,
        giving non-degenerate advantages.
      * Standard PPO tricks: clipped surrogate, value clipping, GAE normalisation,
        entropy bonus, KL early stop, orthogonal init (in networks.py).
    """
    def __init__(self, obs_dim: int, act_dim: int, state_dim: int,
                 hidden_dim: int = 128, actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 ppo_epochs: int = 4, mini_batch_size: int = 64,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, device: str = "cpu"):
        self.device = torch.device(device)
        self.obs_dim, self.act_dim, self.state_dim = obs_dim, act_dim, state_dim
        self.hidden_dim = hidden_dim
        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.ppo_epochs, self.mini_batch_size = ppo_epochs, mini_batch_size
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.max_grad_norm = max_grad_norm

        self.actor = ActorMLP(obs_dim, act_dim, hidden_dim).to(self.device)
        # NOTE: critic now takes state_dim (which the caller sets = global_state_dim + obs_dim)
        self.critic = CriticMLP(state_dim, hidden_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Adaptive entropy (mild): if policy collapses, bump entropy; else decay to base.
        self.entropy_coef_base = entropy_coef
        self.entropy_coef_current = entropy_coef
        self.entropy_target = 0.5
        self.entropy_coef_max = entropy_coef * 10.0

        # KL early stopping — standard PPO trick to avoid destructive updates.
        self.target_kl = 0.02

    @staticmethod
    def _np(x, dtype):
        if isinstance(x, np.ndarray): return x.astype(dtype, copy=False)
        if torch.is_tensor(x): return x.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(x, dtype=dtype)

    def _ten(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def update(self, buffer: dict) -> dict:
        obs     = self._ten(self._np(buffer["obs"], np.float32))
        actions = self._ten(self._np(buffer["actions"], np.int64), torch.long)
        old_lp  = self._ten(self._np(buffer["log_probs"], np.float32))
        returns = self._ten(self._np(buffer["returns"], np.float32))
        adv     = self._ten(self._np(buffer["advantages"], np.float32))
        states  = self._ten(self._np(buffer["states"], np.float32))
        masks   = self._ten(self._np(buffer["masks"], np.float32))

        N = obs.size(0)
        if N == 0: return {}

        # ── Diagnostics on raw (pre-normalisation) values ─────────────────
        # values = returns - advantages(raw). The "explained variance" of the
        # value function is the standard PPO health check (SB3-style): it is
        # ~1 when the critic predicts returns well, ~0 when it is no better
        # than predicting the mean, and negative when it is actively wrong.
        ret_mean = float(returns.mean())
        ret_std = float(returns.std(unbiased=False))
        adv_std_raw = float(adv.std(unbiased=False))
        _ret_var = float(returns.var(unbiased=False))
        explained_var = 1.0 - float(adv.var(unbiased=False)) / (_ret_var + 1e-8)

        # Advantage normalisation (per rollout).
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        B = min(self.mini_batch_size, N)
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        nsteps = 0
        early_stopped = False

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            idx = torch.randperm(N, device=self.device)
            epoch_kl, epoch_batches = 0.0, 0

            for s in range(0, N, B):
                mb = idx[s:s+B]
                o, a, olp, ret, ad, st, mk = (
                    obs[mb], actions[mb], old_lp[mb], returns[mb],
                    adv[mb], states[mb], masks[mb]
                )

                # ── Actor forward ─────────────────────────────────────────
                dist, _ = self.actor(o, None)
                p = dist.probs
                pm = p * mk
                Z = pm.sum(dim=1, keepdim=True).clamp_min(1e-8)
                use_mask = (mk.sum(dim=1, keepdim=True) > 0).float()
                p_final = use_mask * (pm / Z) + (1.0 - use_mask) * p
                mdist = Categorical(probs=p_final)

                logp = mdist.log_prob(a)
                entropy = mdist.entropy().mean()

                # PPO clipped surrogate objective
                ratio = (logp - olp).exp()
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * ad
                actor_loss = -(torch.min(surr1, surr2)).mean() - self.entropy_coef_current * entropy

                with torch.no_grad():
                    approx_kl = (olp - logp).mean().clamp_min(0.0)
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                # ── Critic forward ────────────────────────────────────────
                # Use Huber loss (smooth L1) instead of MSE for robustness against
                # outlier returns. With goal_reward=200 and shaping ~1-5, a single
                # goal transition in a mini-batch produces huge MSE gradients that
                # destabilise the critic. Huber bounds this: quadratic for small
                # errors, linear for large errors.
                v_pred, _ = self.critic(st, None)
                critic_loss = F.smooth_l1_loss(v_pred, ret)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += approx_kl.item()
                stats["clip_frac"] += clip_frac.item()
                nsteps += 1
                epoch_kl += approx_kl.item()
                epoch_batches += 1

            # KL early stopping: if the policy drifted too far this epoch, stop.
            if epoch_batches > 0 and (epoch_kl / epoch_batches) > self.target_kl:
                early_stopped = True

        if nsteps > 0:
            for k in stats: stats[k] /= nsteps

            # Adaptive entropy: nudge coefficient based on current policy entropy.
            avg_entropy = stats["entropy"]
            if avg_entropy < self.entropy_target:
                self.entropy_coef_current = min(self.entropy_coef_max, self.entropy_coef_current * 1.5)
            else:
                self.entropy_coef_current = max(self.entropy_coef_base, self.entropy_coef_current * 0.98)
            stats["entropy_coef"] = self.entropy_coef_current
            stats["early_stopped"] = float(early_stopped)
            stats["explained_var"] = explained_var
            stats["ret_mean"] = ret_mean
            stats["ret_std"] = ret_std
            stats["adv_std_raw"] = adv_std_raw

        return stats