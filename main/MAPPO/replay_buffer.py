from __future__ import annotations
import numpy as np


class ReplayBuffer:
    """
    Simple on-policy rollout buffer for MAPPO.

    Stores (obs, action, log_prob, reward, done, value, state, mask) per transition.
    `state` is expected to be already concatenated with the agent's obs upstream
    (centralised-critic input in CTDE).

    Note on GAE with mixed agents: we store transitions from many agents in
    a single sequential buffer. `done=True` correctly terminates bootstrap for
    that particular transition, which is the correct behaviour per-agent.
    The ordering between agents' transitions is not temporal in the usual sense,
    but since advantages are computed backwards and `done` zeros bootstrap for
    terminal steps, the per-agent trajectory boundaries are respected as long
    as terminal transitions are stored with done=True (which main.py does).
    """

    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.clear()

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
        self.states, self.masks = [], []

    def __len__(self):
        return len(self.rewards)

    def store(self, obs, action, logp, reward, done, value, state, mask=None):
        if len(self.rewards) >= self.capacity:
            for lst in [self.obs, self.actions, self.log_probs, self.rewards,
                        self.dones, self.values, self.states, self.masks]:
                lst.pop(0)

        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(logp))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.masks.append(np.asarray(mask, dtype=np.float32) if mask is not None else None)

    def stats(self):
        """Quick diagnostics used by main.py to monitor buffer health."""
        n = len(self.rewards)
        if n == 0:
            return {"n": 0, "n_terminal": 0, "pct_terminal": 0.0,
                    "mean_reward": 0.0, "mean_value": 0.0}
        dones = np.asarray(self.dones, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        return {
            "n": n,
            "n_terminal": int(dones.sum()),
            "pct_terminal": float(dones.mean()),
            "mean_reward": float(rewards.mean()),
            "mean_value": float(values.mean()),
        }

    def compute_returns_and_advantages(self, last_value: float,
                                       gamma: float = 0.99, lam: float = 0.95):
        n = len(self.rewards)
        if n == 0:
            self.returns = np.array([], dtype=np.float32)
            self.advantages = np.array([], dtype=np.float32)
            return

        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        self.returns = (adv + values[:-1]).astype(np.float32)
        self.advantages = adv.astype(np.float32)

    def get(self):
        if any(m is None for m in self.masks):
            act_dim = next((len(m) for m in self.masks if m is not None), None)
            if act_dim is None:
                masks_np = None
            else:
                masks_np = np.stack([(m if m is not None else np.ones(act_dim, dtype=np.float32))
                                     for m in self.masks], axis=0).astype(np.float32)
        else:
            masks_np = np.stack(self.masks, axis=0).astype(np.float32)

        return {
            "obs": np.stack(self.obs, axis=0).astype(np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "returns": np.asarray(self.returns, dtype=np.float32),
            "advantages": np.asarray(self.advantages, dtype=np.float32),
            "states": np.stack(self.states, axis=0).astype(np.float32),
            "masks": masks_np if masks_np is not None else np.ones((len(self.obs), 1), dtype=np.float32),
        }
