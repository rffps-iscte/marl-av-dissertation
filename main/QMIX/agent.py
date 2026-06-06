import random
import numpy as np
import torch


class QMIXAgent:
    def __init__(self, shared_q, shared_tq, action_dim: int, device: str = "cpu"):
        self.q = shared_q
        self.tq = shared_tq
        self.action_dim = action_dim
        self.device = device

    @torch.no_grad()
    def select_action(self, state, epsilon: float = 0.1, mask=None) -> int:
        if random.random() < epsilon:
            if mask is not None and np.any(mask):
                return int(np.random.choice(np.nonzero(mask)[0]))
            return random.randrange(self.action_dim)

        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s).squeeze(0)
        if mask is not None:
            q = q.masked_fill(~torch.as_tensor(mask, dtype=torch.bool, device=self.device), -1e9)
        return int(torch.argmax(q).item())