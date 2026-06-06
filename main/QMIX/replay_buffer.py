import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, batch_size, state_dim, action_dim, n_agents):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def add(self, global_state, state, action, reward, next_state, next_global_state, done, active_agents, next_mask=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        rec = {"global_state": global_state, "state": state, "action": action, "reward": reward,
               "next_state": next_state, "next_global_state": next_global_state, "done": done,
               "active_agents": active_agents}
        if next_mask is not None:
            rec["next_mask"] = next_mask
        self.buffer[self.position] = rec
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        return {k: np.array([b[k] for b in batch]) for k in batch[0].keys()}

    def __len__(self):
        return len(self.buffer)