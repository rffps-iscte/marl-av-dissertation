import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from main.QMIX.networks import QNetwork, MixingNetwork


class QMIX:
    def __init__(self, obs_dim: int, act_dim: int, num_agents: int, global_state_dim: int,
                 gamma: float = 0.99, lr: float = 1e-4, device: str = "cpu",
                 mixer_hidden: int = 32, grad_clip: float = 1.0, tau: float = 0.005,
                 hard_update_interval: int = 0, reward_scale: float = 1.0):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.num_agents = int(num_agents)
        self.global_state_dim = int(global_state_dim)
        self.gamma, self.grad_clip, self.tau = float(gamma), float(grad_clip), float(tau)
        self.reward_scale = float(reward_scale)
        self.device = torch.device(device)
        self.hard_update_interval = hard_update_interval
        self.update_count = 0

        self.shared_q = QNetwork(obs_dim, act_dim).to(self.device)
        self.shared_tq = QNetwork(obs_dim, act_dim).to(self.device)
        self.shared_tq.load_state_dict(self.shared_q.state_dict())
        self.shared_tq.eval()

        self.mixer = MixingNetwork(self.num_agents, self.global_state_dim, hidden_dim=mixer_hidden).to(self.device)
        self.target_mixer = MixingNetwork(self.num_agents, self.global_state_dim, hidden_dim=mixer_hidden).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        self.optimizer = optim.Adam(list(self.shared_q.parameters()) + list(self.mixer.parameters()),
                                    lr=lr, eps=1e-5)

    def make_agent(self):
        from main.QMIX.agent import QMIXAgent
        return QMIXAgent(self.shared_q, self.shared_tq, self.act_dim, device=str(self.device))

    def _hard_update_targets(self):
        self.shared_tq.load_state_dict(self.shared_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _soft_update_targets(self):
        for tgt, src in zip(self.shared_tq.parameters(), self.shared_q.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)
        for tgt, src in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)

    def update(self, replay_buffer) -> dict:
        if len(replay_buffer) < replay_buffer.batch_size:
            return {}

        batch = replay_buffer.sample()
        S = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        nS = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        A = torch.as_tensor(batch["action"], dtype=torch.long, device=self.device)
        R = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        D = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        Gs = torch.as_tensor(batch["global_state"], dtype=torch.float32, device=self.device)
        nGs = torch.as_tensor(batch["next_global_state"], dtype=torch.float32, device=self.device)
        nMK = torch.as_tensor(batch["next_mask"], dtype=torch.float32, device=self.device) if "next_mask" in batch else None

        B, N, _ = S.shape
        present = (S.abs().sum(dim=2) > 0)

        S_flat = S.view(B * N, -1)
        Q_all = self.shared_q(S_flat).view(B, N, -1)
        q_taken = Q_all.gather(2, A).squeeze(2) * present.float()

        nS_flat = nS.view(B * N, -1)
        with torch.no_grad():
            Q_next_online = self.shared_q(nS_flat).view(B, N, -1)
            Q_next_target = self.shared_tq(nS_flat).view(B, N, -1)
            if nMK is not None:
                Q_next_online = Q_next_online.masked_fill(~nMK.bool(), -1e9)
            a_star = Q_next_online.argmax(dim=2, keepdim=True)
            # next_present zeros q_next for absent / terminal agents (whose
            # next_state is zero), so terminal agents contribute nothing to
            # the bootstrap target via the mixer.
            next_present = (nS.abs().sum(dim=2) > 0).float()
            q_next = Q_next_target.gather(2, a_star).squeeze(2) * next_present

        Q_tot = self.mixer(q_taken, Gs)
        with torch.no_grad():
            Q_tot_target = self.target_mixer(q_next, nGs)

        # Joint reward = MEAN of per-agent rewards over present agents.
        # Using the SUM made the target scale with the (fluctuating) number of
        # present agents and let the small negative penalties of ~180
        # circulating agents drown the goal_reward of the few arriving each
        # decision; the mean keeps a stable per-capita scale.
        n_present = present.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        R_tot = (R * present.float()).sum(dim=1, keepdim=True) / n_present / self.reward_scale

        # No (1 - D_max) gate: the per-agent presence mask on q_next already
        # zeroes terminal agents' contribution to Q_tot_target. A D_max gate
        # would wrongly zero the bootstrap for the WHOLE sample whenever a
        # single agent terminated.
        target = R_tot + self.gamma * Q_tot_target
        target = target.clamp(-50.0, 50.0)

        loss = F.huber_loss(Q_tot, target.detach(), delta=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(self.shared_q.parameters()) + list(self.mixer.parameters()),
            self.grad_clip)
        self.optimizer.step()

        # Soft target updates only (hard_update_interval defaults to 0).
        self.update_count += 1
        if self.hard_update_interval > 0 and self.update_count % self.hard_update_interval == 0:
            self._hard_update_targets()
        else:
            self._soft_update_targets()

        return {"td_loss": float(loss.item()), "grad_norm": float(total_norm)}