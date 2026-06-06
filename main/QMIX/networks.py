import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.network(x)


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.hyper_w_1 = nn.Sequential(nn.Linear(state_dim, hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(hidden_dim * 4, n_agents * hidden_dim))
        self.hyper_w_2 = nn.Sequential(nn.Linear(state_dim, hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(hidden_dim * 4, hidden_dim * 1))
        self.hyper_b_1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b_2 = nn.Sequential(nn.Linear(state_dim, hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(hidden_dim * 4, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w_1(states)).view(bs, self.n_agents, -1)
        b1 = self.hyper_b_1(states).view(bs, 1, -1)
        hidden = torch.relu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w_2(states)).view(bs, -1, 1)
        b2 = self.hyper_b_2(states).view(bs, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(bs, 1)