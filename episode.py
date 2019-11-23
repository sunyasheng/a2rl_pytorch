import numpy as np
import torch
import torch.nn.functional as F

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cuda'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = []
        self._actions_list = []
        self._rewards_list = []
        self._h_list = []
        self._c_list = []
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self._hiddens = None

    @property
    def hiddens(self):
        if self._hiddens is None:
            bs, hidden_dim = self._h_list[0][0].size()
            hiddens = torch.zeros(len(self), self.batch_size, hidden_dim).to(self._h_list[0][0])
            ciddens = torch.zeros(len(self), self.batch_size, hidden_dim).to(self._h_list[0][0])
            
            for i in range(self.batch_size):
                length = len(self._h_list[i])
                hiddens[:length, i] = torch.cat(self._h_list[i])
                ciddens[:length, i] = torch.cat(self._c_list[i])
            self._hiddens = (hiddens, ciddens)
        return self._hiddens

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        # values = values.squeeze(2).detach()
        values = values.detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observations_np, actions_np, rewards_np, hs_ts, cs_ts):
        self._observations_list.append(observations_np)
        self._actions_list.append(actions_np)
        self._rewards_list.append(rewards_np)
        self._h_list.append(hs_ts)
        self._c_list.append(cs_ts)

    def __len__(self):
        return max(map(len, self._rewards_list))
