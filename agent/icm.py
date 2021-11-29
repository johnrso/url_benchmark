import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import copy

from agent.ddpg import DDPGAgent


class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.forward_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim))

        self.backward_net = nn.Sequential(nn.Linear(2 * obs_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, action_dim),
                                          nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        #import ipdb; ipdb.set_trace()
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class ICMAgent(DDPGAgent):
    def __init__(self, icm_scale, update_encoder, **kwargs):
        super().__init__(**kwargs)
        self.icm_scale = icm_scale
        self.update_encoder = update_encoder

        self.icm = ICM(self.obs_dim, self.action_dim,
                       self.hidden_dim).to(self.device)

        # optimizers
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(),
                                              lr=self.lr)

        self.online_predictor = MLP(self.encoder.repr_dim, 
                                    self.encoder.repr_dim, 
                                    1024).to(self.device)
        
        self.predictor_opt = torch.optim.Adam(self.online_predictor.parameters(),
                                              lr=self.lr)

        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        set_requires_grad(self.target_encoder, False)

        self.icm.train()

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        forward_error, _ = self.icm(obs, action, next_obs)

        reward = forward_error * self.icm_scale
        reward = torch.log(reward + 1.0)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        orig_obs = obs
        obs = self.aug_and_encode(obs)

        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(
                self.update_icm(obs.detach(), action, next_obs.detach(), step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs,
                                                       step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            
        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()
            obs_byol = obs_byol.detach()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update encoder
        metrics.update(self.update_byol(orig_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def byol_forward(self, x): 
        x1, x2 = self.aug(x), self.aug(x)

        online_proj_one = self.encoder(x1)
        online_proj_two = self.encoder(x2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_proj_one = self.target_encoder(x1).detach()
            target_proj_two = self.target_encoder(x2).detach()

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        loss = loss_one + loss_two
        return x1, loss.mean()

    def update_byol(self, obs):
        _, loss = self.byol_forward(obs)
        self.encoder_opt.zero_grad()
        self.predictor_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()
        self.predictor_opt.step()
        
        update_moving_average(EMA(.99), self.target_encoder, self.encoder)

        return {"repr_loss": loss.item()}

