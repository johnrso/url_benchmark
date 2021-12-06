import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit
from copy import deepcopy

import utils

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

@jit.script
def sinkhorn_knopp(Q):
    Q -= Q.max()
    Q = torch.exp(Q).T
    Q /= Q.sum()

    r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
    c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
    for it in range(3):
        u = Q.sum(dim=1)
        u = r / u
        Q *= u.unsqueeze(dim=1)
        Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
    Q = Q / Q.sum(dim=0, keepdim=True)
    return Q.T


class Projector(nn.Module):
    def __init__(self, pred_dim, proj_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(pred_dim, proj_dim), nn.ReLU(),
                                   nn.Linear(proj_dim, pred_dim))

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.trunk(x)

class ProtoAgent(DDPGAgent):
    def __init__(self, pred_dim, proj_dim, queue_size, num_protos,
                 tau, encoder_target_tau, topk, update_encoder, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.encoder_target_tau = encoder_target_tau
        self.topk = topk
        self.num_protos = num_protos
        self.update_encoder = update_encoder

        # models
        self.encoder_target = deepcopy(self.encoder)

        self.icm = ICM(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.icm.apply(utils.weight_init)
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self.predictor = nn.Linear(self.obs_dim, pred_dim).to(self.device)
        self.predictor.apply(utils.weight_init)
        self.predictor_target = deepcopy(self.predictor)

        self.projector = Projector(pred_dim, proj_dim).to(self.device)
        self.projector.apply(utils.weight_init)

        # prototypes
        self.protos = nn.Linear(pred_dim, num_protos,
                                bias=False).to(self.device)
        self.protos.apply(utils.weight_init)

        # candidate queue
        self.queue = torch.zeros(queue_size, pred_dim, device=self.device)
        self.queue_ptr = 0

        # optimizers
        self.proto_opt = torch.optim.Adam(utils.chain(
            self.encoder.parameters(), self.predictor.parameters(),
            self.projector.parameters(), self.protos.parameters()),
                                          lr=self.lr)

        self.icm.train()
        self.predictor.train()
        self.projector.train()
        self.protos.train()

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)
        utils.hard_update_params(other.protos, self.protos)
        if self.init_critic:
            utils.hard_update_params(other.critic, self.critic)

    def normalize_protos(self):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

    def compute_intr_reward(self, obs, action, next_obs, step):
        obs = self.encoder(obs)
        next_obs = self.encoder(next_obs)
        
        forward_error, _ = self.icm(obs, action, next_obs)

        reward = forward_error
        reward = torch.log(reward + 1.0)
        return reward

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        obs = self.encoder(obs)
        next_obs = self.encoder(next_obs)
        
        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics
    
    def update_proto(self, obs, next_obs, step):
        metrics = dict()

        # normalize prototypes
        self.normalize_protos()

        # online network
        s = self.encoder(obs)
        s = self.predictor(s)
        s = self.projector(s)
        s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.tau, dim=1)

        # target network
        with torch.no_grad():
            t = self.encoder_target(next_obs)
            t = self.predictor_target(t)
            t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)
            q_t = sinkhorn_knopp(scores_t / self.tau)

        # loss
        loss = -(q_t * log_p_s).sum(dim=1).mean()
        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()
        self.proto_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.proto_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        with torch.no_grad():
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)

        if self.reward_free:
            metrics.update(self.update_proto(obs, next_obs, step))
            
            # update icm
            metrics.update(self.update_icm(obs, action, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        obs = self.encoder(obs)
        next_obs = self.encoder(next_obs)
        
        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))


        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.encoder, self.encoder_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.predictor, self.predictor_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
