# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    actor: nn.Module
    critic: nn.Module
    std: nn.Parameter
    distribution: torch.distributions.Distribution

    def __init__(self,
                 num_actions,
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, *observations, **kwargs):
        mean = self.actor(*observations)
        self.distribution = Normal(mean, self.std)

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    def act(self, *observations, **kwargs):
        self.update_distribution(*observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, *observations, **kwargs):
        actions_mean = self.actor(*observations)
        return actions_mean

    def evaluate(self, *critic_observations, **kwargs):
        value = self.critic(*critic_observations)
        return value


class PrivilegedActorCritic(ActorCritic):
    """
    Symmetrical privileged actor-critic.
    """

    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=None,
                 critic_hidden_dims=None,
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):

        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if kwargs:
            print("PrivilegedActorCritic.__init__ got unexpected args, ignoring: " + str([k for k in kwargs.keys()]))
        super(PrivilegedActorCritic, self).__init__(num_actions, init_noise_std)

        activation = get_activation(activation)
        self.num_actor_obs = num_obs

        self.actor = build_MLP(in_dim=num_obs, out_dim=num_actions,
                               activation=activation, hidden_dim=actor_hidden_dims)

        self.critic = build_MLP(in_dim=num_obs, out_dim=1,
                                activation=activation, hidden_dim=critic_hidden_dims)



def build_MLP(in_dim: int, out_dim: int, activation, final_activation=False, hidden_dim=None) -> nn.Sequential:
    """
    Build an MLP policy with single activation function type according to given parameters
    :param in_dim: input dimension
    :param out_dim: out put dimension
    :param activation: activation function. Should be a nn.Module.
    :param final_activation: bool. True: put an activation layer after output
    :param hidden_dim: list of hidden layer dimensions
    :return: nn.Sequential
    """
    if hidden_dim is None:
        hidden_dim = []
    layer_dims = [in_dim, *hidden_dim, out_dim]
    layers = []
    for l in range(len(hidden_dim) + 1):
        layers.append(nn.Linear(layer_dims[l], layer_dims[l + 1]))
        if l < len(hidden_dim) or final_activation:
            layers.append(activation)
    return nn.Sequential(*layers)


def get_activation(act_name: str):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
