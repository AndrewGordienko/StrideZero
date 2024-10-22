import torch
import torch.nn as nn
import numpy as np
import jax
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork

class Agent:
    def __init__(self, input_shape, action_shape, device):
        self.actor = ActorNetwork(input_shape, action_shape).to(device)
        self.critic = CriticNetwork(input_shape).to(device)
        self.device = device

    def choose_action(self, states):
        if isinstance(states, jax.Array):
            states = np.array(states)
        if not states.flags['WRITEABLE']:
            states = np.copy(states)
        
        states = torch.FloatTensor(states).to(self.device)
        dist = self.actor.get_dist(states)
        actions = dist.sample()

        log_probs = dist.log_prob(actions)
        values = self.critic(states)

        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), values.detach().cpu().numpy()
