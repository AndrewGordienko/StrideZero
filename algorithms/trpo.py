import torch
import torch.optim as optim
import numpy as np
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from config.hyperparameters import AGENT
import jax

class Agent:
    def __init__(self, input_shape, action_shape, device):
        self.actor = ActorNetwork(input_shape, action_shape).to(device)
        self.critic = CriticNetwork(input_shape).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=AGENT["ACTOR_LR"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=AGENT["CRITIC_LR"])

        self.gamma = AGENT["GAMMA"]
        self.lambda_ = AGENT["LAMBDA"]
        self.entropy_coef = AGENT["ENTROPY_COEF_INIT"]
        self.entropy_coef_decay = AGENT["ENTROPY_COEF_DECAY"]
        self.a_optim_batch_size = AGENT["BATCH_SIZE"]
        self.c_optim_batch_size = AGENT["BATCH_SIZE"]
        self.device = AGENT["DEVICE"]
        self.n_epochs = AGENT["N_EPOCHS"]

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

    def train(self):
        torch.cuda.empty_cache()
        self.entropy_coef *= self.entropy_coef_decay
 
