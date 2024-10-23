import torch
import torch.nn as nn
import numpy as np
from config.hyperparameters import ACTOR_NETWORK

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, action_std_init=0.5):
        super(ActorNetwork, self).__init__()
        FC1_DIMS = ACTOR_NETWORK['FC1_DIMS']
        FC2_DIMS = ACTOR_NETWORK['FC2_DIMS']

        # Define the network layers
        self.fc1 = nn.Linear(input_dims, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc_mu = nn.Linear(FC2_DIMS, n_actions)

        # Initialize log_std with a reasonable value to avoid instability
        self.log_std = nn.Parameter(torch.ones(1, n_actions) * np.log(1.0))  # Set higher std for more exploration

    def forward(self, state):
        # Clamp input state to avoid extreme values and ensure numerical stability
        state = torch.clamp(state, -10, 10)

        # Forward pass through the network
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))

       # Clamp mu to avoid extreme values
        mu = torch.clamp(mu, -20, 20)
        return mu

    def get_dist(self, state):
        # Get the action distribution for a given state
        mu = self.forward(state)
        std = torch.exp(self.log_std)

        # Clamp std to prevent too small or NaN values
        std = torch.clamp(std, min=1e-3)

        # Debugging: Check for NaNs in mu or std
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print("NaN detected in mu or std!")
            print(f"mu: {mu}")
            print(f"std: {std}")
            raise ValueError("NaNs detected in actor network!")

        # Return the Normal distribution for the actions
        dist = torch.distributions.Normal(mu, std)
        return dist

    def get_log_prob(self, states, actions):
        # Get the log probabilities of the actions taken in given states
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions
        return log_probs
 
