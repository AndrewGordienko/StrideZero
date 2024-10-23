import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

# Policy Network for Discrete Action Space
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        logits = self.fc(state)
        return logits

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.fc(state)

# Collect Trajectories
def collect_trajectories(env, policy, horizon):
    states, actions, rewards = [], [], []
    state = env.reset()

    for _ in range(horizon):
        # Ensure the state is a NumPy array and print its shape for debugging
        state = np.array(state)  # Convert state to a NumPy array if it's a list
        print(f"State shape: {state.shape}")  # Print shape for debugging

        if state.shape[0] != 4:  # Make sure state has the correct length (4 for CartPole)
            raise ValueError(f"Unexpected state shape: {state.shape}")

        # Convert state to a tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  
        states.append(state_tensor)

        # Get action from policy
        logits = policy(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        actions.append(action)

        # Take action in the environment
        next_state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

       # Handle episode termination
        if done:
            state = env.reset()  # Reset environment if episode is done
        else:
            state = next_state  # Continue with next state

    # Return collected trajectories
    return states, actions, rewards
 
# Compute rewards-to-go and advantages
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages, returns = [], []
    advantage = 0
    next_value = 0

    for r in reversed(rewards):
        advantage = r + gamma * next_value - r  # for simplicity, using reward directly
        advantages.insert(0, advantage)
        returns.insert(0, r)
        next_value = r  # assuming value function is simple
    
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)

# Surrogate Loss for TRPO
def surrogate_loss(policy, old_policy, states, actions, advantages):
    log_probs, old_log_probs = [], []
    
    for state, action in zip(states, actions):
        logits = policy(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)

        old_logits = old_policy(state)
        old_dist = Categorical(logits=old_logits)
        old_log_prob = old_dist.log_prob(action)
        old_log_probs.append(old_log_prob)

    log_probs = torch.stack(log_probs)
    old_log_probs = torch.stack(old_log_probs)
    
    ratio = torch.exp(log_probs - old_log_probs)
    return (ratio * advantages).mean()

# KL Divergence Calculation for Trust Region Constraint
def compute_kl(policy, states):
    kl_divs = []
    for state in states:
        logits = policy(state)
        dist = Categorical(logits=logits)

        old_logits = policy(state).detach()
        old_dist = Categorical(logits=old_logits)

        kl_div = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl_divs.append(kl_div)

    return torch.stack(kl_divs).mean()

# Fisher Vector Product
def fisher_vector_product(policy, states, p, damping=0.1):
    kl = compute_kl(policy, states)
    kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([g.view(-1) for g in kl_grad])

    kl_v = (kl_grad_vector * p).sum()
    fisher_product = torch.autograd.grad(kl_v, policy.parameters())
    fisher_product_vector = torch.cat([g.contiguous().view(-1) for g in fisher_product])

    return fisher_product_vector + damping * p

# Conjugate Gradient Descent
def conjugate_gradient(policy, states, b, max_iters=10, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    r_dot_old = torch.dot(r, r)

    for _ in range(max_iters):
        fisher_product = fisher_vector_product(policy, states, p)
        alpha = r_dot_old / torch.dot(p, fisher_product)
        x += alpha * p
        r -= alpha * fisher_product
        r_dot_new = torch.dot(r, r)

        if r_dot_new < residual_tol:
            break
        beta = r_dot_new / r_dot_old
        p = r + beta * p
        r_dot_old = r_dot_new
    
    return x

# Train Value Network
def train_value_function(value_net, states, returns, optimizer):
    value_loss = nn.MSELoss()(value_net(torch.stack(states)).squeeze(), returns)
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()

# TRPO Main Loop
def trpo(env, policy, value_net, iterations, horizon):
    value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)

    for iteration in range(iterations):
        states, actions, rewards = collect_trajectories(env, policy, horizon)
        values = [value_net(state) for state in states]

        advantages, returns = compute_advantages(rewards, values)
        
        # Save old policy
        old_policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        old_policy.load_state_dict(policy.state_dict())

        # Compute policy loss and gradient
        loss = surrogate_loss(policy, old_policy, states, actions, advantages)
        policy_grad = torch.autograd.grad(loss, policy.parameters())
        policy_grad_vector = torch.cat([g.view(-1) for g in policy_grad])

        # Solve for step direction
        step_dir = conjugate_gradient(policy, states, policy_grad_vector)

        # Backtracking line search to ensure KL constraint
        max_kl = 0.01
        for step_size in [0.5, 1.0]:
            update_params = torch.cat([p.view(-1) for p in policy.parameters()]) + step_size * step_dir
            kl = compute_kl(policy, states)
            if kl < max_kl:
                break

        # Update value function
        train_value_function(value_net, states, returns, value_optimizer)

        print(f"Iteration {iteration+1}/{iterations}, Loss: {loss.item()}")

# Initialize environment and networks
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

# Run TRPO
trpo(env, policy, value_net, iterations=50, horizon=1000)

env.close()

