import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Environment setup
env = gym.make('BipedalWalker-v3', render_mode="human")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPISODES = 1001
MEM_SIZE = 1000000
BATCH_SIZE = 5
GAMMA = 0.99
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
LEARNING_RATE = 0.01
FC1_DIMS = 1024
FC2_DIMS = 512

best_reward = float("-inf")
average_reward = 0
episode_number = []
average_reward_number = []

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, action_space)
        self.log_std = nn.Parameter(torch.ones(1, action_space) * 0.01)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# PPO Memory
class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.probs, self.vals = [], [], []
        self.actions, self.rewards, self.dones = [], [], []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        n_states = len(self.states)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batch_start = np.arange(0, n_states, self.batch_size)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def clear_memory(self):
        self.states, self.probs, self.actions = [], [], []
        self.rewards, self.dones, self.vals = [], [], []

# Agent
class Agent:
    def __init__(self, n_actions, input_dims):
        self.gamma, self.policy_clip = 0.99, 0.2
        self.n_epochs, self.gae_lambda = 4, 0.95
        self.actor = ActorNetwork(input_dims, n_actions)
        self.critic = CriticNetwork(input_dims)
        self.memory = PPOMemory(BATCH_SIZE)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(DEVICE)
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(state).squeeze()
        return action.cpu().numpy()[0], log_prob.item(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()
            advantage = np.zeros(len(rewards), dtype=np.float32)
            values = vals

            # Calculate advantage using GAE
            for t in range(len(rewards) - 1):
                discount, a_t = 1, 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(DEVICE)
            for batch in batches:
                batch_states = torch.tensor(states[batch], dtype=torch.float).to(DEVICE)
                batch_old_probs = torch.tensor(old_probs[batch]).to(DEVICE)
                batch_actions = torch.tensor(actions[batch]).to(DEVICE)

                dist = self.actor(batch_states)
                critic_value = self.critic(batch_states).squeeze()
                new_probs = dist.log_prob(batch_actions).sum(axis=-1)
                prob_ratio = new_probs.exp() / batch_old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()

                returns = (torch.tensor(advantage[batch], dtype=torch.float32, device=DEVICE) + torch.tensor(values[batch], dtype=torch.float32, device=DEVICE))
                critic_loss = F.mse_loss(returns.view_as(critic_value), critic_value)
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

agent = Agent(env.action_space.shape[0], env.observation_space.shape[0])
episode_number, average_reward_number = [], []
best_reward, average_reward = float("-inf"), 0

# Training Loop
for i in range(1, EPISODES):
    observation, info = env.reset()
    score, done, step = 0, False, 0

    while not done:
        # env.render()
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        score += reward
        agent.memory.store_memory(observation, action, prob, val, reward, done)

        if step % 20 == 0:
            agent.learn()

        observation = observation_
        step += 1

    if score > best_reward:
        torch.save(agent.actor.state_dict(), 'agent_actor.pth')
        best_reward = score
    average_reward += score
    episode_number.append(i)
    average_reward_number.append(average_reward / i)
    print(f"Episode {i}, Avg Reward {average_reward / i:.2f}, Best Reward {best_reward:.2f}, Last Reward {score:.2f}")

plt.plot(episode_number, average_reward_number)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Training Progress")
plt.show()

