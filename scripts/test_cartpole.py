import torch
import numpy as np
from envs import create_continuous_cartpole_env
from algorithms.trpo import Agent
from utils.logger import info, success, step

if __name__ == "__main__":
    step("Initializing the environment")
    env = create_continuous_cartpole_env()
    obs = env.reset()
    info(f"Initial observation: {obs}")

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    step("Initializing the agent")
    agent = Agent(input_dims, n_actions, "cuda")
    info("Agent initialized with input dimensions and number of actions")

    step("Starting training loop")

    # Training hyperparameters
    num_episodes = 100  # Number of training episodes
    max_steps_per_episode = 500  # Max steps per episode

    for episode in range(num_episodes):
        done = False
        obs = env.reset()  # Reset environment for a new episode
        total_reward = 0

        for step_num in range(max_steps_per_episode):
            action, log_prob, value = agent.choose_action(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Interact with environment
            obs_next, reward, done, _ = env.step(action[0])  # Taking action
            total_reward += reward
            # Store transition in buffer (with dw set to False for now)
            agent.buffer.add(np.expand_dims(obs, axis=0), action, reward, log_prob, value, np.expand_dims(obs_next, axis=0), done, False)

            obs = obs_next

            if done:
                break

        info(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

        # After each episode, train the agent
        agent.train()

    success("Training complete")
    env.close()
    success("Environment closed")
