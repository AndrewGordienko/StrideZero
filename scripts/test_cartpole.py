import torch
import numpy as np
from envs import create_continuous_cartpole_env
from algorithms.trpo import Agent

if __name__ == "__main__":
    env = create_continuous_cartpole_env()
    obs = env.reset()

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    agent = Agent(input_dims, n_actions, "cuda")

    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(obs)[0][0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

