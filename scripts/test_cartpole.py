import torch
import numpy as np
from envs import create_continuous_cartpole_env
from algorithms.trpo import Agent
from utils.logger import info, success, warning, error, debug, step

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

    done = False
    total_reward = 0
    step("Starting environment loop")

    while not done:
        action = agent.choose_action(obs)[0][0]
        # debug(f"Chosen action: {action}")
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        # debug(f"New observation: {obs}, Reward: {reward}, Done: {done}")

    success(f"Total reward: {total_reward}")
    env.close()
    success("Environment closed")

