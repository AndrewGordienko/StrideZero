import torch
import numpy as np
from envs import create_continuous_cartpole_env
from models.actor_network import ActorNetwork

if __name__ == "__main__":
    env = create_continuous_cartpole_env()
    obs = env.reset()

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = ActorNetwork(input_dims, n_actions)
    actor.eval()

    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = actor.get_dist(obs_tensor)
        action = dist.sample()
        print(action)        
        # Ensure the action is the correct shape (1,)
        action_np = action.detach().numpy().flatten()

        obs, reward, done, _ = env.step(action_np)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

