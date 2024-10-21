from envs import create_continuous_cartpole_env

if __name__ == "__main__":
    env = create_continuous_cartpole_env()
    obs = env.reset()
    
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

