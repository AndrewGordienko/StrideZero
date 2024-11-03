from envs.modified_biped import BipedalWalker
import numpy as np

"""
if __name__ == "__main__":
    print("hello world")
    env = BipedalWalker(render_mode="human")
    env.reset()
    print(env.reset())
    steps = 0
    total_reward = 0
    a = np.random.uniform(-1, 1, 6)  # Random values between -1 and 1 for each action element
    # Heurisic: suboptimal, have no notion of balance.
    while True:
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
            print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
            print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
        steps += 1

        a = np.random.uniform(-1, 1, 6)  # Random values between -1 and 1 for each action element

        if terminated or truncated:
            break
"""

import optuna
import torch
import numpy as np
import gymnasium as gym
from algorithms.ppo import Agent
from utils.logger import info, success, step
from rich.table import Table
from rich.live import Live
from rich import box
from rich.console import Console
from envs.modified_biped import BipedalWalker

# Define the environment setup function
def create_bipedal_walker_env():
    # env = gym.make("BipedalWalker-v3")
    env = BipedalWalker(render_mode=None)
    return env

# Objective function for Optuna optimization
def objective(trial):
    # Define the hyperparameter search space
    AGENT = {
        "ACTOR_LR": trial.suggest_float("ACTOR_LR", 1.2e-5, 1.5e-5, log=True),
        "CRITIC_LR": trial.suggest_float("CRITIC_LR", 1e-4, 1.2e-4, log=True),
        "ENTROPY_COEF_INIT": trial.suggest_float("ENTROPY_COEF_INIT", 0.08, 0.1),
        "ENTROPY_COEF_DECAY": trial.suggest_float("ENTROPY_COEF_DECAY", 0.982, 0.984),
        "GAMMA": trial.suggest_float("GAMMA", 0.96, 0.97),
        "LAMBDA": trial.suggest_float("LAMBDA", 0.87, 0.89),
        "KL_DIV_THRESHOLD": trial.suggest_float("KL_DIV_THRESHOLD", 0.0083, 0.0087),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [256, 512]),
        "CLIP_RATIO": trial.suggest_float("CLIP_RATIO", 0.27, 0.29),
        "ENTROPY_COEF": trial.suggest_float("ENTROPY_COEF", 0.0020, 0.0025),
        "VALUE_LOSS_COEF": trial.suggest_float("VALUE_LOSS_COEF", 0.45, 0.5),
        "UPDATE_EPOCHS": trial.suggest_int("UPDATE_EPOCHS", 5, 6),
        "MAX_GRAD_NORM": trial.suggest_float("MAX_GRAD_NORM", 0.22, 0.23)
    }

    N_STEPS = 2048
    num_episodes = 1000  # Reduced for faster trials; adjust as needed
    max_steps_per_episode = 1600
    best_reward = -float('inf')
    scroll_limit = 10

    # Initialize the environment
    step("Initializing the environment")
    env = create_bipedal_walker_env()
    obs = env.reset()
    info(f"Initial observation: {obs}")

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    # print("---")
    # print(input_dims)
    # print(n_actions)
    # Initialize the agent
    step("Initializing the agent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, AGENT)
    info("Agent initialized with input dimensions and number of actions")

    # Console setup for live display
    console = Console()
    recent_rows = []

    # Display hyperparameters in a table
    hyperparam_table = Table(
        title="Hyperparameters",                    
        title_style="bold green",           
        border_style="white",
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
    )
    hyperparam_table.add_column("Parameter", justify="center")                  
    hyperparam_table.add_column("Value", justify="center")
    for param, value in AGENT.items():
        hyperparam_table.add_row(param, str(value))
    console.print(hyperparam_table)

    step("Starting training loop")
    all_reward = 0
    total_steps = 0

    # Training loop with live display
    with Live(console=console, refresh_per_second=4) as live:
        for episode in range(num_episodes):
            done = False
            obs = env.reset()[0]
            # print("obs")
            # print(obs)
            total_reward = 0
            step_count = 0
            policy_loss, value_loss, entropy = 0, 0, 0

            while not done and step_count < max_steps_per_episode:
                action, log_prob, value = agent.choose_action(obs)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action = action[0]
                # Interact with the environment
                obs_next, reward, done, _, info_env = env.step(action)
                dw_flags = info_env.get('TimeLimit.truncated', False) if info_env else False
                agent.buffer.add(
                    obs,
                    action,
                    reward,
                    log_prob,
                    value,
                    obs_next,
                    done,
                    dw_flags
                )

                obs = obs_next
                total_reward += reward
                step_count += 1
                total_steps += 1

                # Train at the end of each episode or if the buffer is full
                if done or agent.buffer.size() >= agent.batch_size:
                    policy_loss, value_loss, entropy = agent.train()

            # Update the best reward if this episode's total reward is higher
            if total_reward > best_reward:
                best_reward = total_reward
            all_reward += total_reward

            # Update recent rows for dynamic table display
            row_data = [
                f"[white]{episode + 1}[/white]",
                f"[white]{total_reward:.2f}[/white]",
                f"[green]{best_reward:.2f}[/green]",
                f"[white]{policy_loss:.4f}[/white]",
                f"[white]{value_loss:.4f}[/white]",
                f"[white]{entropy:.4f}[/white]",
                f"[white]{all_reward/(episode+1):.4f}[/white]",
                f"[white]{total_steps:.4f}[/white]",
            ]
            recent_rows.append(row_data)
            if len(recent_rows) > scroll_limit:
                recent_rows.pop(0)

            # Rebuild the table for display
            table = Table(
                title=f"Training Progress for {num_episodes} Episodes",
                title_style="bold cyan",
                border_style="white",
                header_style="bold white",
                box=box.SIMPLE_HEAVY,
            )
            table.add_column("Episode", justify="center")
            table.add_column("Total Reward", justify="center")
            table.add_column("Best Reward", justify="center")
            table.add_column("Policy Loss", justify="center")
            table.add_column("Value Loss", justify="center")
            table.add_column("Entropy", justify="center")
            table.add_column("Average Reward", justify="center")
            table.add_column("Total Steps", justify="center")
            for row in recent_rows:
                table.add_row(*row)
            live.update(table)

    env.close()
    success("Environment closed")

    # Return the average reward as the objective value for Optuna
    return all_reward / num_episodes

# Run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Adjust n_trials as desired

# Display the best parameters
print("Best hyperparameters:", study.best_params)

