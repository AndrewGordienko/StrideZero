from envs.modified_biped import BipedalWalker
import numpy as np
import optuna
import torch
import gymnasium as gym
from algorithms.ppo import Agent
from utils.logger import info, success, step
from rich.table import Table
from rich.live import Live
from rich import box
from rich.console import Console

# Define the environment setup function
def create_bipedal_walker_env():
    env = BipedalWalker(render_mode=None)
    return env

# Objective function for Optuna optimization
def objective(trial):
    # Define the hyperparameter search space
    AGENT = {
        "ACTOR_LR": trial.suggest_float("ACTOR_LR", 1e-6, 5e-4, log=True),
        "CRITIC_LR": trial.suggest_float("CRITIC_LR", 5e-5, 5e-4, log=True),
        "ENTROPY_COEF_INIT": trial.suggest_float("ENTROPY_COEF_INIT", 0.01, 0.1),
        "ENTROPY_COEF_DECAY": trial.suggest_float("ENTROPY_COEF_DECAY", 0.9, 0.99),
        "GAMMA": trial.suggest_float("GAMMA", 0.92, 0.99),
        "LAMBDA": trial.suggest_float("LAMBDA", 0.85, 0.95),
        "KL_DIV_THRESHOLD": trial.suggest_float("KL_DIV_THRESHOLD", 0.005, 0.02),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [256, 512, 1024, 2048]),
        "CLIP_RATIO": trial.suggest_float("CLIP_RATIO", 0.1, 0.2),
        "ENTROPY_COEF": trial.suggest_float("ENTROPY_COEF", 0.0001, 0.005),
        "VALUE_LOSS_COEF": trial.suggest_float("VALUE_LOSS_COEF", 0.2, 0.8),
        "UPDATE_EPOCHS": trial.suggest_int("UPDATE_EPOCHS", 3, 9),
        "MAX_GRAD_NORM": trial.suggest_float("MAX_GRAD_NORM", 0.1, 0.5)
    }

    N_STEPS = 2048
    num_episodes = 1000
    max_steps_per_episode = 1600
    best_reward = -float('inf')
    best_100_avg = -float('inf')
    reward_last_100 = []
    scroll_limit = 10

    # Initialize the environment
    step("Initializing the environment")
    env = create_bipedal_walker_env()
    obs = env.reset()
    print(len(obs))
    info(f"Initial observation: {obs}")

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    input_dims = 42

    step("Initializing the agent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, AGENT)
    info("Agent initialized with input dimensions and number of actions")
        
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

    with Live(console=console, refresh_per_second=4) as live:
        for episode in range(num_episodes):
            done = False
            obs = env.reset()[0]
            total_reward = 0
            step_count = 0
            policy_loss, value_loss, entropy = 0, 0, 0

            while not done and step_count < max_steps_per_episode:
                action, log_prob, value = agent.choose_action(obs)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action = action[0]

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

                if done or agent.buffer.size() >= agent.batch_size:
                    policy_loss, value_loss, entropy = agent.train()

            # Track rewards for the last 100 episodes
            reward_last_100.append(total_reward)
            if len(reward_last_100) > 100:
                reward_last_100.pop(0)
            avg_last_100 = sum(reward_last_100) / len(reward_last_100)

            # Check for best episode reward and best last 100 average reward
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(agent.actor.state_dict(), f"best_single_reward_model.pth")
            if avg_last_100 > best_100_avg:
                best_100_avg = avg_last_100
                torch.save(agent.actor.state_dict(), f"best_100_avg_model.pth")

            all_reward += total_reward

            # Update recent rows for table display
            row_data = [
                f"[white]{episode + 1}[/white]",
                f"[white]{total_reward:.2f}[/white]",
                f"[green]{best_reward:.2f}[/green]",
                f"[cyan]{best_100_avg:.2f}[/cyan]",
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
            table.add_column("Best 100-Avg", justify="center")
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

    return all_reward / num_episodes

# Run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)
 
