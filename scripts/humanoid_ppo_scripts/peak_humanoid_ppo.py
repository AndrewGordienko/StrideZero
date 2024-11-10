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

# Define the environment setup function
def create_bipedal_walker_env():
    env = gym.make("BipedalWalker-v3")
    return env

# Objective function for Optuna optimization
def objective(trial):
    # Define fixed hyperparameters
    AGENT = {
        "ACTOR_LR": 1.1533070735302821e-05,
        "CRITIC_LR": 0.00011670790938402237,
        "ENTROPY_COEF_INIT": 0.08346871561853314,
        "ENTROPY_COEF_DECAY": 0.9845136718237683,
        "GAMMA": 0.9631902401102875,
        "LAMBDA": 0.8734031304947364,
        "KL_DIV_THRESHOLD": 0.00807602938974265,
        "BATCH_SIZE": 512,
        "CLIP_RATIO": 0.28850409904445223,
        "ENTROPY_COEF": 0.0021244760700221297,
        "VALUE_LOSS_COEF": 0.6071602003509949,
        "UPDATE_EPOCHS": 5,
       "MAX_GRAD_NORM": 0.21728696180119855,
    }

    N_STEPS = 2048
    num_episodes = 10000
    max_steps_per_episode = 1600
    best_reward = -float('inf')
    best_avg_reward = -float('inf')
    reward_history = []

    # Initialize the environment
    step("Initializing the environment")
    env = create_bipedal_walker_env()
    obs = env.reset()
    info(f"Initial observation: {obs}")

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # Initialize the agent
    step("Initializing the agent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, AGENT)
    info("Agent initialized with input dimensions and number of actions")

    # Console setup for live display
    console = Console()
    recent_rows = []
    scroll_limit = 10

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

            # Update best_reward model
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(agent.actor.state_dict(), "best_reward_model.pth")

            # Update reward history and best_avg_reward model
            reward_history.append(total_reward)
            if len(reward_history) >= 100:
                avg_reward_last_100 = np.mean(reward_history[-100:])
                if avg_reward_last_100 > best_avg_reward:
                    best_avg_reward = avg_reward_last_100
                    torch.save(agent.actor.state_dict(), "best_avg_reward_model.pth")

            # Track cumulative rewards
            all_reward += total_reward

            # Update recent rows for dynamic table display
            row_data = [
                f"[white]{episode + 1}[/white]",
                f"[white]{total_reward:.2f}[/white]",
                f"[green]{best_reward:.2f}[/green]",
                f"[cyan]{best_avg_reward:.2f}[/cyan]",
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
            table.add_column("Best Avg Reward (last 100)", justify="center")
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
study.optimize(objective, n_trials=1)  # Adjust n_trials as desired

# Display the best parameters
print("Best hyperparameters:", study.best_params)
 
