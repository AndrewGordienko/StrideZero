import torch
import numpy as np
import optuna
from envs import create_continuous_cartpole_env
from algorithms.ppo import Agent
from utils.logger import info, success, step
from rich.table import Table
from rich.live import Live
from rich import box
from rich.console import Console

# Fixed hyperparameters
AGENT = {
    "ACTOR_LR": 0.00021703891689103257,
    "CRITIC_LR": 0.0004271253752630734,
    "ENTROPY_COEF_INIT": 0.08801608152663996,
    "ENTROPY_COEF_DECAY": 0.9958697832658145,
    "GAMMA": 0.9908654911298033,
    "LAMBDA": 0.9580440174249956,
    "KL_DIV_THRESHOLD": 0.006302958474657382,
    "BATCH_SIZE": 512,
    "CLIP_RATIO": 0.27307659244433324,
    "ENTROPY_COEF": 0.003919224323254069,
    "VALUE_LOSS_COEF": 0.5820465065839804,
    "UPDATE_EPOCHS": 8,
    "MAX_GRAD_NORM": 0.33127351532987814
}

def objective():
    # Initialize the environment
    step("Initializing the environment")
    env = create_continuous_cartpole_env()
    obs = env.reset()
    info(f"Initial observation: {obs}")

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # Initialize the agent with fixed hyperparameters
    step("Initializing the agent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, AGENT)

    # Display hyperparameters
    console = Console()
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

    # Training parameters
    num_episodes = 5000
    max_steps_per_episode = 500
    best_reward = -float('inf')
    scroll_limit = 10

    # Initialize the table for dynamic training updates
    recent_rows = []
    step("Starting training loop")

    # Run training loop
    total_reward_sum = 0
    with Live(console=console, refresh_per_second=4) as live:
        for episode in range(num_episodes):
            done = False
            obs = env.reset()
            total_reward = 0

            while not done:
                action, log_prob, value = agent.choose_action(obs)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action = action[0]
                # Interact with the environment
                obs_next, reward, done, info_env = env.step(action)
                dw_flags = info_env.get('TimeLimit.truncated', False)
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

                if done or agent.buffer.size() >= agent.batch_size:
                    policy_loss, value_loss, entropy = agent.train()

            # Update best reward
            if total_reward > best_reward:
                best_reward = total_reward

            # Update recent rows for dynamic table display
            row_data = [
                f"[white]{episode + 1}[/white]",
                f"[white]{total_reward:.2f}[/white]",
                f"[green]{best_reward:.2f}[/green]",
                f"[white]{policy_loss:.4f}[/white]",
                f"[white]{value_loss:.4f}[/white]",
                f"[white]{entropy:.4f}[/white]",
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
            for row in recent_rows:
                table.add_row(*row)
            live.update(table)

            # Track cumulative reward for evaluation
            total_reward_sum += total_reward

    env.close()
    success("Environment closed")

    # Return the negative cumulative reward to match the Optuna objective
    return -total_reward_sum

# Run the Optuna study with only one trial to ensure the fixed hyperparameters are used
objective()
 
