import time
import torch
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from rich.table import Table
from rich.live import Live
from rich import box
from rich.console import Console
from algorithms.ppo import Agent  # Import your Agent class

# Check if CUDA is available and print the device being used
device = jax.devices()[0]
print(f"Using device: {device.device_kind}")
torch_device = torch.device("cuda")
print(f"Using torch device: {torch_device}")

# Define the environment setup function
def create_parallel_bipedal_walker_envs(num_envs=10):
    def make_env():
        return gym.make("BipedalWalker-v3")

    # Create multiple environments for parallel processing
    envs = SyncVectorEnv([make_env for _ in range(num_envs)])
    return envs

# Set up hyperparameters
HYPERPARAMS = {
    "ACTOR_LR": 3e-4,
    "CRITIC_LR": 3e-4,
    "GAMMA": 0.999,
    "LAMBDA": 0.95,
    "BATCH_SIZE": 64,
    "UPDATE_EPOCHS": 10,
    "CLIP_RATIO": 0.18,
    "ENTROPY_COEF": 0.0,
    "VALUE_LOSS_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "NUM_ENVS": 5,  # Parallel environments
    "NUM_EPISODES": 2000,
    "MAX_STEPS_PER_EPISODE": 1600,
}

NUM_ENVS = 5
UPDATE_THRESHOLD = 2048

# Initialize environments
console = Console()
envs = create_parallel_bipedal_walker_envs(HYPERPARAMS["NUM_ENVS"])
obs = envs.reset()

input_dims = envs.single_observation_space.shape[0]
n_actions = envs.single_action_space.shape[0]

# Initialize agent
agent = Agent(input_dims, n_actions, torch_device, HYPERPARAMS)  # Ensure this matches Agent's init

# Training parameters
num_episodes = HYPERPARAMS["NUM_EPISODES"]
max_steps_per_episode = HYPERPARAMS["MAX_STEPS_PER_EPISODE"]
best_reward = -float('inf')
scroll_limit = 10

# Display hyperparameters in a table
hyperparam_table = Table(
    title="Hyperparameters", title_style="bold green",
    border_style="white",
    header_style="bold white",
    box=box.SIMPLE_HEAVY,
)
hyperparam_table.add_column("Parameter", justify="center")
hyperparam_table.add_column("Value", justify="center")
for param, value in HYPERPARAMS.items():
    hyperparam_table.add_row(param, str(value))
console.print(hyperparam_table)
print("Starting training loop with 5 parallel environments")
all_reward = 0
total_steps = 0

# Training loop with live display
with Live(console=console, refresh_per_second=4) as live:
    recent_rows = []
    for episode in range(num_episodes):
        done_flags = jnp.zeros(HYPERPARAMS["NUM_ENVS"], dtype=bool)
        obs = envs.reset()[0]
        total_rewards = jnp.zeros(HYPERPARAMS["NUM_ENVS"])
        step_count = 0
        start_time = time.time()

        while not jnp.all(done_flags) and step_count < max_steps_per_episode:
            # Generate random actions for each environment in parallel
            active_envs = ~done_flags  # Boolean array indicating active environments
            actions, log_probs, values = agent.choose_action_jax(obs)

            actions = jnp.where(active_envs[:, None], actions, 0.0)
            log_probs = jnp.where(active_envs, log_probs, 0.0)
            values = jnp.where(active_envs, values, 0.0)

            # Interact with all environments in parallel
            obs_next, rewards, dones, _, info_envs = envs.step(actions)
            
            # Calculate which environments are done based on terminations and truncations
            # dw_flags = [(info.get('TimeLimit.truncated', False) if info else False) for info in info_envs]
            dw_flags = [False] * HYPERPARAMS["NUM_ENVS"]
            for env_idx, info in enumerate(info_envs):
                if info and info.get('TimeLimit.truncated', False):
                    dw_flags[env_idx] = True
            
            # Loop over each environment and add individual transitions
            for env_idx in range(HYPERPARAMS["NUM_ENVS"]):
                # print(dw_flags[env_idx])
                # print(dones[env_idx])
                agent.buffer.add(
                    obs[env_idx],              # State for env_idx (reshaped to 2D)
                    actions[env_idx],          # Action for env_idx (reshaped to 2D)
                    rewards[env_idx],                 # Reward for env_idx (1D)
                    log_probs[env_idx],               # Log probability for env_idx (1D)
                    values[env_idx][0],                  # Value for env_idx (1D)
                    obs_next[env_idx],         # Next state for env_idx (shaped to 2D)
                    dones[env_idx],                   # Done flag for env_idx (1D)
                    dw_flags[env_idx]                 # DW flag for env_idx (1D)
                )
            
            # Update the done flags and accumulated rewards
            total_rewards += rewards * (1 - done_flags)
            done_flags = jnp.logical_or(done_flags, dones)
            obs = obs_next
            total_steps += HYPERPARAMS["NUM_ENVS"]
            if total_steps % UPDATE_THRESHOLD == 0:
                policy_loss, value_loss, entropy = agent.train()

        # Update best reward if needed
        max_reward = total_rewards.max()
        avg_reward = total_rewards.mean()
        if max_reward > best_reward:
            best_reward = max_reward
        all_reward += avg_reward
        end_time = time.time()
        total_time = end_time - start_time

        # Update recent rows for dynamic table display
        row_data = [
            f"[white]{episode + 1}[/white]",
            f"[white]{avg_reward:.2f}[/white]",
            f"[blue]{max_reward:.2f}[/blue]",
            f"[green]{best_reward:.2f}[/green]",
            f"[white]{all_reward / (episode + 1):.2f}[/white]",
            f"[white]{total_steps:.2f}[/white]",
            f"[white]{total_time:.2f}[/white]",
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
        table.add_column("Average Reward", justify="center")
        table.add_column("Max Reward", justify="center")
        table.add_column("Best Reward", justify="center")
        table.add_column("Cumulative Avg Reward", justify="center")
        table.add_column("Total Steps", justify="center")
        table.add_column("Total Time", justify="center")
        for row in recent_rows:
            table.add_row(*row)
        live.update(table)

envs.close()
print("Environment closed")

