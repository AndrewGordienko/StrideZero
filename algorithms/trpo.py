import torch
import torch.optim as optim
import numpy as np
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from buffers.replay_buffer import ReplayBuffer
from config.hyperparameters import AGENT
from rich.table import Table
from rich import box, console
from utils import logger
from rich.console import Console

console = Console()

class Agent:
    def __init__(self, input_shape, action_shape, device):
        self.device = device
        self.actor = ActorNetwork(input_shape, action_shape).to(self.device)
        self.critic = CriticNetwork(input_shape).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=AGENT["ACTOR_LR"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=AGENT["CRITIC_LR"])
        self.buffer = ReplayBuffer()
        self.gamma = AGENT["GAMMA"]
        self.lambda_ = AGENT["LAMBDA"]
        self.entropy_coef = AGENT["ENTROPY_COEF_INIT"]
        self.entropy_coef_decay = AGENT["ENTROPY_COEF_DECAY"]
        self.a_optim_batch_size = AGENT["BATCH_SIZE"]
        self.c_optim_batch_size = AGENT["BATCH_SIZE"]
        self.n_epochs = AGENT["N_EPOCHS"]
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.kl_threshold = 0.005
        self.damping_coeff = 0.1

    def choose_action(self, states):
        states = torch.FloatTensor(states).to(self.device)
        dist = self.actor.get_dist(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(states)
        return (
            actions.detach().cpu().numpy(),
            log_probs.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )

        self.buffer.clear()
        torch.save(self.actor.state_dict(), 'actor_network.pth')

    def compute_gae(self, rewards, values, next_values, dones):
        """Provides an estimate of how much better an action is compared to the average action"""

        T = len(rewards)  # Length of trajectory
        advantages = np.zeros(T)
        returns = np.zeros(T)
    
        # Initialize the last advantage and return to 0 (for terminal state)
        gae = 0

        # Loop backward through the trajectory
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def fit_value_function(self, states, returns):
        states_tensor = torch.Tensor(states).to(self.device)
        returns_tensor = torch.Tensor(returns).to(self.device)    
        self.critic_optimizer.zero_grad()
        values_pred = self.critic(states_tensor).squeeze(-1) 
        value_loss = torch.nn.functional.mse_loss(values_pred, returns_tensor)
    
        value_loss.backward()
        self.critic_optimizer.step()
    
        return value_loss.item()

    def compute_policy_loss(self, states, actions, log_probs_old, advantages):
        states_tensor = torch.Tensor(states).to(self.device)
        actions_tensor = torch.Tensor(actions).to(self.device)
        log_probs_old_tensor = torch.Tensor(log_probs_old).to(self.device)
        advantages_tensor = torch.Tensor(advantages).to(self.device)
    
        log_probs_new = self.actor.get_log_prob(states_tensor, actions_tensor)
        ratios = torch.exp(log_probs_new - log_probs_old_tensor)

        # Calculate entropy for regularization
        dist = self.actor.get_dist(states_tensor)
        entropy = dist.entropy().mean()

        # Include entropy bonus in policy loss
        policy_loss = -torch.mean(ratios * advantages_tensor) - self.entropy_coef * entropy
    
        return policy_loss
    
    def fisher_vector_product(self, vector, states):
        states_tensor = torch.Tensor(states).to(self.device)
        dist = self.actor.get_dist(states_tensor)  # Get current policy distribution
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Compute KL divergence with respect to the current distribution
        kl = torch.sum(log_probs)  # Just sum the log_probs to get a scalar
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_dot_vector = torch.dot(kl_grad_vector, vector)

        hvp = torch.autograd.grad(kl_dot_vector, self.actor.parameters())
        fisher_vector_product = torch.cat([grad.view(-1) for grad in hvp])
    
        return fisher_vector_product + self.damping_coeff * vector

    def compute_kl_divergence(self, old_dist, new_dist):
        mu_old, std_old = old_dist.mean, old_dist.stddev
        mu_new, std_new = new_dist.mean, new_dist.stddev

        # Normalize KL divergence across batch size
        kl_divergence = torch.sum(
            torch.log(std_new / std_old) + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * std_new.pow(2)) - 0.5,
            dim=-1
        ).mean()  # Normalize KL divergence

        return kl_divergence

    def conjugate_gradient(self, fisher_vector_prod_fn, gradient, max_iterations=10, tolerance=1e-10):
        x = torch.zeros_like(gradient)
        r = gradient.clone()
        p = r.clone()
        r_dot_old = torch.dot(r, r)

        for i in range(max_iterations):
            fisher_p = fisher_vector_prod_fn(p)
            alpha = r_dot_old / torch.dot(p, fisher_p)
            x += alpha * p
            r -= alpha * fisher_p
            r_dot_new = torch.dot(r, r)

            if r_dot_new < tolerance:
                break

            beta = r_dot_new / r_dot_old
            p = r + beta * p
            r_dot_old = r_dot_new

        return x
    
    def get_flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def set_flat_params(self, model, flat_params):
        # Set the parameters back into the model using a flat tensor
        offset = 0
        for param in model.parameters():
            param_length = param.numel()
            param.data.copy_(flat_params[offset:offset + param_length].view(param.size()))
            offset += param_length
    
    def line_search(self, states, actions, log_probs_old, step_direction, max_kl, max_backtracks=10, step_size=1.0):
        states_tensor = torch.Tensor(states).to(self.device)
        old_dist = self.actor.get_dist(states_tensor)  # Old policy distribution

        # Flatten parameters for easy manipulation
        old_params = self.get_flat_params(self.actor)

        # Start with the full step
        for step in range(max_backtracks):
            full_step = step_size * step_direction  # Recompute full step every iteration
            new_params = old_params + full_step

            # Set new parameters for the actor
            self.set_flat_params(self.actor, new_params)

            # Compute the new policy distribution and KL divergence
            new_dist = self.actor.get_dist(states_tensor)
            kl = self.compute_kl_divergence(old_dist, new_dist)

            if kl <= max_kl:
                return step_size, True  # Success: KL is within the threshold

            # If KL divergence exceeds threshold, reduce the step size
            step_size *= 0.5  # Reduce scalar step size

        # If line search fails, revert to old parameters
        self.set_flat_params(self.actor, old_params)
        return 0, False
 
    def update_policy(self, step_size, step_direction):
        old_params = self.get_flat_params(self.actor)
        new_params = old_params + step_size * step_direction
        self.set_flat_params(self.actor, new_params)

    def train(self):
        table = Table(
            title=f"Training Progress for {self.n_epochs} Episodes",
            title_style="bold cyan",
            border_style="white",
            show_header=True,
            header_style="bold white",
            box=box.SIMPLE_HEAVY
        )

        table.add_column("Episode", justify="center", style="white")
        table.add_column("Total Reward", justify="center", style="white")
        table.add_column("Policy Loss", justify="center", style="white")
        table.add_column("Value Loss", justify="center", style="white")
        table.add_column("KL Divergence", justify="center", style="white")
        table.add_column("Step Size", justify="center", style="white")

        failure_count = 0

        for episode in range(self.n_epochs):
            s, a, r, logprob_a, val, s_next, done, dw = self.buffer.get_batch()
        
            next_val = self.actor(torch.Tensor(s_next).to(self.device))
            advantages, returns = self.compute_gae(r, val, next_val, done)
        
            value_loss = self.fit_value_function(s, returns)
            policy_loss = self.compute_policy_loss(s, a, logprob_a, advantages)

            # Save the old policy distribution before updating
            old_dist = self.actor.get_dist(torch.Tensor(s).to(self.device))
        
            # Get the policy gradient and perform updates
            policy_grad = torch.autograd.grad(policy_loss, self.actor.parameters())
            policy_grad = torch.cat([grad.view(-1) for grad in policy_grad])

            # Clip the policy gradient
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            
            policy_grad_norm = torch.norm(policy_grad).item()
            print(f"Policy gradient norm: {policy_grad_norm}")

            fvp_fn = lambda v: self.fisher_vector_product(v, s)
            step_direction = self.conjugate_gradient(fvp_fn, policy_grad)

            step_size, success = self.line_search(s, a, logprob_a, step_direction, self.kl_threshold, max_backtracks=5, step_size=0.5)

            if success:
                # Update policy using the step size
                self.update_policy(step_size, step_direction)

                # Compute KL divergence using old and new distributions                 
                new_dist = self.actor.get_dist(torch.Tensor(s).to(self.device))  # New policy
                kl_divergence = self.compute_kl_divergence(old_dist, new_dist)

                total_reward = np.sum(r)

                logger.info(f"KL Divergence after update: {kl_divergence}")

                table.add_row(
                    f"{episode + 1}/{self.n_epochs}",
                    f"{total_reward}",
                    f"{policy_loss.item():.4f}",
                    f"{value_loss:.4f}",
                    f"{kl_divergence:.8f}",
                    f"{step_size:.4f}"
                )
                failure_count = 0  # Reset failure count on success
            else:
                console.print(f"[bold red]Episode {episode + 1}/{self.n_epochs}: Line search failed[/bold red]")
                failure_count += 1
                if failure_count >= 5:
                    print("Too many line search failures, stopping training early.")
                    break

        console.print(table)

