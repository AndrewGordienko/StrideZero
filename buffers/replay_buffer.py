import torch
import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.dones = []
        self.dws = []

    def add(self, states, actions, rewards, log_probs, values, next_states, done_flags, dw_flags):
        # Ensure rewards, done_flags, and dw_flags are stored as 1D arrays
        self.states.append(np.array(states))
        self.actions.append(np.array(actions))
        self.rewards.append(np.array([rewards]))  # Wrap scalar in an array to make it 1D
        self.log_probs.append(np.array(log_probs))
        self.values.append(np.array(values))
        self.next_states.append(np.array(next_states))
        self.dones.append(np.array([done_flags]))  # Same for done_flags
        self.dws.append(np.array([dw_flags]))  # Same for dw_flags

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.dones = []
        self.dws = []

    def get_batch(self):
        # Concatenate stored experiences into a batch
        s = np.concatenate(self.states, axis=0)
        a = np.concatenate(self.actions, axis=0)
        r = np.concatenate(self.rewards, axis=0)
        logprob_a = np.concatenate(self.log_probs, axis=0)
        val = np.concatenate(self.values, axis=0)
        s_next = np.concatenate(self.next_states, axis=0)
        done = np.concatenate(self.dones, axis=0)
        dw = np.concatenate(self.dws, axis=0)
        return s, a, r, logprob_a, val, s_next, done, dw

class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, log_probs, advantages, td_targets):
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.advantages = advantages
        self.td_targets = td_targets

    def __len__(self):
        return len(self.states)
    def __getitem__(self, idx):
        # Return data as tensors, but do not move to GPU here
        return (
            torch.from_numpy(self.states[idx]).float(),
            torch.from_numpy(self.actions[idx]).float(),
            torch.from_numpy(self.log_probs[idx]).float(),
            torch.tensor(self.advantages[idx]).float(),
            torch.tensor(self.td_targets[idx]).float(),
        )

