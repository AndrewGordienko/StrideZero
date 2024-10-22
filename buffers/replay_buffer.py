import torch

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, states, actions, log_probs, advantages, td_targets):
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.advantages = advantages
        self.td_targets = td_targets

    def __len__(self):
        return len(self.states)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]).float(),
            torch.from_numpy(self.actions[idx]).float(),
            torch.from_numpy(self.log_probs[idx]).float(),
            torch.tensor(self.advantages[idx]).float(),
            torch.tensor(self.td_targets[idx]).float(),
        )

