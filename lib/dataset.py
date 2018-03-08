import numpy as np
import torch
from torch.utils.data import Dataset


class DummyBballDataset(Dataset):
    """
    A random number generator pretending to be a Bball dataset.
    Here to give an idea about expected data structure
    """

    def __init__(self, data_size, seed):
        np.random.seed(seed)
        self.state_court_data = np.random.uniform(0, 1, size=(data_size, 17, 47, 50))
        self.state_flat_data = np.random.normal(size=(data_size, 93))
        self.action = np.random.randint(0, 20, size=data_size)
        self.reward = np.random.normal(size=data_size)
        self.next_state_court_data = np.random.uniform(0, 1, size=(data_size, 17, 47, 50))
        self.next_action = np.random.randint(0, 20, size=data_size)
        self.feasible_actions = np.random.choice(np.arange(20), size=(data_size, 4))
        # make sure feasible actions includes observed action
        self.feasible_actions = np.concatenate((self.feasible_actions, self.action.reshape(data_size, 1)), axis=1)

        self.train_ind = np.arange(int(data_size/2))
        self.valid_ind = np.arange(int(data_size/4)) + int(data_size/2)
        self.test_ind = np.arange(int(data_size/4)) + int(data_size/2) + int(data_size/4)

    def __len__(self):
        return len(self.state_court_data)

    @staticmethod
    def feasible_action_mask(feasible_actions):
        """
        Given list of feasible actions, returns byte mask
        """
        feasible_mask = np.zeros(20)
        feasible_mask[feasible_actions] = 1
        return feasible_mask

    def __getitem__(self, idx):
        state = (self.state_court_data[idx], self.state_flat_data[idx])
        action = self.action[idx]
        reward = self.reward[idx]
        next_state = (self.next_state_court_data[idx], self.state_flat_data[idx])
        next_action = self.next_action[idx]
        feasible_mask = self.feasible_action_mask(self.feasible_actions[idx])
        return ((torch.from_numpy(state[0]).float(),
                 torch.from_numpy(state[1]).float()),
                torch.from_numpy(np.array(action).reshape(1)).long(),
                torch.from_numpy(np.array(reward).reshape(1)).float(),
                (torch.from_numpy(next_state[0]).float(),
                 torch.from_numpy(next_state[1]).float()),
                torch.from_numpy(np.array(next_action).reshape(1)).long(),
                torch.from_numpy(feasible_mask).byte(),
                )
