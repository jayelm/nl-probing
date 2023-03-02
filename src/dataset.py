import torch
from torch.utils import data

class TrainDataset(data.Dataset):
    def __init__(self, encoder_states, exp_dataset):
        self.encoder_states = encoder_states
        self.exp_dataset = exp_dataset
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.encoder_states)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.tensor(self.encoder_states[index][-1])

        # Load data and get label
        MAX_LEN = 100
        y = torch.tensor([*self.exp_dataset[index]['input_ids'][:MAX_LEN], *([50256] * (MAX_LEN - len(self.exp_dataset[index]['input_ids'][:MAX_LEN])))])

        return X, y