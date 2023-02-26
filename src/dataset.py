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
        X = self.encoder_states[index][-1].squeeze().tolist()

        # Load data and get label
        y = self.exp_dataset[index]['input_ids']

        return X, y