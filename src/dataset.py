import torch
from torch.utils import data

TRAIN_MAX_LEN = 50
VALID_MAX_LEN = 100

class CombinedDataset(data.Dataset):
    def __init__(self, split, encoder_states, exp_dataset, layer):
        self.encoder_states = encoder_states
        self.exp_dataset = exp_dataset
        self.layer = layer
        self.split = split
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.encoder_states)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.tensor(self.encoder_states[index][-self.layer])
        
        MAX_LEN = 0
        if self.split == 'train':
            MAX_LEN = TRAIN_MAX_LEN
        else:
            MAX_LEN = VALID_MAX_LEN

        # Load data and get label
        y = torch.tensor([*self.exp_dataset[index]['input_ids'][:MAX_LEN], *([50256] * (MAX_LEN - len(self.exp_dataset[index]['input_ids'][:MAX_LEN])))])
        y_2 = self.exp_dataset[index]['explanation2']
        y_3 = self.exp_dataset[index]['explanation3']

        return X, y, y_2, y_3