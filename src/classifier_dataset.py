from torch.utils import data

class ClassifierDataset(data.Dataset):
    def __init__(self, eval_explanations, label):
        self.eval_explanations = eval_explanations
        self.label = label
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.eval_explanations)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.eval_explanations[index]
        # Load data and get label
        y = self.label[index]['label']
        item = {}
        item['input_ids'] = X
        item['labels'] = y
        return item