import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        features = torch.tensor(
            self.X[idx].toarray(),
            dtype=torch.float32
        ).squeeze()

        label = torch.tensor(
            self.y[idx],
            dtype=torch.long
        )

        return features, label