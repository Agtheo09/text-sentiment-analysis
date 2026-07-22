from torch.utils.data import Dataset
import torch

class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        review = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.y[idx],
            dtype=torch.long
        )

        return review, label