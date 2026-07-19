from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self,
        data=None,
        transform=None,
    ):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.transform(self.data[idx][0]),
            self.data[idx][1],
        )  # (review, sentiment)
