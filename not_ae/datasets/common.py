from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, dataset: Dataset, length: int):
        self.dataset = dataset
        assert hasattr(self.dataset, "__len__")
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]


class IgnoreLabelDataset(Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)