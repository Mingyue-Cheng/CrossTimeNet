from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class Dataset(Dataset):
    def __init__(self, device, mode, Train_data, Test_data):
        self.device = device
        
        if mode == 'train':
            ecgs, labels = Train_data
        else:
            ecgs, labels = Test_data
        
        # Convert data to tensor only once and store
        self.ecgs = [torch.tensor(ecg, dtype=torch.float32).to(device) for ecg in ecgs]
        self.labels = [torch.tensor(label, dtype=torch.float32).to(device) for label in labels]

    def __len__(self):
        return len(self.ecgs)

    def __getitem__(self, item):
        # Just index the pre-converted tensors
        return self.ecgs[item], self.labels[item]