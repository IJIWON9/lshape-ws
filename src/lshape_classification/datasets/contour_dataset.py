import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class Contour1DDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = [
            (os.path.join(data_dir, f[:-5] + ".npy"), os.path.join(data_dir, f))
            for f in os.listdir(data_dir) if f.endswith(".json")
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, json_path = self.samples[idx]
        x = np.load(npy_path).astype(np.float32)
        with open(json_path) as f:
            label = json.load(f)['label_idx']
        return torch.tensor(x), torch.tensor(label)