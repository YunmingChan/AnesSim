import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class AnesthesiaDataset(Dataset):
    def __init__(self, path, stride=1, window_size=100, *, normalize=False, mean=None, std=None):
        self.state = []

        files = [os.path.join(path, filename) for filename in os.listdir(path)]
        for file in files:
            df = pd.read_excel(file)
            data = df.iloc[:, 1:].to_numpy(dtype=np.float32)
            
            if normalize:
                data = (data - mean) / std
                
            for i in range(0, len(data)-window_size, stride):
                self.state.append(data[i:i+window_size])

        
    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return self.state[idx]
    
if __name__ == '__main__':
    training_dataset = AnesthesiaDataset('data/train')
    data = training_dataset[0]
    print(data.shape)