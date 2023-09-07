import torch 
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

class MovieLens20MDataset(Dataset):
    def __init__(self, dataset_path, sep =','):
        data = pd.read_csv(dataset_path, sep=sep)
        data.dropna(inplace = True)
        data = data.to_numpy()[:,:3]
        
        self.items = data[:,:2].astype('int') - 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_idx = np.array((0, ), dtype = np.int_) 
        self.item_idx = np.array((1, ), dtype = np.int_)
        
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, index):
        return self.items[index], self.targets[index]
        
    def __preprocess_target(self, target):
        # array boolean indexing
        target[target < 3.5] = 0
        target[target >= 3.5] = 1
        return target     