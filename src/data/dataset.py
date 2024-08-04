"""
Created on : 2024-08-04
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/github/ea-ptrnet/src/data/dataset.py
Original Reporsitory: https://github.com/qiang-ma/graph-pointer-network
Description: Generating Datasets
---
Updated    : 
---
Todo       : 
"""


from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import time
from scipy.spatial import distance

from typing import Optional,List,Tuple,Dict,Union,Protocol




# Training Data
class TSPDataset(Dataset):
    """TSP Dataset
    """
    def __init__(self, dataset_fname=None, train=False, size=50, num_samples=100000, random_seed=1111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)
        self.data_set = []

        # randomly sample points uniformly from [0, 1]
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, size).uniform_(0, 1)
            # x = torch.cat([start, x], 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]


if __name__ == "__main__":
        
    # define problem dimension
    input_dim = 2
    # define tsp size
    size = 50
    # define training size
    train_size = 1000

    dataset = TSPDataset(train=True, size=size,
        num_samples=train_size)

    # save the dataset
    torch.save(dataset, './TSP50_1000.pt')
    # load the dataset
    # dataset = torch.load('./TSP50_1000.pt')