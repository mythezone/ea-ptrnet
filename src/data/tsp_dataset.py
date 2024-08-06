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

# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# Training Data
class TSPDataset(Dataset):
    """TSP Dataset
    """
    def __init__(self, file_path=None, train=False,dim = 2,  size=50, num_samples=100000, random_seed=1111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)
        self.size = size 
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.file_path = file_path
        self.data_set = []
        self.train = train
        self.dim = dim
        
        if os.path.exists(file_path):
            self.data_set = self.load_data()
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.data_set = self.generate_data()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
    
    def __str__(self) -> str:
    
        output = []
        for index, problem in enumerate(self.data_set):
            output.append("-------------------------------------------")
            output.append(f"Problem {index}:")
            for city_index, city in enumerate(problem):
                output.append(f"|--> City {city_index}: {city}")
            output.append("///////////////////////////////////////////")
        return "\n".join(output)
    
    def generate_data(self)->None:
        """Generate Random TSP Data
        
        Args:
        -----
        
        Returns:
        --------
            List[torch.Tensor]: List of Tensors containing the TSP data
        """
        
        # randomly sample points uniformly from [0, 1]
        for l in tqdm(range(self.num_samples)):
            x = torch.FloatTensor(self.size,self.dim).uniform_(0, 1)
            self.data_set.append(x)
        
        self.size = len(self.data_set)

        torch.save(self.data_set, self.file_path)
        
        return self.data_set

    def load_data(self)->List[torch.Tensor]:
        """Load TSP Data
        
        Args:
        -----
            path (str): Path to the TSP data
        
        Returns:
        --------
            List[torch.Tensor]: List of Tensors containing the TSP data
        """
        self.data_set = torch.load(self.file_path)
        
        return self.data_set


if __name__ == "__main__":
        
    # define problem dimension
    input_dim = 2
    # define tsp size
    size = 5
    # define training size
    train_size = 10

    dataset = TSPDataset(file_path="./dataset/train/test_2_5_10.pt",train=True,dim =input_dim, size=size,
        num_samples=train_size)
    dp = dataset.generate_data()
    print(dataset )
