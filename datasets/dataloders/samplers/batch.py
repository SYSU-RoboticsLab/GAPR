import random
import numpy as np
from typing import List
from datasets.lprdataset import LPRDataset
from datasets.dataloders.samplers.base import BaseSample

class BatchSample(BaseSample):
    """
    # Batch sampling for dataset
    """
    def __init__(self, dataset:LPRDataset, shuffle:bool, max_batches:int):
        self.dataset = dataset
        self.max_batches = max_batches
        self.k = 1
        self.shuffle = shuffle

    def get_k(self):
        return self.k

    def __call__(self, batch_size:int) -> List[List[int]]:
        indices = self.dataset.get_indices()
        indices = np.sort(indices)
        if self.shuffle: random.shuffle(indices)
        # remove tail
        indices = indices[:indices.shape[0] - indices.shape[0] % batch_size]
        # reshape to (batches, batch_size) & tolist
        batch_idx = indices.reshape((-1, batch_size)).tolist()
        return batch_idx

