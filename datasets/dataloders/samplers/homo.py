# Author:   Jacek Komorowski, https://github.com/jac99/MinkLoc3D
# Modified: Yingrui Jie,      https://github.com/SYSU-RoboticsLab/GAPR

import random
import numpy as np
from typing import List
from datasets.lprdataset import LPRDataset
from datasets.dataloders.samplers.base import BaseSample

class HomoTripletSample(BaseSample):
    def __init__(self, dataset:LPRDataset, max_batches:int):
        """
        # Homogeneous sampling
        * Sampler returning list of indices to form a mini-batch
        * Samples elements in groups consisting of k=2 similar elements (positives)
        * Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
        ## Input
        * dataset
        * max_batches
        """
        self.dataset = dataset
        self.max_batches = max_batches
        self.k = 2

    def get_k(self) -> int:
        return self.k      

    def __call__(self, batch_size:int) -> List[List[int]]:
        assert self.k == 2, "HomoTripletSample: sampler can sample only k=2 elements from the same class"
        assert batch_size >= 2*self.k, "HomoTripletSample: batch_size > 2*k"
        assert batch_size%self.k == 0, "HomoTripletSample: batch_size%k == 0"
        
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        batch_idx:List[List[int]] = []

        unused_elements_ndx:List[int] = self.dataset.get_indices().tolist()

        current_batch:List[int] = []

        while True:
            if len(current_batch) >= batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert len(current_batch) % self.k == 0, "HomoTripletSample: Incorrect bach size: {}".format(len(current_batch))
                    batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (len(batch_idx) >= self.max_batches):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = random.choice(unused_elements_ndx)

            unused_elements_ndx.remove(selected_element)

            positives = list(self.dataset.get_positives(selected_element))
            if len(positives) == 0:
                # Broken dataset element without any positives
                continue
            unused_positives = [e for e in positives if e in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.remove(second_positive)
            else:
                second_positive = random.choice(list(positives))
            current_batch += [selected_element, second_positive]

        for batch in batch_idx:
            assert len(batch) % self.k == 0, "Incorrect bach size: {}".format(len(batch))
        
        return batch_idx