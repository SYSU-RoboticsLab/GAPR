from typing import Dict, List, Any
import torch.distributed as dist
from torch.utils.data import Sampler
from datasets.lprdataset import LPRDataset
from datasets.dataloders.samplers.utils import broadcast_batch_idx

class LPRBatchSampler(Sampler[List[int]]):
    """
    # Wrapper for all sampler
    """
    def __init__(
        self, 
        dataset:LPRDataset, 
        name:str,
        batch_size:int, 
        batch_size_limit:int,
        batch_expansion_rate:float, 
        **kw,
    ):
        """
        # Re-generate batch indices
        # Input
        * dataset
        * batch_size: initial batch size
        * sample: `["BaseSample", "HomoTripletSample", "HeteroTripletSample", "RandomSample"]`
        * batch_size_limit: max batch size
        * batch_expansion_rate
        * max_batches
        """
        # sample factory
        self.sample_fn = None
        if name == "BatchSample":
            from datasets.dataloders.samplers.batch import BatchSample
            self.sample_fn = BatchSample(dataset=dataset, **kw)
        elif name == "HomoTripletSample":
            from datasets.dataloders.samplers.homo import HomoTripletSample
            self.sample_fn = HomoTripletSample(dataset=dataset, **kw)
        elif name == "HeteroTripletSample":
            from datasets.dataloders.samplers.hetero import HeteroTripletSample
            self.sample_fn = HeteroTripletSample(dataset=dataset, **kw)
        else:
            raise NotImplementedError("LPRBatchSampler: %s sample_fn not implemented" % name)

        # gpu mode
        self.use_dist = False
        if dist.is_initialized():
            # multi-gpu
            self.use_dist = True
            if dist.get_rank() == 0: print("LPRBatchSampler: multi-gpu mode")
        else:
            # single-gpu
            print("LPRBatchSampler: sigle-gpu mode")
        

        self.batch_size = batch_size - batch_size%self.sample_fn.get_k()
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        if batch_expansion_rate is not None:
            assert batch_expansion_rate > 1., "LPRBatchSampler: batch_expansion_rate must be greater than 1"
            assert batch_size <= batch_size_limit, "LPRBatchSampler: batch_size_limit must be greater or equal to batch_size"

        self.batch_idx = []
        

    def __iter__(self):
        """
        # Generate A Bacth_idx
        """
        # multi-gpu
        if self.use_dist: 
            gen_rank = 0
            all_batch_idx:List[List[int]] = None
            if dist.get_rank() == gen_rank:
                # generate all_batch_idx
                all_batch_idx = self.sample_fn(self.batch_size)
            else: pass
            # broadcast all_batch_idx to all process
            self.batch_idx = broadcast_batch_idx(
                batch_size=self.batch_size, 
                all_batch_idx=all_batch_idx,
                gen_rank=gen_rank
            )
        # single-gpu
        else: 
            self.batch_idx = self.sample_fn(self.batch_size)


        for batch in self.batch_idx: yield batch


    def __len__(self):
        return len(self.batch_idx)

    def expand_batch(self):
        """
        # Expand batch_size by batch_expansion_rate
        """
        if self.batch_expansion_rate is None:
            print("LPRBatchSampler: WARNING batch_expansion_rate is None")
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        
        self.batch_size = self.batch_size - self.batch_size%self.sample_fn.get_k()

        print("LPRBatchSampler: Batch size increased from: {} to {}".format(old_batch_size, self.batch_size))
