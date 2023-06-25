import copy
import torch
import torch.distributed as dist
from typing import List

def broadcast_batch_idx(
    batch_size:int,
    all_batch_idx:List[List[int]],
    gen_rank:int,
):
    """
    # Assign all_batch_idx to all processes evenly
    ## Input
    * batch_size
    * all_batch_idx
    * gen_rank: process id that genenrates all_batch_idx
    """
    assert dist.is_available() and dist.is_initialized(), "broadcast: sampler broadcast must be dist.is_initialized()"
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert gen_rank < world_size, "broadcast: sampler gen_rank >= word_size"

    # num_size[0] is the number of batch, num_size[1] is the batch_size, broadcast for initilize
    num_size = torch.tensor([0, 0], dtype=torch.int64).to(rank)

    broadcast_batch_idx:torch.Tensor = None

    if rank == gen_rank:
        assert all_batch_idx is not None, "broadcast_batch_idx: rank == gen_rank and batch_idx is None"
        # print("HeteroBatchSampler: rank %d generating batch idx" % rank)
        # deepcopy all_batch_idx before writing
        all_batch_idx = copy.deepcopy(all_batch_idx)
        all_batch_idx = [e for e in all_batch_idx if len(e)==batch_size]
        # remove tail to ensure all_batch_idx % world_size == 0
        all_batch_idx = all_batch_idx[:len(all_batch_idx)-len(all_batch_idx)%world_size]
        # print("cut len = {}, each = {}".format(num_cut_batchs, num_cut_batchs/num_replicas))
        broadcast_batch_idx = torch.tensor(all_batch_idx).detach().type(torch.int64).to(rank)
        # record num_size
        num_size[0], num_size[1] = broadcast_batch_idx.size()
    else:
        pass

    # board caset num_size
    dist.broadcast(num_size, gen_rank)
    # print("rank {} num_size = {}".format(rank, num_size))

    # initialize batch_idx according to num_size
    if rank == gen_rank: 
        pass
    else:
        broadcast_batch_idx = torch.zeros((num_size[0], num_size[1])).detach().type(torch.int64).to(rank)
    
    # print("rank {} broadcast_batch_idx = {}".format(rank, broadcast_batch_idx.size()))
    dist.broadcast(broadcast_batch_idx, gen_rank)

    # broadcast_batch_idx = [[int(c) for c in r] for r in broadcast_batch_idx.cpu()]
    broadcast_batch_idx = broadcast_batch_idx.cpu().numpy().tolist()
    assert len(broadcast_batch_idx)%world_size == 0, "broadcast: len(broadcast_batch_idx)%world_size != 0"
    avg_num = int(len(broadcast_batch_idx)/world_size)
    batch_idx = broadcast_batch_idx[rank*avg_num: (rank+1)*avg_num]

    return batch_idx