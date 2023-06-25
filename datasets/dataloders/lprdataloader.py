import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist
from typing import Dict, List, Any
from time import sleep
from datasets.dataloders.augments.augment import Augment
from datasets.dataloders.samplers.lprbatchsampler import LPRBatchSampler
from datasets.dataloders.collates.lprcollate import LPRCollate
from datasets.lprdataset import LPRDataset

from misc.utils import str2bool

def LPRDataLoader(**kw):
    """
    Create dataloaders
    """
    
    augment = None
    if "augment" in kw: augment = Augment(**kw["augment"])

    dataset = LPRDataset(
        rootpath=kw["dataset"], 
    )
    
    sampler = LPRBatchSampler(
        dataset=dataset,
        **kw["sampler"]
    )
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    collate = LPRCollate(dataset=dataset, augment=augment, **kw["collate"])
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=collate,
        num_workers=kw["num_workers"], 
        pin_memory=True
    )
    return dataloader


def parse_opt()->dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml",       type=str, default="config/dataloader.yaml")
    parser.add_argument("--local_rank", type=int, default=None) 
    opt = parser.parse_args()
    opt = vars(opt)
    f = open(opt["yaml"], encoding="utf-8")
    kw:Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader) #读取yaml文件
    f.close()
    kw.update(opt)
    return kw

def test_lprataloader(**kw):
    if kw["local_rank"] is not None: 
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    

    dataloader = LPRDataLoader(**kw["dataloader"])
    for epoch in range(kw["show"]["epoch"]):
        print("epoch", epoch)
        for data, mask in dataloader:
            if kw["show"]["data"]: 
                for e in data: print(e, "\n", data[e])
            if kw["show"]["mask"]: 
                for e in mask: print(e, "\n", mask[e])
            sleep(kw["show"]["sleep"])
    return

if __name__ == "__main__":
    test_lprataloader(**parse_opt())