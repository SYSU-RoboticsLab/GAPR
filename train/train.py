# Author:   Jacek Komorowski, https://github.com/jac99/MinkLoc3D
# Modified: Yingrui Jie,      https://github.com/SYSU-RoboticsLab/GAPR

import os
import time
import argparse
import yaml
import torch
from tqdm import tqdm
from typing import Dict, Any, List
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.dataloders.lprdataloader import LPRDataLoader
from models.lprmodel import LPRModel
from loss.lprloss import LPRLoss
from misc.utils import get_datetime, tensors2device, avg_stats

from torch.utils.tensorboard import SummaryWriter

def parse_opt()->dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml",       type=str, default="config/train.yaml")
    parser.add_argument("--local_rank", type=int, required=True)
    opt = parser.parse_args()
    opt = vars(opt)
    f = open(opt["yaml"], encoding="utf-8")
    lprtrain = yaml.load(f, Loader=yaml.FullLoader) #读取yaml文件
    f.close()
    return lprtrain

def main(**kw):
    # 初始化torch.distributed
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend=kw["dist"]["backend"])  # nccl是GPU设备上最快、最推荐的后端

    # get dataloders
    dataloaders = {phase: LPRDataLoader(**kw["dataloaders"]["train"]) for phase in kw["dataloaders"]}
    # get model
    model = LPRModel()
    model.construct(**kw["method"]["model"])
    # get loss function
    loss_fn  = LPRLoss(**kw["method"]["loss"])
    # model to local_rank
    model.model = model.model.to(local_rank)
    # construct DDP model
    model.model = DDP(model.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=kw["dist"]["find_unused_parameters"])
    # initialize optimizer after construction of DDP model
    optimizer = torch.optim.Adam(model.model.parameters(), lr=kw["train"]["lr"], weight_decay=kw["train"]["weight_decay"])
    # get scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, kw["train"]["scheduler_milestones"], gamma=0.1)
    
    # set results
    writer, weights_path = None, None
    if local_rank == 0:
        model_name = get_datetime()
        if kw["results"]["weights"] is not None:
            weights_path = os.path.join(kw["results"]["weights"], model_name)
            if not os.path.exists(weights_path): os.mkdir(weights_path)
            # save config yaml
            with open(os.path.join(weights_path, "config.yaml"), "w") as file:
                file.write(yaml.dump(dict(kw), allow_unicode=True))
        if kw["results"]["logs"] is not None:
            logs_path = os.path.join(kw["results"]["logs"], model_name)
            writer = SummaryWriter(logs_path)


    # get phases from dataloaders
    phases = list(dataloaders.keys())
    # visualize len of phases database
    if local_rank == 0:
        for phase in phases:
            print("Dataloder: {} set len = ".format(phase), len(dataloaders[phase].dataset))
    
    itera = None
    if local_rank == 0: itera = tqdm(range(kw["train"]["epochs"]))
    else: itera = range(kw["train"]["epochs"])
    for epoch in itera:
        for phase in phases:
            # switch mode
            if phase=="train": model.model.train()
            else: model.model.eval()

            # wait barrier
            dist.barrier()

            phase_stats:List[Dict] = []

            for data, mask in dataloaders[phase]:
                # data to device
                data = tensors2device(data, device=local_rank)
                # clear grad
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    output = model(data)
                    loss, stats = loss_fn(output, mask)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                phase_stats.append(stats)
                torch.cuda.empty_cache()
            
            # ******* PHASE END *******
            # compute mean stats for the epoch
            phase_avg_stats = avg_stats(phase_stats)
            # print and save stats
            if local_rank == 0: loss_fn.print_stats(epoch, phase, writer, phase_avg_stats)

        # ******* EPOCH END *******
        # scheduler
        if scheduler is not None: scheduler.step()

        if local_rank == 0 and weights_path is not None:
            model.save(os.path.join(weights_path, "{}.pth".format(epoch)))

if __name__ == "__main__":
    main(**parse_opt())

# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 train.py