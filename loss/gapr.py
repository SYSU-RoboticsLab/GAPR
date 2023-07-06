import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Tuple
from torch.utils.tensorboard import SummaryWriter
import random


from loss.base import BaseLoss
from loss.triplet import BatchTripletLoss
from loss.point import PointTripletLoss
from loss.overlap import OverlapLoss

class GAPRLoss(BaseLoss):
    def __init__(self, batch_loss:Dict, point_loss:Dict, overlap_loss:Dict, point_loss_scale:float, overlap_loss_scale:float):
        super().__init__()
        print("GAPRLoss: point_loss_scale=%.2f overlap_loss_scale=%.2f"%(point_loss_scale, overlap_loss_scale))
        self.batch_loss = BatchTripletLoss(**batch_loss)
        self.point_loss = PointTripletLoss(**point_loss)
        self.overlap_loss = OverlapLoss(**overlap_loss)
        self.point_loss_scale = point_loss_scale
        self.overlap_loss_scale = overlap_loss_scale

    def __call__(self, 
        # model 
        embeddings:torch.Tensor, 
        coords:List[torch.Tensor], 
        feats:List[torch.Tensor], 
        scores:List[torch.Tensor],
        # mask
        rotms:torch.Tensor, 
        trans:torch.Tensor,
        positives_mask:torch.Tensor, 
        negatives_mask:torch.Tensor,
        geneous:torch.Tensor
    ):
        # get global coords
        device, BS = embeddings.device, embeddings.shape[0]
        rotms, trans = rotms.to(device), trans.to(device)
        # R*p + T
        global_coords = [torch.mm(rotms[ndx], coords[ndx].clone().detach().transpose(0,1)).transpose(0,1) + trans[ndx].unsqueeze(0) for ndx in range(BS)]
        # compute point loss
        point_loss, point_stats = self.point_loss(feats, global_coords, positives_mask)
        # compute attention loss
        overlap_loss, overlap_stats = self.overlap_loss(scores, global_coords, positives_mask, geneous)
        # compute  batch loss
        batch_loss, batch_stats = self.batch_loss(embeddings, embeddings, positives_mask, negatives_mask)

        stats = {"batch":batch_stats, "point":point_stats, "overlap":overlap_stats}
        # stats.update(mean_point_stats_show)
        return batch_loss+self.point_loss_scale*point_loss+self.overlap_loss_scale*overlap_loss, stats
    
    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        self.batch_loss.print_stats(epoch, phase, writer, stats["batch"])
        self.point_loss.print_stats(epoch, phase, writer, stats["point"])
        self.overlap_loss.print_stats(epoch, phase, writer, stats["overlap"])
        # print("point_consistence_loss: pos=%.3f, neg=%.3f" % (stats["pos_l2ds"], stats["neg_l2ds"]))
        return