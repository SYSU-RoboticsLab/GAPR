# Author:   Jacek Komorowski, https://github.com/jac99/MinkLoc3D
# Modified: Yingrui Jie,      https://github.com/SYSU-RoboticsLab/GAPR

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from loss.base import BaseLoss
import matplotlib.pylab as plt

def get_max_per_row(mat:torch.Tensor, mask:torch.Tensor):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows

def get_min_per_row(mat:torch.Tensor, mask:torch.Tensor):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float("inf")
    return torch.min(mat_masked, dim=1), non_inf_rows

class TripletMiner:
    def __init__(self):
        return
    def __call__(self, dist_mat:torch.Tensor, positives_mask:torch.Tensor, negatives_mask:torch.Tensor):
        # [Ns, Nt] mat
        assert dist_mat.shape == positives_mask.shape == negatives_mask.shape
        with torch.no_grad():
            # Based on pytorch-metric-learning implementation
            (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
            (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
            a_keep_idx = torch.where(a1p_keep & a2n_keep)[0]
            anc_ind = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
            pos_ind = hardest_positive_indices[a_keep_idx]
            neg_ind = hardest_negative_indices[a_keep_idx]

            stats = {
                "triplet_num"  :a_keep_idx.shape[0],
                "max_pos_dist" :torch.max(hardest_positive_dist[a_keep_idx]).item(),
                "mean_pos_dist":torch.mean(hardest_positive_dist[a_keep_idx]).item(),
                "min_pos_dist" :torch.min(hardest_positive_dist[a_keep_idx]).item(),
                "max_neg_dist" :torch.max(hardest_negative_dist[a_keep_idx]).item(),
                "mean_neg_dist":torch.mean(hardest_negative_dist[a_keep_idx]).item(),
                "min_neg_dist" :torch.min(hardest_negative_dist[a_keep_idx]).item(),
            }
            return anc_ind, pos_ind, neg_ind, stats



class BatchTripletLoss(BaseLoss):
    def __init__(self, margin:float, style:str):
        super().__init__()
        assert style in ["soft", "hard"]
        print("BatchTripletLoss: margin=%.1f, style=%s"%(margin, style))
        self.miner = TripletMiner()
        self.margin = margin
        self.style  = style
        return

    def __call__(self, 
        source_feats:torch.Tensor, target_feats:torch.Tensor, 
        positives_mask:torch.Tensor, negative_mask:torch.Tensor
    ):
        stats = {}
        # get dist l2d mat
        dist_mat = torch.norm(source_feats.unsqueeze(1) - target_feats.unsqueeze(0), dim=2)
        # miner
        anc, pos, neg, miner_stats = self.miner(dist_mat, positives_mask, negative_mask)
        stats.update(miner_stats)
        pos_dist = torch.norm(source_feats[anc] - target_feats[pos], dim=1) 
        neg_dist = torch.norm(source_feats[anc] - target_feats[neg], dim=1)
        triplet_dist = pos_dist - neg_dist
        with torch.no_grad():
            stats["norm"] = torch.norm(source_feats, dim=1).mean().item()
            stats["non_zero_triplet_num"] = torch.where((triplet_dist + self.margin) > 0)[0].shape[0]
        
        if self.style == "hard":
            loss = F.relu(triplet_dist + self.margin).mean()
        elif self.style == "soft":
            loss = torch.log(1+self.margin*torch.exp(triplet_dist)).mean()
        else:
            raise NotImplementedError(f"BatchTripletLoss: unkown style {self.style}")

        stats["loss"] = loss.item()
        return loss, stats

    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        print("TripletLoss: %.3f, Norm: %.3f, All/Non-zero: %.1f/%.1f"%(
            stats["loss"], stats["norm"], stats["triplet_num"], stats["non_zero_triplet_num"]
        ))
        print("Positive: %.3f, %.3f, %.3f | Negative: %.3f, %.3f, %.3f (min, avg, max)"%(
            stats["min_pos_dist"], stats["mean_pos_dist"], stats["max_pos_dist"],
            stats["min_neg_dist"], stats["mean_neg_dist"], stats["max_neg_dist"],
        ))
        return
