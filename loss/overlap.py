import torch
import open3d as o3d
import numpy as np
from typing import List, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from loss.base import BaseLoss

class OverlapLoss(BaseLoss):
    def __init__(self, corr_dist:float):
        super().__init__()
        print("OverlapLoss: corr_dist=%.2f"%(corr_dist))
        self.corr_dist = corr_dist
    def __call__(self, scores:List[torch.Tensor], coords:List[torch.Tensor],  positives_mask:torch.Tensor, geneous:torch.Tensor):
        source_indices, target_indices = torch.where(positives_mask == True)
        keep_0 = geneous[source_indices] != geneous[target_indices]
        keep_1 = source_indices < target_indices
        select_indices = np.where(keep_0 & keep_1)
        source_indices, target_indices = source_indices[select_indices].tolist(), target_indices[select_indices].tolist()
        losses = []
        stats = {"fitness":[],"inpair_min":[], "inpair_mean":[], "inpair_max":[], "nopair_min":[], "nopair_mean":[], "nopair_max":[]}
        if len(source_indices) == 0:
            # no hetero positive pair, refer to sampler
            return torch.zeros((1,), device=scores[0].device).type_as(scores[0]), {"loss":0.0,"fitness":0.0,"inpair_min":0.0, "inpair_mean":0.0, "inpair_max":0.0, "nopair_min":0.0, "nopair_mean":0.0, "nopair_max":0.0}

        for sndx, tndx in zip(source_indices, target_indices):
            # construct pcd from coords
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(coords[sndx].clone().detach().cpu().numpy())
            source_pcd.paint_uniform_color([0,0,1])
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(coords[tndx].clone().detach().cpu().numpy())
            target_pcd.paint_uniform_color([1,0,0])
            reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, self.corr_dist, np.eye(4))
            corr_set = np.asarray(reg_p2p.correspondence_set)
            assert reg_p2p.fitness > 0.05 


            Ns, Nt = coords[sndx].shape[0], coords[tndx].shape[0]
            source_inpair_indices, source_nopair_indices = corr_set[:, 0], np.setdiff1d(np.arange(Ns), corr_set[:, 0])
            target_inpair_indices, target_nopair_indices = corr_set[:, 1], np.setdiff1d(np.arange(Nt), corr_set[:, 1])
            
            
            source_loss = scores[sndx][source_nopair_indices].mean() + 1.0 - scores[sndx][source_inpair_indices].mean() 
            target_loss = scores[tndx][target_nopair_indices].mean() + 1.0 - scores[tndx][target_inpair_indices].mean() 
            losses += [source_loss, target_loss]

            stats["fitness"]     += [reg_p2p.fitness]
            stats["inpair_min"]  += [scores[sndx][source_inpair_indices].min().item(),  scores[tndx][target_inpair_indices].min().item()]
            stats["inpair_mean"] += [scores[sndx][source_inpair_indices].mean().item(), scores[tndx][target_inpair_indices].mean().item()]
            stats["inpair_max"]  += [scores[sndx][source_inpair_indices].max().item(),  scores[tndx][target_inpair_indices].max().item()]
            stats["nopair_min"]  += [scores[sndx][source_nopair_indices].min().item(),  scores[tndx][target_nopair_indices].min().item()]
            stats["nopair_mean"] += [scores[sndx][source_nopair_indices].mean().item(), scores[tndx][target_nopair_indices].mean().item()]
            stats["nopair_max"]  += [scores[sndx][source_nopair_indices].max().item(),  scores[tndx][target_nopair_indices].max().item()]
        
        loss = torch.stack(losses).mean()
        avg_stats = {e: np.mean(stats[e]) for e in stats}
        avg_stats["loss"] = loss.item()

        return loss, avg_stats

    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        print("OverlapLoss: %.3f" % (stats["loss"]))
        print("Overlap: %.3f, %.3f, %.3f | Non-overlap: %.3f, %.3f, %.3f | Fitness: %.3f" % (
            stats["inpair_min"], stats["inpair_mean"], stats["inpair_max"], 
            stats["nopair_min"], stats["nopair_mean"], stats["nopair_max"],
            stats["fitness"] 
        ))
        return