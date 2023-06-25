# Author:   Jacek Komorowski, https://github.com/jac99/MinkLoc3D
# Modified: Yingrui Jie,      https://github.com/SYSU-RoboticsLab/GAPR

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple
from datasets.lprdataset import LPRDataset

def align_pcs(pcs:List[torch.Tensor], align_size:int=None)->List[torch.Tensor]:
    """
    # align points number in pointclouds
    ## Input
    * pcs
    * align_size: points number \n
    ## Output
    * newpcs
    """
    if align_size is None: 
        # if None, find the max size
        max_size = 0
        for pc in pcs: 
            if pc.size()[0] > max_size: max_size = pc.size()[0]
        align_size = max_size
    else:
        for pc in pcs: 
            assert pc.size()[0] <= align_size, "LPRCollate: pc.size()[0] <= align_size"
    
    newpcs:List[torch.Tensor] = []
    for pc in pcs:
        # zero padding
        newpcs.append(F.pad(pc, (0,0,0,align_size-pc.size()[0]), "constant", 0))
    return newpcs

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Removal Outlier', width=1920,
                                      height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

def triplet_mask(dataset:LPRDataset, labels:List[int])->Tuple[torch.Tensor, torch.Tensor]:
    positives_mask = [[in_sorted_array(e, np.sort(np.asarray(dataset.get_positives(label)))) for e in labels] for label in labels]
    negatives_mask = [[not in_sorted_array(e, np.sort(np.asarray(dataset.get_non_negatives(label)))) for e in labels] for label in labels]
    positives_mask = torch.tensor(positives_mask)
    negatives_mask = torch.tensor(negatives_mask)
    return positives_mask, negatives_mask