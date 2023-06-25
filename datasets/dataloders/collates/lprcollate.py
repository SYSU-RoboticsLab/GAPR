# Author:   Jacek Komorowski, https://github.com/jac99/MinkLoc3D
# Modified: Yingrui Jie,      https://github.com/SYSU-RoboticsLab/GAPR

from time import sleep
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from datasets.lprdataset import LPRDataset
from scipy.spatial.transform import Rotation as R
# common
from datasets.dataloders.collates.utils import align_pcs, in_sorted_array, triplet_mask

def LPRCollate(dataset:LPRDataset, augment, name:str, **kw):
    """
    # Wrapper for all collate_fn
    ## Format
    ```
    def collate_fn(data_list):
        # data_list defined in LPRdataset.__getitem__()
        # data_list = [
        #     [label0, pc0], 
        #     [label1, pc1]
        # ]
        ...
        # data 
        data:Dict[str, torch.Tensor] = {"data":data}

        mask:Dict[str, torch.Tensor] = {"mask":mask}
        return data, mask
    ```
    ## Training
    ```
    for data, mask in dataloader:
        data = tensors2device(data, device)
        output = model(data)
        loss = loss_fn(output, mask)
        ...
    ```
    """
    if name == "BaseCollate":
        def BaseCollate(data_list):
            """
            # BaseCollate
            * align clouds
            * convert labels to tensor
            """
            clouds = [e[1] for e in data_list]
            clouds = align_pcs(clouds)
            clouds = torch.stack(clouds, dim=0)

            labels = torch.tensor([e[0] for e in data_list])
            data:Dict[str, torch.Tensor] = {"clouds":clouds}
            mask:Dict[str, torch.Tensor] = {"labels":labels}
            return data, mask
        return BaseCollate
    
    elif name == "MetricCollate":
        def MetricCollate(data_list):
            """
            # Metric Learning Collate Function
            """
            # constructs a batch object
            raw_coords = [e[1] for e in data_list]
            labels = [e[0] for e in data_list]
            # align points number
            raw_coords = align_pcs(raw_coords)
            # Tensor: raw_coords: [BS, PN, 3],
            raw_coords = torch.stack(raw_coords, dim=0)
            BS, device = raw_coords.shape[0], raw_coords.device

            # get tums from dataset
            tums = []
            for ndx in labels: tums.append(dataset.get_tum(ndx))
            tums = np.asarray(tums)
            # Tensor: raw_rotms: [BS, 3, 3], raw_trans: [BS, 3]
            # tf_global2raw is [raw_rotms, raw_trans]
            raw_trans = torch.tensor(tums[:, 1:4], device=device).type_as(raw_coords) 
            raw_rotms = torch.tensor(R.from_quat(tums[:, 4:8]).as_matrix(), device=device).type_as(raw_coords) 

            # apply augment
            if augment is not None:
                # Tensor: aug_coords: [BS, PN, 3], aug_rotms: [BS, 3, 3], aug_trans: [BS, 3]
                # coords = aug_rotms * raw_coords + aug_trans, tf_aug2raw is [aug_rotms, aug_trans]
                coords, aug_rotms, aug_trans = augment(raw_coords.clone()) 
                # compute tf_global2aug = tf_global2raw * tf_aug2raw.inverse()
                # tf_aug2raw.inverse() = [aug_rotms.T, -aug_rotms.T*aug_trans]
                aug_rotms_inv, aug_trans_inv = aug_rotms.transpose(1,2), -torch.bmm(aug_rotms.transpose(1,2), aug_trans.unsqueeze(2)).squeeze(2)
                # trans = raw_trans + raw_rotms * aug_rotms.inverse() * aug_trans 
                trans = raw_trans + torch.bmm(raw_rotms, aug_trans_inv.unsqueeze(2)).squeeze(2)
                # rotms = raw_rotms * aug_rotms.inverse()
                rotms = torch.bmm(raw_rotms, aug_rotms_inv)
            else:
                trans, rotms, coords = raw_trans.clone(), raw_rotms.clone(), raw_coords.clone()

            # set feats to 1, or color if rgb pointcloud
            feats = torch.ones((coords.shape[0], coords.shape[1], 1))
            # compute positives and negatives mask
            positives_mask, negatives_mask = triplet_mask(dataset, labels)

            # get geneous
            geneous = dataset.get_all_geneous()[labels]
            geneous = torch.tensor(geneous, dtype=torch.int)
            # get labels
            labels = torch.tensor(labels)
            # write to data and mask
            data:Dict[str, torch.Tensor] = {"coords":coords, "feats":feats, "geneous":geneous}
            mask:Dict[str, torch.Tensor] = {"labels":labels, "geneous":geneous, "rotms":rotms, "trans":trans, "positives":positives_mask, "negatives":negatives_mask}
            return data, mask
        return MetricCollate
    else:
        raise NotImplementedError("LPRCollate: %s not implemented" % name)

