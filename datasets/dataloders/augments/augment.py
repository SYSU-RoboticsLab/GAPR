import torchvision.transforms as transforms
import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple
from datasets.dataloders.augments.utils import *

class Augment:
    """
    # Wrapper for Pointcloud Augment
    """
    def __init__(self, name:str, rotate_cmd:str, translate_delta:float, if_jrr:bool):
        print("Augment: name=%s, rotate=%s, translate=%.3f, jrr=%s " % (name, rotate_cmd, translate_delta, if_jrr))
        self.rotate    = RandomRotation(rotate_cmd)
        self.translate = RandomTranslation(translate_delta)

        if if_jrr: raise NotImplementedError("Augment: jrr is currently not implemented.")
        self.jrr = None
        
    def __call__(self, e:torch.Tensor):
        # jrr
        if self.jrr is not None: e0 = self.jrr(e)
        else: e0 = e
        # rotate
        e1, rotms = self.rotate(e0)
        # translate
        e2, trans = self.translate(e1)
        # align data type and device
        e2 = e2.to(e.device).type_as(e).contiguous()
        rotms = rotms.to(e.device).type_as(e).contiguous()
        trans = trans.to(e.device).type_as(e).contiguous()
        return e2, rotms, trans

