import os
import random
import torch
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import List
from misc.utils import str2bool

class LPRDataset(Dataset):
    """
    # Dataset wrapper for LPRDataset
    """
    def __init__(self, rootpath:str,):
        self.rootpath = rootpath
        assert os.path.exists(self.rootpath), "Cannot access rootpath {}".format(self.rootpath)
        # 0: ground, 1: aerial
        self.geneous_names = ["ground", "aerial"]
        self.Ng = len(self.geneous_names)
        self.geneous = np.load(os.path.join(self.rootpath, "geneous.npy"))
        self.Nm = self.geneous.shape[0]
        # self.homoindices
        # ground: [0, 2, 3, 5, 8, ...]
        # aerial: [1, 4, 6, 7, 9, ...]
        self.homoindices = [[] for _ in self.geneous_names]
        for ndx in range(self.Nm):
            self.homoindices[self.geneous[ndx]].append(ndx)
        self.homoindices = [np.asarray(e) for e in self.homoindices]

        # tum format (Nm, 8) [t, x, y, z, qx, qy, qz, qw]
        self.tum = np.load(os.path.join(self.rootpath, "tum.npy"))
        assert self.Nm == self.tum.shape[0], "LPRDataset: self.Nm != self.tum.shape[0]"

        # make self check files
        self.checkpath = os.path.join(self.rootpath, "selfcheck")
        if not os.path.exists(self.checkpath): os.mkdir(self.checkpath)

        self.anchors:np.ndarray = None
        # load data 
        self.pcs           = [np.load(os.path.join(self.rootpath, "items", "%06d"%ndx, "pointcloud.npy"))    for ndx in range(self.Nm)]
        self.positives     = [np.load(os.path.join(self.rootpath, "items", "%06d"%ndx, "positives.npy"))     for ndx in range(self.Nm)]
        self.non_negatives = [np.load(os.path.join(self.rootpath, "items", "%06d"%ndx, "non_negatives.npy")) for ndx in range(self.Nm)]


    def __len__(self):
        return self.Nm

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        pc = torch.tensor(self.get_pc(ndx))
        return ndx, pc

    def get_indices(self) -> np.ndarray:
        return np.arange(self.Nm)
    
    def get_homoindices(self, geneous_id:int) -> np.ndarray:
        return np.copy(self.homoindices[geneous_id])

    def get_geneous_names(self) -> List[str]:
        return self.geneous_names

    def get_all_geneous(self) -> np.ndarray:
        return np.copy(self.geneous)

    def get_positives(self, ndx:int) -> np.ndarray:
        return np.copy(self.positives[ndx])
    
    def get_non_negatives(self, ndx:int) -> np.ndarray:
        return np.copy(self.non_negatives[ndx])

    def get_tum(self, ndx:int):
        return np.copy(self.tum[ndx])

    def get_correspondences(self, source_ndx:int, target_ndx:int)  -> np.ndarray:
        path = os.path.join(
            self.rootpath, 
            "items", 
            "%06d"%source_ndx, 
            "correspondence",
            "%06d.npy"%target_ndx
        )
        return np.load(path)

    def get_pc(self, ndx) -> np.ndarray:
        return np.copy(self.pcs[ndx])
    
    def get_anchors(self) -> np.ndarray:
        """
        # Get indices of items with heterogeneous positive samples in dataset
        """
        if self.anchors is not None: return np.copy(self.anchors)
        print("LPRDataset: self.anchors is None, generating")
        anchors = []
        for i in tqdm(self.get_indices()):
            positives = self.get_positives(i)
            is_anchor = True
            for gid, gname in enumerate(self.geneous_names):
                if np.intersect1d(positives, self.get_homoindices(gid)).shape[0] == 0:
                    is_anchor = False
                    break
            if is_anchor: anchors.append(i)

        self.anchors = np.asarray(anchors)
        print("LPRDataset: Found %d anchors" % len(anchors))

        for gid, gname in enumerate(self.get_geneous_names()):
            ganchors = np.intersect1d(
                self.anchors,
                self.get_homoindices(gid)
            ).shape[0]
            print("LPRDataset: %s has %d anchors" % (gname, ganchors))
        
        return np.copy(self.anchors)

    def check_hetero_triplet(self):
        """
        # Count hetero triplet number
        """
        print("LPRDataset: check hetero triplet")
        # multi_geneous_positives 
        mgp = np.zeros((self.Ng, self.Ng))
        mgn = np.zeros((self.Ng, self.Ng))
        for i in tqdm(self.get_indices()):
            sgid = self.geneous[i]
            positives = self.get_positives(i)
            non_negative = self.get_non_negatives(i)
            for tgid in range(self.Ng):
                mgp[sgid][tgid] += np.intersect1d(positives, self.homoindices[tgid]).shape[0]
                # mgn[sgid][tgid] += np.intersect1d(non_negative, self.homoindices[tgid]).shape[0]
                mgn[sgid][tgid] += np.intersect1d(
                    np.setdiff1d(self.get_indices(), non_negative),
                    self.homoindices[tgid]
                ).shape[0]
        mgp = mgp/np.array([self.homoindices[0].shape[0], self.homoindices[1].shape[0]])
        mgn = mgn/np.array([self.homoindices[0].shape[0], self.homoindices[1].shape[0]])
        print("Avg positive:")
        print(str(mgp))
        print("Avg negative:")
        print(str(mgn))
        return
    
    def check_positives(self, step:int=1):
        print("LPRDataset: check_positives")
        pos_map:dict[str, np.ndarray] = {}
        for sgid, source in enumerate(self.get_geneous_names()):
            for tgid, target in enumerate(self.get_geneous_names()):
                keyname = "%s-%s" % (source, target)
                npos = []
                nmap = []
                sindices = self.get_homoindices(sgid)
                tindices = self.get_homoindices(tgid)
                for sndx in tqdm(sindices, desc=keyname):
                    this_npos = np.intersect1d(
                        self.get_positives(sndx),
                        tindices
                    ).shape[0]
                    this_npos = int(this_npos/step)*step
                    if this_npos in npos: nmap[npos.index(this_npos)] += 1
                    else:
                        npos.append(this_npos)
                        nmap.append(1)
                this_pos_map = np.asarray([npos, nmap])
                sort_ndx = np.argsort(this_pos_map[0])
                this_pos_map = this_pos_map[:, sort_ndx]
                pos_map[keyname] = this_pos_map

        plt.figure(figsize=(7,4))

        plt.grid()
        for keyname in pos_map:
            plt.plot(pos_map[keyname][0], pos_map[keyname][1])
            

        plt.xlabel("Number of positive samples in database")
        plt.ylabel("Number of queries")
        plt.legend(list(pos_map.keys()))
        plt.show()
        
        return

    def check_pn(self):
        print("LPRDataset: check points number")
        for gid, geneous in enumerate(self.geneous_names):
            if self.get_homoindices(gid).shape[0] == 0: continue
            pn = 0
            for i in self.homoindices[gid]:
                pn += self.get_pc(i).shape[0]
            avgpn = pn / self.homoindices[gid].shape[0]
            print("%s avg pn = %.3f" % (geneous, avgpn))
        return
    
    
    def show_submaps(self, N=10):
        """
        # visualize some submaps
        """
        anchors = self.get_anchors()

        ganchors = np.intersect1d(
            anchors,
            self.get_homoindices(1)
        ) 
        for _ in range(N):
            a = random.choice(ganchors)
            p = random.choice(
                np.intersect1d(
                    self.get_homoindices(0),
                    self.get_positives(a)
                )
            )
            pcda = o3d.geometry.PointCloud()
            pcda.points = o3d.utility.Vector3dVector(self.get_pc(a))
            
            pcdp = o3d.geometry.PointCloud()
            pcdp.points = o3d.utility.Vector3dVector(self.get_pc(p) + np.asarray([-70, 0, 0]))
            
            o3d.visualization.draw_geometries(
                [pcda, pcdp], 
                window_name="left:gorund, right: aerial"
            )
            



def test_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",              type=str,      required=True)
    parser.add_argument("--check_positives",      type=str2bool, default=True)
    parser.add_argument("--check_hetero_triplet", type=str2bool, default=True)
    parser.add_argument("--check_pn",             type=str2bool, default=True)
    parser.add_argument("--get_anchors",          type=str2bool, default=True)
    parser.add_argument("--show_submaps",         type=str2bool, default=5)
    opt = parser.parse_args()
    opt = vars(opt)
    lprdataset = LPRDataset(rootpath=opt["dataset"])

    if opt["check_positives"]:      lprdataset.check_positives()
    if opt["check_hetero_triplet"]: lprdataset.check_hetero_triplet()
    if opt["check_pn"]:             lprdataset.check_pn()
    if opt["get_anchors"]:          lprdataset.get_anchors()
    if opt["show_submaps"] > 0:     lprdataset.show_submaps(opt["show_submaps"])

    return

if __name__ == "__main__":
    test_dataset()