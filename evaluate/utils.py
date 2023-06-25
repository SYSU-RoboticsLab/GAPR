import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from typing import List
from time import sleep
from sklearn.neighbors import KDTree
from sklearn import manifold
from torch.utils.data import DataLoader
from datasets.lprdataset import LPRDataset
from models.lprmodel import LPRModel
from misc.utils import tensors2device






def get_embeddings(lprmodel: LPRModel, dataloader: DataLoader, device:str, print_stats:bool=True):
    lprmodel.model = lprmodel.model.to(device)
    lprmodel.model.eval()
    embeddings = []
    if print_stats: iterater = tqdm(dataloader, desc="Getting embedding")
    else: iterater=dataloader
    for data, mask in iterater:
        data = tensors2device(data, device)
        with torch.no_grad():
            output = lprmodel(data)
            assert "embeddings" in output, "Evaluate: no embeddings in model output"
            embeddings.append(output["embeddings"].clone().detach().cpu().numpy())
            # visualize_cnn_feats_scores(output["feats"], output["scores"])
            # visualize_cnn_feats(output["coords"], output["feats"], data["coords"])
            # visualize_scores(output["coords"], output["scores"], data["coords"])
    
    embeddings = np.concatenate(embeddings, axis=0)
    if print_stats: print("Embeddings size = ", embeddings.shape)
    # np.save("examples/minkloc3d/results/embeddings", embeddings)
    return embeddings

def get_topN_recall_curve(    
    dataset:LPRDataset, 
    distance:np.ndarray, 
    source_indices:np.ndarray, 
    target_indices:np.ndarray,
    topN:int=10
):
    # top_one_percent = int(target_indices.shape[0]/100)
    topN_count = np.zeros((topN,), dtype=np.int32)
    topN_recall = np.zeros((topN,))
    for j in range(topN):
        for sndx in source_indices:
            real_positive = np.intersect1d(
                dataset.get_positives(sndx),
                target_indices
            )
            # print("true_positives", true_positives)
            pred_positive = target_indices[np.argsort(distance[sndx][target_indices])][:j]

            recall_positive = np.intersect1d(
                pred_positive,
                real_positive
            )
            if recall_positive.shape[0] != 0: 
                topN_count[j] += 1
        topN_recall[j] = float(topN_count[j]) / float(source_indices.shape[0])
    
    return topN_recall[1:]

def get_hetero_topN_recall(
    dataset:LPRDataset, 
    distance:np.ndarray,
    savepath:str=None,
    show:bool=False
):
    assert len(dataset) == distance.shape[0], "Evaluate: len(datasets) == embeddings.shape[0]"

    # source gen
    geneous_names = dataset.get_geneous_names()

    all_topN_recall = {}
    for sgid, source in enumerate(geneous_names):
        sgindices = dataset.get_homoindices(sgid)
        for tgid, target in enumerate(geneous_names):
            tgindices = dataset.get_homoindices(tgid)
            # print(source, "-", target)
            if sgindices.shape[0] == 0 or tgindices.shape[0] == 0:
                print("no instance in source or target, continue")
                continue
            topN_recall = get_topN_recall_curve(dataset, distance, sgindices, tgindices)
            all_topN_recall["{}-{}".format(source, target)] = topN_recall
    # print("all-all")
    all_topN_recall["all-all"] = get_topN_recall_curve(dataset, distance, dataset.get_indices(), dataset.get_indices())
    
    plt.figure()
    
    # plt.ylim(-0.1, 1.1)
    plt.grid()
    for topN_recall in all_topN_recall:
        plt.plot(all_topN_recall[topN_recall])
    plt.xlabel("topN")
    plt.ylabel("recall")
    plt.legend(list(all_topN_recall))
    if savepath is not None: plt.savefig(os.path.join(savepath, "topN-recall.png"))
    elif show: plt.show()
    else: plt.close()
    return all_topN_recall

def get_recall_precision_curve(
    dataset:LPRDataset, 
    distance:np.ndarray, 
    source_indices:np.ndarray, 
    target_indices:np.ndarray,
    num_eval:int,
):
    # rp = np.array([[0.0, 1.0], [1.0, 0.0]])
    rp = np.empty((0,2))
    ds = np.linspace(np.min(distance)-0.01, np.max(distance)+0.01, num_eval)
    for threshold in ds:
        threshold_rp = np.empty((0,2))
        for i in source_indices:
            real_positive = np.intersect1d(
                dataset.get_positives(i),
                target_indices,
            )
            pred_positive = np.intersect1d(
                np.where(distance[i] < threshold)[0],
                target_indices,
            )
            
            # if real_positive.shape[0] == 0 or pred_positive.shape[0] == 0: continue
            tp = np.intersect1d(real_positive, pred_positive).shape[0]
            fn = np.setdiff1d(real_positive, pred_positive).shape[0]
            fp = np.setdiff1d(pred_positive, real_positive).shape[0]
            # tqdmiter.write(str(tp)+" "+str(fn)+" "+str(fp))
            recall, precision = 0., 0.
            if tp == 0:
                if   fn == 0 and fp == 0: continue
                elif fn == 0 and fp != 0: recall, precision = 1., 0.
                elif fn != 0 and fp == 0: recall, precision = 0., 1.
                else:                     recall, precision = 0., 0.
            else:
                recall = float(tp)/float(tp+fn)
                precision = float(tp)/float(tp+fp)
            # this_rp.append([recall, pricision])
            threshold_rp = np.concatenate([threshold_rp, np.asarray([[recall, precision]])], axis=0)
        
        if threshold_rp.shape[0] == 0: continue
        threshold_rp = np.mean(np.asarray(threshold_rp), axis=0)
        
        # tqdm_iter.set_postfix(recall=threshold_rp[0], precision=threshold_rp[1])
        rp = np.concatenate([rp, threshold_rp.reshape(1,2)], axis=0)
    # [N, 2] -> [2, N] 
    rp = rp.T
    indices = np.argsort(rp[0])
    rp = rp[:, indices]
    ds = ds[indices]
    return rp, ds

def get_hetero_recall_precision(
    dataset:LPRDataset, 
    distance:np.ndarray,
    savepath:str=None,
    num_eval:int=100,
    show:bool=False
):
    
    assert len(dataset) == distance.shape[0], "Evaluate: len(datasets) == embeddings.shape[0]"
    Nm = distance.shape[0]
    distance = distance.copy() + np.eye(Nm)*(np.max(distance)+0.01)

    all_rp = {}
    
    geneous_names = dataset.get_geneous_names()
    for sgid, source in enumerate(geneous_names):
        sgndx_all = dataset.get_homoindices(sgid)
        for tgid, target in enumerate(geneous_names):
            tgndx_all = dataset.get_homoindices(tgid)
            st = "{}-{}".format(source, target)
            # print(st)
            if sgndx_all.shape[0] == 0 or tgndx_all.shape[0] == 0:
                print("no instance in source or target, continue")
                continue
            # recall-pricision
            all_rp[st] = {}
            all_rp[st]["xy"], all_rp[st]["ds"] = get_recall_precision_curve(dataset, distance, sgndx_all, tgndx_all, num_eval)
    # print("all-all")
    all_rp["all-all"] = {}
    all_rp["all-all"]["xy"], all_rp["all-all"]["ds"] = get_recall_precision_curve(dataset, distance, dataset.get_indices(), dataset.get_indices(), num_eval)
    
    plt.figure(figsize=(20,20))
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid()
    for st in all_rp:
        plt.plot(all_rp[st]["xy"][0], all_rp[st]["xy"][1])
        for i, d in enumerate(all_rp[st]["ds"]):
            plt.annotate(
                text="%.2f"%d, 
                xy=(all_rp[st]["xy"][0][i], all_rp[st]["xy"][1][i]),
                xytext=(all_rp[st]["xy"][0][i], all_rp[st]["xy"][1][i]),
                fontsize=10,
            )
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(list(all_rp))
    if savepath is not None: plt.savefig(os.path.join(savepath, "recall-precision.png"))
    elif show: plt.show()
    else: plt.close()
    return all_rp


def show_closest(dataset:LPRDataset, distance:np.ndarray):
    print("Show Closest Submaps")

    geneous_names = dataset.get_geneous_names()
    for sgid, source in enumerate(geneous_names):
        sgndx_all = dataset.get_homoindices(sgid)
        for tgid, target in enumerate(geneous_names):
            if source == target: continue

            tgndx_all = dataset.get_homoindices(tgid)
            for _ in range(10):
                d = 4.0
                anchor, closest = None, None
                while d > 2.0:
                    anchor = random.choice(sgndx_all)
                    closest = tgndx_all[np.argsort(distance[anchor, tgndx_all])][0]
                    d = distance[anchor][closest]
                    

                suc = "False"
                if closest in dataset.get_positives(anchor): suc = "True"
                
                anchor_pcd = o3d.geometry.PointCloud()
                anchor_pcd.points = o3d.utility.Vector3dVector(dataset.get_pc(anchor) - np.asarray([40,0,0]))
                
                closest_pcd = o3d.geometry.PointCloud()
                closest_pcd.points = o3d.utility.Vector3dVector(dataset.get_pc(closest) + np.asarray([40,0,0]))

                o3d.visualization.draw_geometries(
                    [anchor_pcd, closest_pcd], 
                    window_name="%s-%s: result=%s, distance=%.3f"%(source, target, suc, d)
                )
