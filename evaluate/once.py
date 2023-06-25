import os
import torch
import argparse
import numpy as np
import yaml
from datasets.dataloders.lprdataloader import LPRDataLoader
from models.lprmodel import LPRModel
from evaluate.utils import get_embeddings, get_hetero_topN_recall, get_hetero_recall_precision, show_closest
import pickle
from tqdm import tqdm

def parse_opt()->dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--yaml",    type=str, required=True)
    parser.add_argument("--n_topn",  type=int, default=100)
    parser.add_argument("--n_rp",    type=int, default=100)
    parser.add_argument("--save",    type=str, default=None)
    opt = parser.parse_args()
    opt = vars(opt)
    f = open(opt["yaml"], encoding="utf-8")
    lpreval = yaml.load(f, Loader=yaml.FullLoader) #读取yaml文件
    
    lpreval.update(opt)
    return lpreval

def main(**kw):

    dataloader = LPRDataLoader(**kw["dataloaders"]["evaluate"])

    device:str = None
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print("Device: {}".format(device))
    assert os.path.exists(kw["weights"]), "Cannot open network weights: {}".format(kw["weights"])
    print("Loading weights: {}".format(kw["weights"]))

    lprmodel = LPRModel()
    lprmodel.load(kw["weights"], device)

    # get recall precision
    if kw["n_rp"] > 0:
        print("Hetero Recall-Precision: %d steps." % kw["n_rp"])
        embeddings = get_embeddings(lprmodel, dataloader, device, print_stats=False)
        assert len(dataloader.dataset) == embeddings.shape[0], "Evaluate: len(datasets) == embeddings.shape[0]"
        Nm, Fs = embeddings.shape
        distance = np.linalg.norm(embeddings.reshape((Nm, 1, Fs)) - embeddings.reshape((1, Nm, Fs)), axis=2)
        distance += np.eye(Nm)*(np.max(distance)+1)
        rp = get_hetero_recall_precision(dataloader.dataset, distance, num_eval=kw["n_rp"])
    else:
        print("Hetero Recall-Precision: Skip.")
        rp = None

    # get avg topN
    topNs = []
    print("Hetero TopN-Recall: %d epochs, the average is taken. \n(The results are saved each epoch. Enter Ctrl+C to stop.)" % kw["n_topn"])
    iterator = tqdm(range(kw["n_topn"]))
    for _ in iterator:
        embeddings = get_embeddings(lprmodel, dataloader, device, print_stats=False)
        # get feats distance
        assert len(dataloader.dataset) == embeddings.shape[0], "Evaluate: len(datasets) == embeddings.shape[0]"
        Nm, Fs = embeddings.shape
        # (Nm, Nm)
        distance = np.linalg.norm(embeddings.reshape((Nm, 1, Fs)) - embeddings.reshape((1, Nm, Fs)), axis=2)
        distance += np.eye(Nm)*(np.max(distance)+1) # eye

        topNs.append(get_hetero_topN_recall(dataloader.dataset, distance))

        # if kw["repr"]:
        # if kw["show"]: show_closest(dataloader.dataset, distance)
        avg_topN = {}
        for keys in topNs[0]: avg_topN[keys] = np.stack([topN[keys] for topN in topNs], axis=0).mean(axis=0)

        results = {"tn": avg_topN, "rp": rp}

        stats = "Top1-Recall: "
        for e in list(avg_topN.keys()): stats += "%s:%.3f|" % (e, avg_topN[e][0])

        if kw["save"] is not None:
            if not os.path.exists(kw["save"]): os.mkdir(kw["save"])
            with open(os.path.join(kw["save"], "config.yaml"), "w") as f:
                f.write(yaml.dump(dict(kw), allow_unicode=True))
            with open(os.path.join(kw["save"], "tnrp.pickle"), "wb") as f:
                pickle.dump(results, f) 
            with open(os.path.join(kw["save"], "top1.txt"), "w") as f:
                f.write(stats)


        iterator.set_postfix_str(stats)

    return


if __name__ == "__main__":
    main(**parse_opt())