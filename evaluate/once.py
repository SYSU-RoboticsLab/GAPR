import os
import torch
import argparse
import numpy as np
import yaml
from datasets.dataloders.lprdataloader import LPRDataLoader
from models.lprmodel import LPRModel
from evaluate.utils import get_embeddings, get_hetero_topN_recall, get_hetero_recall_precision, show_closest

from tqdm import tqdm
from misc.utils import get_datetime
import matplotlib.pyplot as plt

def parse_opt()->dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--yaml",    type=str, required=True)
    parser.add_argument("--tn",      type=int, default=30)
    parser.add_argument("--rp",      type=int, default=100)
    parser.add_argument("--save",    type=str, default="results/evaluate/")
    opt = parser.parse_args()
    opt = vars(opt)
    f = open(opt["yaml"], encoding="utf-8")
    lpreval = yaml.load(f, Loader=yaml.FullLoader)
    
    lpreval.update(opt)
    return lpreval


def feat_l2d_mat(embeddings: np.ndarray) -> np.ndarray:
    Nm, Fs = embeddings.shape
    distance = np.linalg.norm(embeddings.reshape((Nm, 1, Fs)) - embeddings.reshape((1, Nm, Fs)), axis=2)
    distance += np.eye(Nm)*(np.max(distance)+1) # eye
    return distance


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

    # check savepath
    savepath = None
    if kw["save"] is not None:
        assert os.path.exists(kw["save"]), "Path does not exist, please run: mkdir " + kw["save"]
        savepath = os.path.join(kw["save"], get_datetime())
        os.mkdir(savepath)
    print("Save path:", savepath)

    # recall-precision
    if kw["rp"] < 1:
        print("Evaluation of Recall-Precision: Skip.")
    else:
        print("Evaluation of Recall-Precision: %d steps." % kw["rp"])
        distance = feat_l2d_mat(get_embeddings(lprmodel, dataloader, device, print_stats=False))
        rp = get_hetero_recall_precision(dataloader.dataset, distance, num_eval=kw["rp"])
        plt.figure()
        
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.grid()
        for st in rp:
            plt.plot(rp[st]["xy"][0], rp[st]["xy"][1])
            for i, d in enumerate(rp[st]["ds"]):
                plt.annotate(
                    text="%.2f"%d, 
                    xy=(rp[st]["xy"][0][i], rp[st]["xy"][1][i]),
                    xytext=(rp[st]["xy"][0][i], rp[st]["xy"][1][i]),
                    fontsize=10,
                )
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(list(rp))
        if savepath is not None: plt.savefig(os.path.join(savepath, "recall-precision.png"))
        plt.close()


    # average topN-recall
    if kw["tn"] < 1: 
        print("Evaluation of TopN-Recall: Skip.")
    else:
        topNs = []
        print("Evaluation of TopN-Recall: %d epochs, the average is taken." % kw["tn"])
        print("(The results are saved each epoch. Enter Ctrl+C to stop.)")
        iterator = tqdm(range(kw["tn"]))
        for _ in iterator:
            # get descriptor distance
            distance = feat_l2d_mat(get_embeddings(lprmodel, dataloader, device, print_stats=False))
            # append to all topN recall
            topNs.append(get_hetero_topN_recall(dataloader.dataset, distance))
            # take average values
            tn = {}
            for e in topNs[0]: tn[e] = np.stack([topN[e] for topN in topNs], axis=0).mean(axis=0)

            plt.figure()
            plt.grid()
            for e in tn: plt.plot(tn[e])
            plt.xlabel("TopN")
            plt.ylabel("Recall")
            plt.legend(list(tn))
            if savepath is not None: plt.savefig(os.path.join(savepath, "topN-recall.png"))
            plt.close()

            stats = "Top1-Recall: "
            for e in list(tn.keys()): stats += "%s:%.3f|" % (e, tn[e][0])
            iterator.set_postfix_str(stats)

    return


if __name__ == "__main__":
    main(**parse_opt())