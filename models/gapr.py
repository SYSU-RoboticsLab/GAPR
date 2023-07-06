import torch
import torch.nn as nn
from typing import Dict

from models.utils.aggregation.gem import MeanGeM
from models.utils.extraction.mink.minkfpn import MinkFPN
from models.utils.transformers.transgeo import PCTrans

class GAPR(nn.Module):
    def __init__(self, minkfpn:Dict, pctrans:Dict, meangem:Dict, **kw):
        super(GAPR, self).__init__()
        print("Model: GAPR")
        self.minkfpn = MinkFPN(**minkfpn)
        self.geneous_names = ["ground", "aerial"]

        self.ground_trans = PCTrans(**pctrans)
        self.aerial_trans = PCTrans(**pctrans)

        self.meangem = MeanGeM(**meangem)

    
    def forward(self, coords:torch.Tensor, feats:torch.Tensor, geneous:torch.Tensor):
        BS = geneous.shape[0]
        cnn_coords, cnn_feats = self.minkfpn(coords, feats)
        attn_feats, attn_scores = [], []
        for ndx in range(BS): 
            if self.geneous_names[geneous[ndx].item()] == "ground":
                attn_feat, attn_score = self.ground_trans(cnn_feats[ndx].unsqueeze(0))
                attn_feats.append(attn_feat.squeeze(0))
                attn_scores.append(attn_score.squeeze(0))
            elif self.geneous_names[geneous[ndx].item()] == "aerial":
                attn_feat, attn_score = self.aerial_trans(cnn_feats[ndx].unsqueeze(0))
                attn_feats.append(attn_feat.squeeze(0))
                attn_scores.append(attn_score.squeeze(0))
            else: raise NotImplementedError
        
    
        batch_feats = torch.stack([self.meangem(feat) for feat in attn_feats], dim=0)
        # batch_feats = torch.stack([self.meangem(feat) for feat in cnn_feats], dim=0)
     
        return cnn_coords, cnn_feats, attn_scores, batch_feats 
