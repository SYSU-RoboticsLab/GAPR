from typing import Any, Dict
from misc.utils import tensors2numbers
from torch.utils.tensorboard import SummaryWriter

class LPRLoss:
    """
    # Wrapper loss function 
    """
    def __init__(self, name:str, **kw):
        self.name = name
        if self.name == "GAPRLoss":
            from loss.gapr import GAPRLoss
            self.loss_fn = GAPRLoss(**kw)
        else:
            raise NotImplementedError("LPRLoss: loss_fn %s not implemented" % self.name)

    def __call__(self, output:Dict[str, Any], mask:Dict[str, Any]):
        loss, stats = None, None
        if self.name == "GAPRLoss":
            assert set(["embeddings", "coords", "feats", "scores"]) <= set(output.keys())
            assert set(["positives", "negatives", "rotms", "trans", "geneous"]) <= set(mask.keys())
            loss, stats = self.loss_fn(
                output["embeddings"], 
                output["coords"], 
                output["feats"], 
                output["scores"], 
                mask["rotms"],
                mask["trans"],
                mask["positives"], 
                mask["negatives"],
                mask["geneous"]
            ) 
        else:
            raise NotImplementedError("LPRLoss: loss_fn %s not implemented" % self.name)
        
        assert loss is not None and stats is not None
        stats = tensors2numbers(stats)
        return loss, stats
    
    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        """
        # visualize stats
        """
        if self.name == "GAPRLoss":
            self.loss_fn.print_stats(epoch, phase, writer, stats)
        else:
            raise NotImplementedError("LPRLoss: loss_fn %s.print_stats() not implemented" % self.name)