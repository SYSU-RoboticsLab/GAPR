import torch
import os
import copy
import yaml
from typing import Dict, Any

class LPRModel:
    """
    # Wrapper for models
    """ 
    def __init__(self):
        return

    def construct(self, name:str, **kw):
        self.config = copy.deepcopy(kw)
        self.config["name"] = name
        self.name = name
        self.model = None
        if self.name == "GAPR":
            from models.gapr import GAPR
            self.model = GAPR(**kw)
        else:
            raise NotImplementedError("LPRModel: model %s not implemented" % self.name)

    def __call__(self, data:Dict[str, Any]) -> Dict[str, Any]:
        output:Dict[str, Any] = {}
        if self.name == "GAPR":
            assert set(["coords", "feats", "geneous"]) <= set(data.keys())
            output["coords"], output["feats"], output["scores"], output["embeddings"] = self.model(data["coords"], data["feats"],  data["geneous"])
        else:
            raise NotImplementedError("LPRModel: model %s not implemented" % self.name)
        return output
    
    def save(self, path:str):
        pth_file_dict = {"config":self.config, "weight": self.model.module.state_dict()}
        torch.save(pth_file_dict, path)
        return

    def load(self, path, device):
        pth_file_dict = torch.load(path, map_location=device)
        print("LPRModel: load\n", pth_file_dict["config"])
        self.construct(**pth_file_dict["config"])
        self.model.load_state_dict(pth_file_dict["weight"])
        return
    

    def import_and_save(self, config_path: str, weight_path:str, save_path:str):
        """
        import models from other project and save it
        """
        assert (os.path.exists(config_path)) and (os.path.exists(weight_path)) and (not os.path.exists(save_path))
        pth_file_dict = {}
        # load weights
        pth_file_dict["weight"] = torch.load(weight_path)
        # load config
        f = open(config_path, encoding="utf-8")
        pth_file_dict["config"] = yaml.load(f, Loader=yaml.FullLoader) #读取yaml文件
        f.close()
        torch.save(pth_file_dict, save_path)
        return
