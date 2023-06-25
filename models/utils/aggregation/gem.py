import torch
import torch.nn as nn

class GeM(nn.Module):
    def __init__(self, pn=256, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = nn.AvgPool1d(pn) # pn = 256
    def forward(self, x:torch.Tensor):
        temp = x.clamp(min=self.eps).pow(self.p)
        temp = self.f(temp)
        temp = temp.pow(1./self.p)
        # 防止把第一维压缩掉
        temp = temp.squeeze(dim=2)
        return temp

class MeanGeM(nn.Module):
    def __init__(self, p:float, eps:float):
        # p=3, eps=0.0000001
        super(MeanGeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x:torch.Tensor):
        # x: [pn, fs]
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=0)
        x = x.pow(1./self.p)
        return x