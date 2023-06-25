# Author:   Sijie Zhu,   https://github.com/Jeff-Zilence/TransGeo2022
# Modified: Yingrui Jie, https://github.com/SYSU-RoboticsLab/GAPR

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
class PCTrans(nn.Module):
    def __init__(self, 
        dim:int, 
        num_heads:int, 
        mlp_ratio:int, 
        depth:int,  
        qkv_bias:bool, 
        init_values:float, 
        drop:float,
        attn_drop:float,
        drop_path_rate:float
    ):
        super().__init__()
        assert depth >= 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            )
            for i in range(depth)])
        self.norm = partial(nn.LayerNorm, eps=1e-6)(dim) 
        self.num_heads = float(num_heads)

    def forward(self, x:torch.Tensor):
        attn_score = None

        for i, blk in enumerate(self.blocks):
            attn_x = blk.norm1(x)
            if i == len(self.blocks)-1:
                # decompose attn forward
                B, N, C = attn_x.shape
                qkv = blk.attn.qkv(attn_x).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)

                # get attn_score
                attn_score = attn.sum(axis=1).sum(axis=1) - self.num_heads
                attn_score = torch.sigmoid(attn_score)

                attn = blk.attn.attn_drop(attn)

                attn_x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                attn_x = blk.attn.proj(attn_x)
                attn_x = blk.attn.proj_drop(attn_x)
            else:
                attn_x = blk.attn(attn_x)

            x = x + blk.drop_path1(blk.ls1(attn_x))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        x = self.norm(x)  
        return x, attn_score

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def main():
    BS, PN, FS = 1, 2342, 256
    # model = deit_small_distilled_patch16_224(save="/home/jieyr/code/TransGeo2022/save")
    model = PCTrans(
        dim=256,
        num_heads=8, 
        mlp_ratio=4, 
        qkv_bias=True, 
        depth=4,  
        init_values=None, 
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0
    )
    feats = torch.rand((BS, PN, FS))
    attn_feats, attn_score = model(feats)
    print(attn_feats.size(), attn_score)

if __name__ == "__main__":
    main()