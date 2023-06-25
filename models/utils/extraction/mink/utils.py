import torch
import MinkowskiEngine as ME

def minkowski_sparse(coords:torch.Tensor, feats:torch.Tensor, quant_size:float):
    device = coords.device
    # sparse_quantize
    quant_coords, quant_feats = [], []
    for coord, feat in zip(coords.clone().detach().cpu(), feats.clone().detach().cpu()):
        quant_coord, quant_feat = ME.utils.sparse_quantize(
            coordinates=coord, features=feat, quantization_size=quant_size
        )
        quant_coords.append(quant_coord)
        quant_feats.append(quant_feat)

    # batch collate
    batch_coords, batch_feats = ME.utils.sparse_collate(quant_coords, quant_feats)
    # to sparse tensor 
    sparse_tensor = ME.SparseTensor(features=batch_feats.to(device=device), coordinates=batch_coords.to(device=device))
    return sparse_tensor

def minkowski_decomposed(sparse_tensor, quant_size):
    coords, feats = sparse_tensor.decomposed_coordinates_and_features
    # de-quantize coordinates
    a = torch.tensor([2])
    coords = [e.double()*quant_size for e in coords]
    return coords, feats
