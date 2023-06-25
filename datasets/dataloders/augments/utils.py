# Author:   Jacek Komorowski, https://github.com/jac99/MinkLoc3D
# Modified: Yingrui Jie,      https://github.com/SYSU-RoboticsLab/GAPR

import numpy as np
import math
import random
import torch


class RandomRotation:
    """
    # Random Rotate pointclouds and return matrix
    """
    def __init__(self, cmd:str):
        if cmd not in [None, "zxy10", "zxy20", "so3"]: raise NotImplementedError("RandomRotate: cmd in [None, zxy10, zxy20,so3]")
        self.cmd = cmd

    def getRotateMatrixFromRotateVector(self, axis:torch.Tensor, theta:torch.Tensor)->torch.Tensor:
        """
        # Get Rotate Matrix from Rotate Vector\n
        ## input \n
        axis.size()   == [bs, 3] \n
        theta.size()  == [bs] \n
        ## output \n
        rotateMatrix.size() == [bs, 3, 3] \n
        """
        device = axis.device
        bs = axis.size()[0]
        # [bs, 3]
        axis = axis / torch.norm_except_dim(v=axis, pow=2, dim=0)
        # [bs, 1, 3], [bs, 3, 1]
        axisH, axisV = axis.unsqueeze(1), axis.unsqueeze(2)
        # [bs, 1, 1]
        cosTheta = torch.cos(theta).reshape(-1, 1, 1)
        # [bs, 1, 1]
        sinTheta = torch.sin(theta).reshape(-1, 1, 1)
        # [bs, 3, 3]
        eye = torch.eye(3, device=device).expand(bs, 3, 3)
        # axis^ [bs, 3, 3]
        axisCaret = torch.cross(eye, axisH.expand(bs, 3, 3), dim=2)
        # so3: R = cos(theta) * I + (1-cos(theta)) * dot(a, aT) + sin(theta) * a^
        r = cosTheta * eye + (1.0-cosTheta) * torch.bmm(axisV, axisH) + sinTheta * axisCaret
        return r

    def __call__(self, coords:torch.Tensor):
        device = coords.device
        BS, PN, D = coords.shape
        # initial theta and axis
        theta, axis = torch.zeros((BS), device=device), torch.tensor([[0.0, 0.0, 1.0]] * BS, device=device)
        if self.cmd == "zxy10":
            # theta [-pi, pi]
            theta = torch.rand(BS,device=device) * 2 * np.pi - np.pi
            alpha = torch.rand(BS,device=device) * 2 * np.pi - np.pi
            beta  = torch.rand(BS,device=device) * np.pi * 10.0 / 180.0
            # alpha_axis is a vector in xOy plane
            alpha_axis = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros((BS), device=device)], dim=1)
            # print(alpha_axis, beta)
            alpha_mat = self.getRotateMatrixFromRotateVector(alpha_axis, beta)
            axis = torch.bmm(alpha_mat, axis.unsqueeze(2)).squeeze(2)
        elif self.cmd == "zxy20":
            # theta [-pi, pi]
            theta = torch.rand(BS,device=device) * 2 * np.pi - np.pi
            alpha = torch.rand(BS,device=device) * 2 * np.pi - np.pi
            beta  = torch.rand(BS,device=device) * np.pi * 20.0 / 180.0
            # alpha_axis is a vector in xOy plane
            alpha_axis = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros((BS), device=device)], dim=1)
            # print(alpha_axis, beta)
            alpha_mat = self.getRotateMatrixFromRotateVector(alpha_axis, beta)
            axis = torch.bmm(alpha_mat, axis.unsqueeze(2)).squeeze(2)
        elif self.cmd == "so3":
            theta = torch.rand(BS,device=device) * 2 * np.pi - np.pi
            axis  = torch.rand((BS, 3), device=device) - 0.5
            axis = axis / torch.norm_except_dim(axis, dim=1)
            
        rots_mat = self.getRotateMatrixFromRotateVector(axis, theta).type_as(coords)
        coords = torch.bmm(rots_mat, coords.transpose(1,2)).transpose(1, 2)
        return coords, rots_mat


class RandomTranslation:
    """
    # Random Translation
    """
    def __init__(self, delta=0.05):
        self.delta = delta

    def __call__(self, coords:torch.Tensor):
        BS, device = coords.shape[0], coords.device
        trans = self.delta * torch.randn(BS, 3, device=device)
        return coords + trans.unsqueeze(1), trans


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords

class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)

class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e:torch.Tensor):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e:torch.Tensor):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords:torch.Tensor):
        # Find point cloud 3D bounding box
        flattened_coords = coords.contiguous().view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords
