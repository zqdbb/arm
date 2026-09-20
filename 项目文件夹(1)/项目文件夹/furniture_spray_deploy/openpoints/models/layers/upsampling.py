from typing import List, Tuple
from torch.autograd import Function
import torch
import torch.nn as nn

class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        B, N, _ = unknown.size(); m = known.size(1)
        d = torch.cdist(unknown, known)
        top_d, top_i = d.topk(3, dim=2, largest=False)
        return top_d, top_i.int()
    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        B, c, m = features.size(); n = idx.size(1)
        idx = idx.clamp(0, m-1)
        output = torch.zeros(B, c, n, device=features.device)
        for b in range(B):
            for j in range(c):
                gathered = features[b, j][idx[b]]
                output[b, j] = (gathered * weight[b]).sum(1)
        ctx.save_for_backward(idx, weight, torch.tensor([B,c,m,n]))
        return output
    @staticmethod
    def backward(ctx, grad_out):
        idx, weight, dims = ctx.saved_tensors
        B, c, m, n = dims.tolist()
        grad_features = torch.zeros(B, c, m, device=grad_out.device)
        for b in range(B):
            for j in range(c):
                i = idx[b].clamp(0, m-1)
                grad_features[b, j].scatter_add_(0, i[:,0], grad_out[b,j]*weight[b,:,0])
                grad_features[b, j].scatter_add_(0, i[:,1], grad_out[b,j]*weight[b,:,1])
                grad_features[b, j].scatter_add_(0, i[:,2], grad_out[b,j]*weight[b,:,2])
        return grad_features, None, None

three_interpolate = ThreeInterpolate.apply

def three_nn_wrapper(B, N, m, unknown, known, dist2, idx):
    d = torch.cdist(unknown, known)
    top_d, top_i = d.topk(3, dim=2, largest=False)
    dist2.copy_(top_d); idx.copy_(top_i.int())

def three_interpolation(unknown_xyz, known_xyz, know_feat):
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats
