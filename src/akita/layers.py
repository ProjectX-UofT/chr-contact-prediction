import torch
import torch.nn as nn


class AverageTo2D(nn.Module):

    def forward(self, z):
        b, w, d = z.shape
        z_2d = z.tile([1, w, 1])
        z_2d = z_2d.reshape(b, w, w, d)
        return (z_2d + z_2d.transpose(1, 2)) / 2


class ConcatDist2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist_matrix = torch.zeros(1, 1, 1, 1)

    def _cache_dist_matrix(self, width):
        if self.dist_matrix.shape[1] == width:
            return
        dists = torch.tensor([[[[abs(i - j)] for i in range(width)] for j in range(width)]])
        self.dist_matrix = dists.float() / (width - 1)

    def forward(self, z_2d):
        self._cache_dist_matrix(z_2d.shape[1])
        d = self.dist_matrix.tile([z_2d.shape[0], 1, 1, 1])
        return torch.cat([z_2d, d], dim=-1)


class Symmetrize2D(nn.Module):

    def forward(self, x):
        return (x + x.transpose(1, 2)) / 2
