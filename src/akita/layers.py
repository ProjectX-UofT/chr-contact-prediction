import torch.nn as nn
import torch


class AverageTo2D(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = width

        dists = torch.tensor([[[[abs(i - j)] for i in range(width)] for j in range(width)]])
        self.dist_matrix = dists.float() / (width - 1)

    def forward(self, z):
        b, w, d = z.shape
        z_2d = z.tile([1, w, 1])
        z_2d = z_2d.reshape(b, w, w, d)
        z_2d = (z_2d + z_2d.transpose(1, 2)) / 2

        d = self.dist_matrix.tile([z.shape[0], 1, 1, 1])
        return torch.cat([z_2d, d], dim=-1)
