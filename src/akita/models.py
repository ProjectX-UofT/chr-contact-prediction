import functools
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.akita.layers import (
    AverageTo2D,
    ConcatDist2D,
    Conv1dBlock,
    Conv2dBlock,
    DilatedResConv2dBlock
)


# =============================================================================
# Data Transforms
# =============================================================================


def shift_pad(seqs, shift, pad=0.25):
    assert len(seqs.shape) == 3
    if shift == 0:
        return seqs
    seqs = torch.roll(seqs, shifts=shift, dims=1)
    if shift > 0:
        seqs[:, :shift, :] = pad
    else:
        seqs[:, shift:, :] = pad
    return seqs


def reverse_complement(seqs):
    assert len(seqs.shape) == 3
    # flipping dim=1 reverses
    # flipping dim=2 is a hack to perform complement
    return torch.flip(seqs, dims=[1, 2])


@functools.lru_cache()
def _reverse_triu_index_perm(width, offset):
    assert offset > 0
    triu_idxs = torch.triu_indices(width, width, offset)

    mat = torch.zeros(width, width, dtype=torch.long)
    mat[triu_idxs[0], triu_idxs[1]] = torch.arange(triu_idxs.shape[1])
    mat = mat + mat.T

    rev_mat = torch.flip(mat, dims=[0, 1])
    return rev_mat[triu_idxs[0], triu_idxs[1]]


def reverse_triu(trius, width, offset):
    assert offset > 0
    assert len(trius.shape) == 3

    perm = _reverse_triu_index_perm(width, offset)
    assert len(perm) == trius.shape[1]

    return trius[:, perm, :]


# =============================================================================
# Models
# =============================================================================
class VariationalLayer(nn.Module):

    def __init__(self, use_variational_obj=True, VQ_mode=False, input_dim=0,
                 output_dim=0):
        super().__init__()

        self.use_var_obj = use_variational_obj
        self.VQ_mode = VQ_mode  # TODO: finish VQ-VAE implementation
        self.input_dims = input_dim
        self.output_dims = output_dim

        self.mu = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)

    def reparam(self, mu, logvar):
        """
        Suppose we have p(z|x) \approx q(z|x), then
        We want to compute z ~ q(z|x) = mu + N(0, 1) * sigma
        :param mu: the learned mean vector of the posterior distn
        :param logvar: the learned log variance vector of the posterior distn
        :return: a sample from the approximated posterior distn (q(z|x))
        """
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn(size=(mu.shape(0), mu.shape(1)))
        return mu + sigma * eps

    def forward(self, encoder_output):
        if not self.use_var_obj:
            return encoder_output

        mu = self.mu(encoder_output)
        log_var = self.log_var(encoder_output)
        sample_z = self.reparam(mu, log_var)

        return sample_z, mu, log_var


class Trunk(nn.Module):

    def __init__(self):
        super().__init__()

        modules = [Conv1dBlock(4, 96, 11, pool_size=2)]
        modules += [Conv1dBlock(96, 96, 5, pool_size=2) for _ in range(10)]
        modules += []  # TODO: add transformer layer
        modules += [Conv1dBlock(96, 64, 5)]
        self.trunk = nn.Sequential(*modules)

    def forward(self, input_seqs):
        input_seqs = input_seqs.transpose(1, 2)
        return self.trunk(input_seqs).transpose(1, 2)


class HeadHIC(nn.Module):

    def __init__(self):
        super().__init__()
        self.one_to_two = AverageTo2D()
        self.concat_dist = ConcatDist2D()

        modules = [Conv2dBlock(65, 48, 3, symmetrize=True)]
        dilation = 1.0
        for _ in range(6):
            modules.append(
                DilatedResConv2dBlock(48, 24, 48, 3, round(dilation), 0.1,
                                      True))
            dilation *= 1.75
        self.head = nn.Sequential(*modules)

    def forward(self, z):
        z = self.concat_dist(self.one_to_two(z))
        z = torch.permute(z, [0, 3, 1, 2])
        y = self.head(z)
        return torch.permute(y, [0, 2, 3, 1])


class ContactPredictor(nn.Module):

    def __init__(self,
                 variational: bool = False
                 ):
        super().__init__()

        self.variational = variational

        self.trunk = Trunk()
        self.head = HeadHIC()
        self.fc_out = nn.Linear(48, 5)
        self.variational_layer = VariationalLayer(variational, input_dim=64,
                                                  output_dim=64)

        # target_crop = 32
        # diagonal_offset = 2
        self.triu_idxs = torch.triu_indices(448, 448, 2)

    def forward(self, input_seqs, flatten=False):
        z = self.trunk(input_seqs)
        if self.variational:
            z, mu, log_var = self.variational_layer(z)
        y = self.head(z)
        if flatten:
            y = y[:, 32:-32, 32:-32, :]
            y = y[:, self.triu_idxs[0], self.triu_idxs[1], :]
        return self.fc_out(y) if not self.variational else self.fc_out(
            y), mu, log_var


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--augment_shift', type=int, default=11)
        parser.add_argument('--augment_rc', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--variational', type=bool, default=False)
        return parser

    def __init__(
            self,
            model: ContactPredictor,
            augment_shift: int,
            augment_rc: bool,
            lr: float,
            variational: bool
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.augment_shift = augment_shift
        self.augment_rc = augment_rc
        self.lr = lr
        self.variational = variational

    def forward(self, input_seqs):
        return self.model(input_seqs, flatten=True)

    def training_step(self, batch, batch_idx):
        batch = self._stochastic_augment(batch)
        loss, batch_size = self._process_batch(batch)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self._process_batch(batch)
        self.log('val_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _stochastic_augment(self, batch):
        take_rc = bool(random.getrandbits(1))
        shift = random.randrange(-self.augment_shift, self.augment_shift + 1)

        seqs, tgts = batch
        if self.augment_rc and take_rc:
            seqs = reverse_complement(seqs)
            tgts = reverse_triu(tgts, self.out_width, self.diagonal_offset)
        seqs = shift_pad(seqs, shift, pad=0.25)
        return seqs, tgts

    def _process_batch(self, batch):
        seqs, tgts = batch
        preds = self(seqs)

        if self.variational:
            return self._VAE_loss(preds, tgts)

        loss = F.mse_loss(preds, tgts)
        batch_size = tgts.shape[0] * tgts.shape[2]
        return loss, batch_size

    def _VAE_loss(self, preds, tgts, distributions_method=False):
        # unpack preds; will be a tuple in the case of VAE
        y_hat, mu_q, log_var_q = preds

        MSELoss_criterion = nn.MSELoss()
        MSE_loss = MSELoss_criterion(y_hat, tgts)

        # Use pytorch distributions functions to compute KL Divergence
        if distributions_method:
            prior_distribution = torch.distributions.normal.\
                Normal(torch.zeros(mu_q.shape(0)), torch.eye(mu_q.shape(0)))
            variational_posterior = torch.distributions.normal.Normal(mu_q, log_var_q)

            KLDiv_loss = torch.distributions.kl.kl_divergence(variational_posterior, prior_distribution)
            return (MSE_loss + KLDiv_loss).mean()

        # Analytically derived loss function for the KL Divergence:
        # We want to calculate -D_KL[q(z|x) || p(z)]
        # KL divergence with a normal prior and normal posterior is given as:
        # log(sigma_q / sigma_p) - (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma^2_p) + 0.5
        # Equiv derivation here: https://jaketae.github.io/study/vae/
        prior_mean, prior_variance = 0, 1
        KLDiv_loss = torch.sum(0.5 * log_var_q - 0.5 * math.log(prior_variance) -
                               (torch.exp(log_var_q) +
                                (mu_q - prior_mean) ** 2) /
                               (2 * prior_variance) + 0.5)

        return (MSE_loss + KLDiv_loss).mean()
