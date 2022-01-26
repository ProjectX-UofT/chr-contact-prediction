import functools
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage

from src.akita.layers import (
    AverageTo2D,
    ConcatDist2D,
    Conv1dBlock,
    Conv2dBlock,
    DilatedResConv2dBlock,
    TransformerEncoder
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


class Trunk(nn.Module):

    def __init__(self, n_layer, n_head, n_inner, dropout):
        super().__init__()

        transformer = TransformerEncoder(
            n_embd=96,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            dropout=dropout
        )

        modules = [Conv1dBlock(4, 96, 11, pool_size=2)]
        modules += [Conv1dBlock(96, 96, 5, pool_size=2) for _ in range(10)]
        modules += [transformer, Conv1dBlock(96, 64, 5)]
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
        for _ in range(3):
            modules.append(DilatedResConv2dBlock(48, 24, 48, 3, round(dilation), 0.1, True))
            dilation *= 4.0
        self.head = nn.Sequential(*modules)

    def forward(self, z):
        z = self.concat_dist(self.one_to_two(z))
        z = torch.permute(z, [0, 3, 1, 2])
        y = self.head(z)
        return torch.permute(y, [0, 2, 3, 1])


class ContactPredictor(nn.Module):

    def __init__(self, n_layer, n_head, n_inner, dropout):
        super().__init__()
        self.trunk = Trunk(n_layer, n_head, n_inner, dropout)
        self.head = HeadHIC()
        self.fc_out = nn.Linear(48, 5)

        self.target_width = 448
        self.diagonal_offset = 2
        self.triu_idxs = torch.triu_indices(448, 448, 2)

    def forward(self, input_seqs, flatten=False):
        z = self.trunk(input_seqs)
        y = self.head(z)
        if flatten:
            y = y[:, 32:-32, 32:-32, :]
            y = y[:, self.triu_idxs[0], self.triu_idxs[1], :]
        return self.fc_out(y)


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_layer', type=int, default=5)
        parser.add_argument('--n_head', type=int, default=6)
        parser.add_argument('--n_inner', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.2)

        parser.add_argument('--augment_rc', type=int, default=1)
        parser.add_argument('--augment_shift', type=int, default=11)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--ema_decay', type=float, default=0.995)
        return parser

    def __init__(
            self,
            n_layer, n_head, n_inner, dropout,
            augment_rc, augment_shift, lr, ema_decay,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = ContactPredictor(n_layer, n_head, n_inner, dropout)
        self.augment_rc = augment_rc
        self.augment_shift = augment_shift

        self.lr = lr
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)

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

    def test_step(self, batch, batch_idx):
        loss, batch_size = self._process_batch(batch)
        self.log('test_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model.parameters())

    def _stochastic_augment(self, batch):
        take_rc = bool(random.getrandbits(1))
        shift = random.randrange(-self.augment_shift, self.augment_shift + 1)

        seqs, tgts = batch
        if self.augment_rc and take_rc:
            seqs = reverse_complement(seqs)
            tgts = reverse_triu(tgts, self.model.target_width, self.model.diagonal_offset)
        seqs = shift_pad(seqs, shift, pad=0.25)
        return seqs, tgts

    def _process_batch(self, batch):
        seqs, tgts = batch
        batch_size = tgts.shape[0]
        preds = self(seqs)
        loss = F.mse_loss(preds, tgts)
        return loss, batch_size
