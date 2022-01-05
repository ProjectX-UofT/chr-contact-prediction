import functools
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ContactPredictor(nn.Module):

    def __init__(
            self,
            seq_length: int = 1048576,
            seq_depth: int = 4,
            target_width: int = 512,
            num_targets: int = 5
    ):
        super().__init__()

        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.target_width = target_width
        self.num_targets = num_targets

        trunk = list()
        self._add_conv_block(trunk, 4, 96, 11, 2)
        for _ in range(10):
            self._add_conv_block(trunk, 96, 96, 5, 2)
        # TODO: in progress...
        self.trunk = nn.Sequential(*trunk)

    def _add_conv_block(self, trunk, in_channels, out_channels, kernel_size, pool_size):
        conv_padding = (kernel_size - 1) // 2  # padding needed to maintain same size
        return trunk.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=conv_padding),
            nn.BatchNorm1d(out_channels, momentum=0.01),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.ReLU()
        ])

    def forward(self, input_seqs):
        L, D = self.seq_length, self.seq_depth
        W, C = self.target_width, self.num_targets
        assert input_seqs.shape[1:] == (L, D)

        input_seqs = input_seqs.transpose(1, 2)
        x = self.trunk(input_seqs)
        print(x.shape)
        exit()

        return torch.zeros(input_seqs.shape[0], W, W, C, requires_grad=True)


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--augment_shift', type=int, default=11)
        parser.add_argument('--augment_rc', type=int, default=1)
        parser.add_argument('--target_crop', type=int, default=32)
        parser.add_argument('--diagonal_offset', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser

    def __init__(
            self,
            model: ContactPredictor,
            augment_shift: int,
            augment_rc: bool,
            target_crop: int,
            diagonal_offset: int,
            lr: float
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.augment_shift = augment_shift
        self.augment_rc = augment_rc
        self.target_crop = target_crop
        self.diagonal_offset = diagonal_offset
        self.lr = lr

        self.out_width = model.target_width - 2 * target_crop
        self.triu_idxs = torch.triu_indices(self.out_width, self.out_width, diagonal_offset)

    def forward(self, input_seqs):
        contacts = self.model(input_seqs)
        boarder = self.target_crop
        contacts = contacts[:, boarder:-boarder, boarder:-boarder, :]
        return contacts[:, self.triu_idxs[0], self.triu_idxs[1], :]

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
        loss = F.mse_loss(preds, tgts)
        batch_size = tgts.shape[0] * tgts.shape[2]
        return loss, batch_size
