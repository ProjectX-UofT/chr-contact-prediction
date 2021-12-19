import functools
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn


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

        # TODO: add layers here

    def forward(self, input_seqs):
        L, D = self.seq_length, self.seq_depth
        W, C = self.target_width, self.num_targets

        # TODO: replace placeholder with stuff
        assert input_seqs.shape[1:] == (L, D)
        contacts = torch.zeros((1, 99681, 5))
        assert contacts.shape[1:] == (W, W, C)

        return contacts


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
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

    def forward(self, input_seqs):
        return self.model(input_seqs)

    def training_step(self, batch, batch_idx):
        batch = self._stochastic_shift_and_rc(batch)

        loss = None
        self.log('val_loss', loss)

    def validation_step(self, batch, batch_idx):
        # TODO: compute loss and other metrics
        loss = None
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _stochastic_shift_and_rc(self, batch):
        seq, tgt = batch

        return batch
